-- Weak labellers: who is listening, and who is clicking for the milestones.
--
-- The core measure is whether a person's answers depend on the audio. For every first answer on a clip with a
-- reference (a control clip's truth, or the labels at least half of >= 2 other people chose), compare how well it
-- matches its own clip's reference with how well it matches the references of other clips the same person answered:
--
--   follow = (matched - unmatched) / (1 - how alike the references are to each other)
--
-- 1 = the answers track what is in each clip; 0 = they would fit any clip just as well. A random clicker and someone who
-- ticks "whispering" on everything both land near 0, however whisper-heavy the queue is, which plain agreement (what
-- trust measures) cannot tell. Behaviour adds flags: rushing (answers under quality_rush_s seconds after the clip was
-- served), not listening (median playback under 3 s), robotic timing (answer times that barely vary), the same label
-- every time, inconsistency on hidden repeats, and mostly "can't tell".
--
--   too early  fewer than quality_min_refs answers with a reference, and no flag yet
--   ok         answers follow the audio, no flags
--   watch      one flag: served more control clips and hidden repeats, so the evidence comes in faster
--   weak       answers do not follow the audio (confidently), or two or more flags
--
-- A weak labeller's answers stop counting toward clip consensus, the people ring's milestone and the exported labels
-- (pull_labels.py --keep-weak keeps them), and with the 'hide_weak_from_board' setting at 1 they leave the leaderboard.
-- Nothing here is shown to labellers. admin_labeller_quality() is the review list.

create table private.labeller_quality (
  user_id uuid primary key references public.profiles (id) on delete cascade,
  answers integer not null default 0,          -- first answers that named sounds
  refs integer not null default 0,             -- of those, on clips with a reference
  matched real, unmatched real, ref_similarity real,
  follow real, follow_se real,
  retests integer not null default 0, retest_agreement real,
  median_answer_s real, rush_rate real, timing_cv real, median_listen_ms real,
  top_label text, top_label_share real, unsure_rate real, skip_rate real,
  flags text[] not null default '{}',
  status text not null default 'too early' check (status in ('too early', 'ok', 'watch', 'weak')),
  computed_at timestamptz not null default now()
);
revoke all on private.labeller_quality from public, anon, authenticated;

create function private.is_weak(p_user uuid)
returns boolean
language sql stable security definer set search_path = '' as $$
  select exists (select 1 from private.labeller_quality q where q.user_id = p_user and q.status = 'weak');
$$;

create function private.refresh_quality()
returns void
language plpgsql security definer set search_path = '' as $$
declare
  min_refs integer := private.setting('quality_min_refs', 15)::integer;
  weak_follow numeric := private.setting('quality_weak_follow', 0.1);
  watch_follow numeric := private.setting('quality_watch_follow', 0.25);
  rush_s numeric := private.setting('quality_rush_s', 4);
  shifts integer := private.setting('quality_shifts', 20)::integer;
  window_n integer := private.setting('quality_window', 400)::integer;
begin
  with prev_weak as (
    select user_id from private.labeller_quality where status = 'weak'
  ), first as (
    select l.user_id, l.clip_id, l.labels, l.created_at
    from public.labels l where l.attempt = 1 and not l.unsure
  ), ctrl as (
    -- a control clip's truth: the labels at least half of the people who labelled it in the Ear Check kits gave
    select s.clip_id, coalesce(array(
             select x from jsonb_each(s.info -> 'people') e(who, v), jsonb_array_elements_text(e.v) x
             group by x having 2 * count(*) >= (select count(*) from jsonb_object_keys(s.info -> 'people'))), '{}') as ref
    from private.clip_sources s
    where s.kind = 'control' and jsonb_typeof(s.info -> 'people') = 'object'
  ), others as (
    select f.user_id, f.clip_id,
           count(o.user_id) as n_others
    from first f join first o on o.clip_id = f.clip_id and o.user_id <> f.user_id
    where o.user_id not in (select user_id from prev_weak)
    group by f.user_id, f.clip_id
  ), peer_ref as (
    select f.user_id, f.clip_id, coalesce(array_agg(x) filter (where x is not null), '{}') as ref
    from first f
    join others n on n.user_id = f.user_id and n.clip_id = f.clip_id and n.n_others >= 2
    left join lateral (
      select x from first o, unnest(o.labels) x
      where o.clip_id = f.clip_id and o.user_id <> f.user_id and o.user_id not in (select user_id from prev_weak)
      group by x having 2 * count(*) >= n.n_others
    ) m on true
    group by f.user_id, f.clip_id
  ), refd as (
    select f.user_id, f.clip_id, f.labels, coalesce(c.ref, p.ref) as ref,
           row_number() over (partition by f.user_id order by f.created_at desc) as recent
    from first f
    left join ctrl c on c.clip_id = f.clip_id
    left join peer_ref p on p.user_id = f.user_id and p.clip_id = f.clip_id
    where c.clip_id is not null or p.clip_id is not null
  ), usable as (
    select user_id, labels, ref, row_number() over (partition by user_id order by recent) as i,
           count(*) over (partition by user_id) as n
    from refd where recent <= window_n
  ), shifted as (
    -- each answer against the references of up to `shifts` other clips the same person answered (cyclic shifts)
    select u.user_id, u.i, private.jaccard(u.labels, v.ref) as ju, private.jaccard(u.ref, v.ref) as jr
    from usable u
    cross join lateral generate_series(1, least(u.n - 1, shifts)) k
    join usable v on v.user_id = u.user_id and v.i = ((u.i - 1 + k) % u.n) + 1
  ), per as (
    select u.user_id, u.i, private.jaccard(u.labels, u.ref) as jm, avg(s.ju) as ju, avg(s.jr) as jr
    from usable u join shifted s on s.user_id = u.user_id and s.i = u.i
    group by u.user_id, u.i, u.labels, u.ref
  ), fol as (
    select user_id, count(*)::integer as refs, avg(jm) as matched, avg(ju) as unmatched, avg(jr) as ref_sim,
           avg(jm - ju) / greatest(0.1, 1 - avg(jr)) as follow,
           coalesce(stddev_samp(jm - ju), 0) / sqrt(count(*)) / greatest(0.1, 1 - avg(jr)) as follow_se
    from per group by user_id
  ), timing as (
    select l.user_id,
           count(*) filter (where l.attempt = 1) as all_first,
           count(*) filter (where l.attempt = 1 and not l.unsure) as answers,
           avg((l.unsure)::int) filter (where l.attempt = 1) as unsure_rate,
           percentile_cont(0.5) within group (order by l.listen_ms) as med_listen,
           percentile_cont(0.5) within group (order by extract(epoch from l.created_at - a.assigned_at)) as med_answer_s,
           avg((extract(epoch from l.created_at - a.assigned_at) < rush_s)::int) as rush_rate,
           stddev_samp(extract(epoch from l.created_at - a.assigned_at))
             / nullif(avg(extract(epoch from l.created_at - a.assigned_at)), 0) as timing_cv
    from public.labels l
    left join public.assignments a on a.user_id = l.user_id and a.clip_id = l.clip_id and a.attempt = l.attempt
    group by l.user_id
  ), tops as (
    select distinct on (t.user_id) t.user_id, t.x as top_label, t.c::numeric / t.n as top_share
    from (select f.user_id, x, count(*) as c, (select count(*) from first f2 where f2.user_id = f.user_id) as n
          from first f, unnest(f.labels) x group by f.user_id, x) t
    order by t.user_id, t.c desc, t.x
  ), ret as (
    select a.user_id, count(*)::integer as retests, avg(private.jaccard(a.labels, b.labels)) as retest_j
    from public.labels a join public.labels b on b.user_id = a.user_id and b.clip_id = a.clip_id and b.attempt = 2
    where a.attempt = 1 and not a.unsure and not b.unsure
    group by a.user_id
  ), skips as (
    select user_id, avg((state = 'skipped')::int) as skip_rate
    from public.assignments where state in ('done', 'skipped') group by user_id
  ), joined as (
    select t.user_id, coalesce(t.answers, 0)::integer as answers, coalesce(f.refs, 0) as refs,
           f.matched, f.unmatched, f.ref_sim, f.follow, f.follow_se,
           coalesce(r.retests, 0) as retests, r.retest_j,
           t.med_answer_s, t.rush_rate, t.timing_cv, t.med_listen, tp.top_label, tp.top_share, t.unsure_rate, sk.skip_rate,
           t.all_first,
           array_remove(array[
             case when coalesce(f.refs, 0) >= min_refs and f.follow + 1.645 * f.follow_se < weak_follow
                  then 'answers do not follow the audio' end,
             case when coalesce(f.refs, 0) >= min_refs and f.follow < watch_follow
                       and not (f.follow + 1.645 * f.follow_se < weak_follow)
                  then 'answers barely follow the audio' end,
             case when t.answers >= 20 and t.rush_rate > 0.3 then 'rushing' end,
             case when t.answers >= 20 and t.med_listen < 3000 then 'not listening' end,
             case when t.answers >= 30 and t.timing_cv < 0.15 then 'robotic timing' end,
             case when t.answers >= 20 and tp.top_share > 0.85 then 'same label every time' end,
             case when coalesce(r.retests, 0) >= 5 and r.retest_j < 0.3 then 'inconsistent on repeats' end,
             case when t.all_first >= 20 and t.unsure_rate > 0.5 then 'mostly can''t tell' end
           ], null) as flags
    from timing t
    left join fol f on f.user_id = t.user_id
    left join tops tp on tp.user_id = t.user_id
    left join ret r on r.user_id = t.user_id
    left join skips sk on sk.user_id = t.user_id
    where exists (select 1 from public.profiles p where p.id = t.user_id)
  )
  insert into private.labeller_quality as q (user_id, answers, refs, matched, unmatched, ref_similarity, follow, follow_se,
                                             retests, retest_agreement, median_answer_s, rush_rate, timing_cv,
                                             median_listen_ms, top_label, top_label_share, unsure_rate, skip_rate, flags,
                                             status, computed_at)
  select j.user_id, j.answers, j.refs, j.matched, j.unmatched, j.ref_sim, j.follow, j.follow_se, j.retests, j.retest_j,
         j.med_answer_s, j.rush_rate, j.timing_cv, j.med_listen, j.top_label, j.top_share, j.unsure_rate, j.skip_rate,
         j.flags,
         case when 'answers do not follow the audio' = any (j.flags) or cardinality(j.flags) >= 2 then 'weak'
              when cardinality(j.flags) = 1 then 'watch'
              when j.refs < min_refs then 'too early'
              else 'ok' end,
         now()
  from joined j
  on conflict (user_id) do update set
    answers = excluded.answers, refs = excluded.refs, matched = excluded.matched, unmatched = excluded.unmatched,
    ref_similarity = excluded.ref_similarity, follow = excluded.follow, follow_se = excluded.follow_se,
    retests = excluded.retests, retest_agreement = excluded.retest_agreement, median_answer_s = excluded.median_answer_s,
    rush_rate = excluded.rush_rate, timing_cv = excluded.timing_cv, median_listen_ms = excluded.median_listen_ms,
    top_label = excluded.top_label, top_label_share = excluded.top_label_share, unsure_rate = excluded.unsure_rate,
    skip_rate = excluded.skip_rate, flags = excluded.flags, status = excluded.status, computed_at = excluded.computed_at;
end $$;
revoke all on function private.refresh_quality(), private.is_weak(uuid) from public, anon, authenticated;

-- the review list (service role): everyone who has answered, weakest first
create function public.admin_labeller_quality()
returns table (username text, guest boolean, status text, flags text[], answers integer, refs integer, follow real,
               follow_se real, matched real, unmatched real, retests integer, retest_agreement real,
               median_answer_s real, rush_rate real, timing_cv real, median_listen_ms real, top_label text,
               top_label_share real, unsure_rate real, skip_rate real, trust real, joined timestamptz,
               last_label timestamptz, computed_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select p.username, p.guest, q.status, q.flags, q.answers, q.refs, q.follow, q.follow_se, q.matched, q.unmatched,
         q.retests, q.retest_agreement, q.median_answer_s, q.rush_rate, q.timing_cv, q.median_listen_ms, q.top_label,
         q.top_label_share, q.unsure_rate, q.skip_rate, p.trust, p.created_at,
         (select max(l.created_at) from public.labels l where l.user_id = q.user_id), q.computed_at
  from private.labeller_quality q join public.profiles p on p.id = q.user_id
  order by array_position(array['weak', 'watch', 'too early', 'ok'], q.status), q.follow nulls first, q.answers desc;
$$;
revoke all on function public.admin_labeller_quality() from public, anon, authenticated;
grant execute on function public.admin_labeller_quality() to service_role;

-- weak labellers no longer count toward a clip's consensus
create or replace function private.clip_consensus(p_clip uuid)
returns real
language sql stable security definer set search_path = '' as $$
  with v as (
    select l.labels, p.trust as w
    from public.labels l join public.profiles p on p.id = l.user_id
    where l.clip_id = p_clip and l.attempt = 1 and not l.unsure
      and (p.trust_evidence < 20 or p.trust >= private.setting('trust_floor', 0.15))
      and not private.is_weak(l.user_id)
  ), tot as (select sum(w) as w, count(*) as n from v),
  per as (select x as label, sum(v.w) as w from v, unnest(v.labels) x group by x)
  select case when (select n from tot) < 2 or (select w from tot) <= 0 then null
              else coalesce((select avg(abs(2 * per.w / tot.w - 1)) from per, tot), 1)::real end;
$$;

-- ... nor toward the people ring's milestone
create or replace function private.refresh_progress()
returns void
language sql security definer set search_path = '' as $$
  with real as (
    select count(*) as items,
           count(*) filter (where exists (select 1 from public.labels l where l.clip_id = c.id and not l.unsure)
                              or exists (select 1 from public.imported_labels i where i.source_uid = s.source_uid)) as labelled
    from public.clips c join private.clip_sources s on s.clip_id = c.id
    where c.core
  ), seed as (
    select count(*) as labelled
    from public.imported_labels i join public.profiles p on p.id = i.user_id
    where p.synthetic and i.kit = 'synthetic'
  ), answers as (
    select (select count(*) from public.labels l where not l.unsure and not private.is_weak(l.user_id))
         + (select count(*) from public.imported_labels i join public.profiles p on p.id = i.user_id where not p.synthetic) as n
  )
  insert into private.progress_snapshot (id, foundation_items, foundation_human_labelled, corpus_recordings, ai_labelled_recordings,
                                         human_labels, computed_at)
  select true, r.items, least(r.items, r.labelled + (select labelled from seed)),
         coalesce((select value::bigint from private.settings where key = 'corpus_recordings'), 0),
         coalesce((select value::bigint from private.settings where key = 'ai_labelled_recordings'), 0),
         (select n from answers), now()
  from real r
  on conflict (id) do update set foundation_items = excluded.foundation_items,
                                 foundation_human_labelled = excluded.foundation_human_labelled,
                                 corpus_recordings = excluded.corpus_recordings,
                                 ai_labelled_recordings = excluded.ai_labelled_recordings,
                                 human_labels = excluded.human_labels,
                                 computed_at = excluded.computed_at;
$$;

-- ... and, once 'hide_weak_from_board' is 1, not on the leaderboard
create or replace function public.leaderboard(p_days integer default null, p_limit integer default 50)
returns table (rank bigint, username text, display_name text, avatar_url text, labelled bigint, last_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select row_number() over (order by count(*) desc, max(c.created_at), p.username), p.username, p.display_name, p.avatar_url,
         count(*), max(c.created_at)
  from private.contributions c join public.profiles p on p.id = c.user_id
  where p.deleted_at is null and not p.guest and (p_days is null or c.created_at >= now() - make_interval(days => p_days))
    and not (private.setting('hide_weak_from_board', 0) = 1 and private.is_weak(p.id))
  group by p.id
  order by count(*) desc, max(c.created_at), p.username
  limit least(greatest(coalesce(p_limit, 50), 1), 200);
$$;

-- exports carry the quality status; pull_labels.py leaves weak labellers out unless asked to keep them
drop function public.admin_export_labels(timestamptz);
create function public.admin_export_labels(p_since timestamptz default '-infinity')
returns table (label_id bigint, source_uid text, kind text, labeller text, username text, labels text[], other_text text,
               unsure boolean, listen_ms integer, attempt smallint, trust real, quality text, created_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select l.id, s.source_uid, s.kind, coalesce(p.fusion_name, 'crowd:' || left(p.id::text, 8)), p.username, l.labels,
         l.other_text, l.unsure, l.listen_ms, l.attempt, p.trust, coalesce(q.status, 'too early'), l.created_at
  from public.labels l
  join private.clip_sources s on s.clip_id = l.clip_id
  join public.profiles p on p.id = l.user_id
  left join private.labeller_quality q on q.user_id = l.user_id
  where l.created_at >= p_since
  order by l.id;
$$;
revoke all on function public.admin_export_labels(timestamptz) from public, anon, authenticated;
grant execute on function public.admin_export_labels(timestamptz) to service_role;

-- serving gathers the evidence faster: hidden repeats from a person's 30th answer (not their 300th), and someone on
-- watch gets more control clips and repeats until the question is settled
create or replace function public.next_clip()
returns table (clip_id uuid, audio_path text, duration_s real, expires_at timestamptz)
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
  pick public.clips;
  got boolean := false;
  att smallint := 1;
  n integer;
  r double precision := random();
  rep numeric := private.setting('repeat_rate', 0.04);
  gap integer;
  breadth integer := private.setting('breadth_votes', 3)::integer;
  ctrl_rate numeric;
  thin_only boolean;
  audited boolean;
begin
  if uid is null then raise exception 'sign in first' using errcode = '28000'; end if;
  if not exists (select 1 from public.profiles p where p.id = uid and p.deleted_at is null) then
    raise exception 'finish your profile first' using errcode = 'P0001';
  end if;
  return query
    select a.clip_id, c.audio_path, c.duration_s, a.expires_at
    from public.assignments a join public.clips c on c.id = a.clip_id
    where a.user_id = uid and a.state = 'open' and a.expires_at > now() and c.active
    order by a.assigned_at
    limit 1;
  if found then return; end if;

  select p.label_seq into n from public.profiles p where p.id = uid;
  audited := exists (select 1 from private.labeller_quality q where q.user_id = uid and q.status in ('watch', 'weak'));
  gap := least(private.setting('repeat_gap', 300)::integer,
               greatest(private.setting('repeat_gap_min', 30)::integer, n / 2));
  if audited then
    rep := private.setting('audit_repeat_rate', 0.12);
    gap := private.setting('audit_repeat_gap', 15)::integer;
  end if;

  -- a hidden retest: something they answered at least `gap` labels ago, not yet retested
  if n >= gap and r < rep then
    select c.* into pick
    from public.labels l join public.clips c on c.id = l.clip_id
    where l.user_id = uid and l.attempt = 1 and not l.unsure and l.seq <= n - gap and c.active
      and not exists (select 1 from public.assignments m where m.user_id = uid and m.clip_id = c.id and m.attempt = 2)
    order by random()
    limit 1
    for update of c skip locked;
    if found then got := true; att := 2; end if;
  end if;

  -- a control: often while we know little about this person (or they are on watch), rarely once they have answered many
  if not got then
    select greatest(private.setting('control_rate_min', 0.05),
                    private.setting('control_rate_max', 0.25) - 0.01 * count(*)) into ctrl_rate
    from public.labels l join private.clip_sources s on s.clip_id = l.clip_id
    where l.user_id = uid and s.kind = 'control';
    if audited then ctrl_rate := greatest(ctrl_rate, private.setting('audit_control_rate', 0.3)); end if;
    if r >= rep and r < rep + ctrl_rate then
      select c.* into pick
      from public.clips c join private.clip_sources s on s.clip_id = c.id
      where c.active and s.kind = 'control'
        and not exists (select 1 from public.assignments m where m.user_id = uid and m.clip_id = c.id)
      order by c.need desc, random()
      limit 1
      for update of c skip locked;
      if found then got := true; end if;
    end if;
  end if;

  foreach thin_only in array array[true, false] loop
    exit when got;

    -- started clips up to `breadth` judges, fullest first
    select c.* into pick from public.clips c
    where c.active and (c.boost > 0 or not thin_only) and c.votes > 0 and c.votes < least(c.target_votes, breadth)
      and not exists (select 1 from public.assignments m where m.user_id = uid and m.clip_id = c.id)
    order by c.votes desc, c.need + c.boost desc, random()
    limit 1
    for update of c skip locked;
    if found then got := true; exit; end if;

    -- the neediest clip they have never been given (fresh clips rank first; started ones keep filling to target)
    with candidates as (
      select c.id, c.need + c.boost as score from public.clips c
      where c.active and (c.boost > 0 or not thin_only)
        and not exists (select 1 from public.assignments m where m.user_id = uid and m.clip_id = c.id)
      order by c.need + c.boost desc
      limit 300
    )
    select c.* into pick
    from candidates k join public.clips c on c.id = k.id
    order by k.score
             - 0.15 * (select count(*) from public.assignments o
                       where o.clip_id = k.id and o.state = 'open' and o.expires_at > now())
             + 0.05 * random() desc
    limit 1
    for update of c skip locked;
    if found then got := true; end if;
  end loop;
  if not got then return; end if;

  insert into public.assignments (user_id, clip_id, attempt) values (uid, pick.id, att);
  return query select pick.id, pick.audio_path, pick.duration_s, now() + interval '20 minutes';
end $$;

select cron.schedule('refresh-quality', '7,22,37,52 * * * *', 'select private.refresh_quality()');
select private.refresh_quality();
select private.refresh_progress();
