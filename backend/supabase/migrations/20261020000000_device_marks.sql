-- Tell guests apart without knowing who they are.
--
-- Guest labelling (anonymous sign-ins) is on, so one person can make any number of accounts. The /sb proxy
-- (frontend/api/_lib/marks.ts) now sends scrambled marks with every request: a keyed hash of a random id the browser
-- keeps (device), of the client IP (ip) and of its /24 or /48 network (net), plus a coarse "Chrome Android" (ua). The
-- hash key lives only in Vercel, the IP itself never reaches Supabase, and nothing here can be turned back into an IP.
--
-- Marks are recorded when someone saves an answer or gets a profile, as themselves (imports by the service role are
-- not marked), and forgotten 180 days after they were last seen. Accounts sharing a device are a warning sign in the
-- weak-labeller rating (refresh_quality); a shared IP or network is only reported, since households, offices, campuses
-- and mobile networks share those. admin_shared_marks() lists the groups.

create table private.user_marks (
  user_id uuid not null references auth.users (id) on delete cascade,
  kind text not null check (kind in ('device', 'ip', 'net', 'ua')),
  value text not null,
  first_seen timestamptz not null default now(),
  last_seen timestamptz not null default now(),
  hits integer not null default 1,
  primary key (user_id, kind, value)
);
create index user_marks_by_value on private.user_marks (kind, value);
revoke all on private.user_marks from public, anon, authenticated;

-- the marks of the current request (headers the proxy set), recorded for p_user
create function private.note_marks(p_user uuid)
returns void
language plpgsql security definer set search_path = '' as $$
declare
  h json := nullif(current_setting('request.headers', true), '')::json;
  k text;
  v text;
begin
  if h is null or p_user is null then return; end if;
  foreach k in array array['device', 'ip', 'net', 'ua'] loop
    v := h ->> ('x-ab-' || k);
    continue when v is null
      or (k = 'ua' and v !~ '^[A-Za-z ]{1,40}$')
      or (k <> 'ua' and v !~ '^[A-Za-z0-9_-]{16,64}$');
    insert into private.user_marks (user_id, kind, value) values (p_user, k, v)
    on conflict (user_id, kind, value) do update set last_seen = now(), hits = private.user_marks.hits + 1;
  end loop;
end $$;

create function private.marks_on_label()
returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  if new.user_id = (select auth.uid()) then perform private.note_marks(new.user_id); end if;
  return null;
end $$;
create trigger labels_marks after insert on public.labels for each row execute function private.marks_on_label();

create function private.marks_on_profile()
returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  if new.id = (select auth.uid()) then perform private.note_marks(new.id); end if;
  return null;
end $$;
create trigger profiles_marks after insert on public.profiles for each row execute function private.marks_on_profile();

revoke all on function private.note_marks(uuid), private.marks_on_label(), private.marks_on_profile() from public, anon, authenticated;

select cron.schedule('forget-marks', '17 3 * * *', $$delete from private.user_marks where last_seen < now() - interval '180 days'$$);

-- groups of accounts that share a device, an IP or a network (service role); the mark is shown shortened
create function public.admin_shared_marks()
returns table (kind text, mark text, accounts integer, members text[], labels bigint, first_seen timestamptz, last_seen timestamptz)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  with g as (
    select m.kind, m.value, array_agg(distinct m.user_id) as users, min(m.first_seen) as f, max(m.last_seen) as l
    from private.user_marks m
    where m.kind in ('device', 'ip', 'net')
    group by m.kind, m.value
    having count(distinct m.user_id) >= 2
  )
  select g.kind, left(g.value, 8), cardinality(g.users),
         array(select coalesce(p.username, '(no profile)') || case when p.guest then ' (guest)' else '' end
                      || ' ' || (select count(*) from public.labels l where l.user_id = u)
               from unnest(g.users) u left join public.profiles p on p.id = u order by 1),
         (select count(*) from public.labels l where l.user_id = any (g.users)), g.f, g.l
  from g
  order by array_position(array['device', 'ip', 'net'], g.kind), cardinality(g.users) desc, g.l desc;
$$;
revoke all on function public.admin_shared_marks() from public, anon, authenticated;
grant execute on function public.admin_shared_marks() to service_role;

-- the weak-labeller rating learns about shared devices
alter table private.labeller_quality add column device_peers integer not null default 0;

create or replace function private.refresh_quality()
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
  ), dev as (
    -- other accounts (with answers) seen on the same device
    select m.user_id, count(distinct o.user_id)::integer as peers
    from private.user_marks m
    join private.user_marks o on o.kind = 'device' and o.value = m.value and o.user_id <> m.user_id
    where m.kind = 'device' and exists (select 1 from public.labels l where l.user_id = o.user_id)
    group by m.user_id
  ), joined as (
    select t.user_id, coalesce(t.answers, 0)::integer as answers, coalesce(f.refs, 0) as refs,
           f.matched, f.unmatched, f.ref_sim, f.follow, f.follow_se,
           coalesce(r.retests, 0) as retests, r.retest_j,
           t.med_answer_s, t.rush_rate, t.timing_cv, t.med_listen, tp.top_label, tp.top_share, t.unsure_rate, sk.skip_rate,
           t.all_first, coalesce(d.peers, 0) as device_peers,
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
             case when t.all_first >= 20 and t.unsure_rate > 0.5 then 'mostly can''t tell' end,
             case when coalesce(d.peers, 0) >= 1 then 'shares a device with other accounts' end
           ], null) as flags
    from timing t
    left join fol f on f.user_id = t.user_id
    left join tops tp on tp.user_id = t.user_id
    left join ret r on r.user_id = t.user_id
    left join skips sk on sk.user_id = t.user_id
    left join dev d on d.user_id = t.user_id
    where exists (select 1 from public.profiles p where p.id = t.user_id)
  )
  insert into private.labeller_quality as q (user_id, answers, refs, matched, unmatched, ref_similarity, follow, follow_se,
                                             retests, retest_agreement, median_answer_s, rush_rate, timing_cv,
                                             median_listen_ms, top_label, top_label_share, unsure_rate, skip_rate, flags,
                                             device_peers, status, computed_at)
  select j.user_id, j.answers, j.refs, j.matched, j.unmatched, j.ref_sim, j.follow, j.follow_se, j.retests, j.retest_j,
         j.med_answer_s, j.rush_rate, j.timing_cv, j.med_listen, j.top_label, j.top_share, j.unsure_rate, j.skip_rate,
         j.flags, j.device_peers,
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
    skip_rate = excluded.skip_rate, flags = excluded.flags, device_peers = excluded.device_peers, status = excluded.status,
    computed_at = excluded.computed_at;
end $$;

drop function public.admin_labeller_quality();
create function public.admin_labeller_quality()
returns table (username text, guest boolean, status text, flags text[], answers integer, refs integer, follow real,
               follow_se real, matched real, unmatched real, retests integer, retest_agreement real,
               median_answer_s real, rush_rate real, timing_cv real, median_listen_ms real, top_label text,
               top_label_share real, unsure_rate real, skip_rate real, device_peers integer, devices integer,
               networks integer, browsers text[], trust real, joined timestamptz, last_label timestamptz,
               computed_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select p.username, p.guest, q.status, q.flags, q.answers, q.refs, q.follow, q.follow_se, q.matched, q.unmatched,
         q.retests, q.retest_agreement, q.median_answer_s, q.rush_rate, q.timing_cv, q.median_listen_ms, q.top_label,
         q.top_label_share, q.unsure_rate, q.skip_rate, q.device_peers,
         (select count(*)::integer from private.user_marks m where m.user_id = q.user_id and m.kind = 'device'),
         (select count(*)::integer from private.user_marks m where m.user_id = q.user_id and m.kind = 'net'),
         array(select m.value from private.user_marks m where m.user_id = q.user_id and m.kind = 'ua' order by m.hits desc),
         p.trust, p.created_at, (select max(l.created_at) from public.labels l where l.user_id = q.user_id), q.computed_at
  from private.labeller_quality q join public.profiles p on p.id = q.user_id
  order by array_position(array['weak', 'watch', 'too early', 'ok'], q.status), q.follow nulls first, q.answers desc;
$$;
revoke all on function public.admin_labeller_quality() from public, anon, authenticated;
grant execute on function public.admin_labeller_quality() to service_role;

select private.refresh_quality();
