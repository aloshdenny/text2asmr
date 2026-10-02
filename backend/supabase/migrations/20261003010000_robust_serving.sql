-- Robust serving: many judges per clip, hidden retests, trust-weighted consensus.
--
-- 1. Quorum. Every clip wants at least `min_votes` (12) first answers and takes at most `max_votes` (20); between the
--    two, disputed clips keep drawing listeners while settled ones drop back. One person, or a handful, cannot decide
--    a clip.
-- 2. Retests. Now and then (`repeat_rate`) a clip a person labelled at least `repeat_gap` (300) of their own labels
--    ago comes back to them as a second attempt. Nothing marks it; the two answers measure how consistent they are.
--    Second attempts never count toward a clip's quorum.
-- 3. Controls. Clips people already labelled (kind 'control') are served more often to newcomers (`control_rate_max`)
--    and rarely to people who have answered many (`control_rate_min`), so trust is learned fast and then cheaply.
-- 4. Trust. private.refresh_trust() (every 15 minutes) scores each labeller from agreement with the majority of the
--    other answers on the same clip, with control answers and with their own retests, plus a penalty for answering
--    faster than a clip can be heard. Consensus on a clip is trust-weighted, so a poisoner's votes count for little.
-- Labellers see none of this: no quorum, no retest flag, no trust.

insert into private.settings (key, value) values
  ('min_votes', 12), ('max_votes', 20), ('repeat_gap', 300), ('repeat_rate', 0.04),
  ('control_rate_max', 0.25), ('control_rate_min', 0.05), ('trust_prior', 5), ('trust_floor', 0.15)
on conflict (key) do nothing;

create function private.setting(p_key text, p_default numeric)
returns numeric
language sql stable security definer set search_path = '' as $$
  select coalesce((select s.value from private.settings s where s.key = p_key), p_default);
$$;

create function private.jaccard(a text[], b text[])
returns real
language sql immutable as $$
  select case when coalesce(cardinality(a), 0) + coalesce(cardinality(b), 0) = 0 then 1
              else (select count(*) from unnest(a) x where x = any (b))::real
                   / (select count(distinct y) from unnest(a || b) y) end;
$$;

-- people: a running count of their labels (the retest gap is measured in it) and a trust score, neither readable by the API
alter table public.profiles add column label_seq integer not null default 0,
  add column trust real not null default 0.5, add column trust_evidence integer not null default 0;

-- a person can answer a clip twice: the first attempt, and one hidden retest
alter table public.labels add column attempt smallint not null default 1, add column seq integer;
alter table public.labels drop constraint labels_user_id_clip_id_key;
alter table public.labels add constraint labels_user_clip_attempt unique (user_id, clip_id, attempt);
update public.labels l set seq = s.n
from (select id, row_number() over (partition by user_id order by created_at, id) as n from public.labels) s where s.id = l.id;
update public.profiles p set label_seq = coalesce((select max(l.seq) from public.labels l where l.user_id = p.id), 0);

alter table public.assignments add column attempt smallint not null default 1;
alter table public.assignments drop constraint assignments_pkey;
alter table public.assignments add primary key (user_id, clip_id, attempt);

alter table public.clips add column max_votes smallint not null default 20, add column crowd_conf real check (crowd_conf between 0 and 1);
alter table public.clips alter column target_votes set default 12;

-- need: missing judges up to the minimum, how unsettled the crowd is, how unsure the machines are; past the maximum
-- a clip is served only when nothing else is left
create or replace function private.set_need()
returns trigger
language plpgsql as $$
begin
  new.need := (0.5 * greatest(0, 1 - new.votes::real / new.target_votes)
             + 0.3 * coalesce(1 - new.crowd_conf, new.uncertainty)
             + 0.2 * new.uncertainty)
            * case when new.votes >= new.max_votes then 0.05 else 1 end
            / (1 + new.skips);
  return new;
end $$;
drop trigger clips_need on public.clips;
create trigger clips_need before insert or update of votes, target_votes, max_votes, uncertainty, agreement, crowd_conf, skips
  on public.clips for each row execute function private.set_need();
update public.clips set target_votes = 12, max_votes = 20 where target_votes < 12;

create or replace function private.clip_agreement(p_clip uuid)
returns real
language sql stable security definer set search_path = '' as $$
  select avg(private.jaccard(a.labels, b.labels))::real
  from public.labels a join public.labels b on a.clip_id = b.clip_id and a.id < b.id
  where a.clip_id = p_clip and a.attempt = 1 and b.attempt = 1 and not a.unsure and not b.unsure;
$$;

-- trust-weighted consensus: for each label anyone chose, how far its weighted share is from a coin flip, averaged
-- (1 = everyone agrees on every label; 0 = every label split down the middle). Labellers under the trust floor
-- (once there is evidence) do not count.
create function private.clip_consensus(p_clip uuid)
returns real
language sql stable security definer set search_path = '' as $$
  with v as (
    select l.labels, p.trust as w
    from public.labels l join public.profiles p on p.id = l.user_id
    where l.clip_id = p_clip and l.attempt = 1 and not l.unsure
      and (p.trust_evidence < 20 or p.trust >= private.setting('trust_floor', 0.15))
  ), tot as (select sum(w) as w, count(*) as n from v),
  per as (select x as label, sum(v.w) as w from v, unnest(v.labels) x group by x)
  select case when (select n from tot) < 2 or (select w from tot) <= 0 then null
              else coalesce((select avg(abs(2 * per.w / tot.w - 1)) from per, tot), 1)::real end;
$$;

-- serve: an open assignment first; else, by a draw, a hidden retest or a control; else the neediest clip this person
-- has never had
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
  gap integer := private.setting('repeat_gap', 300)::integer;
  ctrl_rate numeric;
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

  -- a control: often while we know little about this person, rarely once they have answered many
  if not got then
    select greatest(private.setting('control_rate_min', 0.05),
                    private.setting('control_rate_max', 0.25) - 0.01 * count(*)) into ctrl_rate
    from public.labels l join private.clip_sources s on s.clip_id = l.clip_id
    where l.user_id = uid and s.kind = 'control';
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

  -- otherwise the neediest clip they have never been given
  if not got then
    with candidates as (
      select c.id, c.need from public.clips c
      where c.active and not exists (select 1 from public.assignments m where m.user_id = uid and m.clip_id = c.id)
      order by c.need desc
      limit 300
    )
    select c.* into pick
    from candidates k join public.clips c on c.id = k.id
    order by k.need
             - 0.15 * (select count(*) from public.assignments o
                       where o.clip_id = k.id and o.state = 'open' and o.expires_at > now())
             + 0.05 * random() desc
    limit 1
    for update of c skip locked;
    if found then got := true; end if;
  end if;
  if not got then return; end if;

  insert into public.assignments (user_id, clip_id, attempt) values (uid, pick.id, att);
  return query select pick.id, pick.audio_path, pick.duration_s, now() + interval '20 minutes';
end $$;

create or replace function public.submit_label(p_clip uuid, p_labels text[], p_other text default null,
                                               p_unsure boolean default false, p_listen_ms integer default 0)
returns table (total bigint, today bigint)
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
  asg public.assignments;
  clean text[];
  n integer;
begin
  if uid is null then raise exception 'sign in first' using errcode = '28000'; end if;
  select * into asg from public.assignments a
  where a.user_id = uid and a.clip_id = p_clip and a.state = 'open'
  order by a.attempt desc
  limit 1
  for update;
  if not found then
    raise exception 'this clip is not assigned to you' using errcode = 'P0001';
  end if;
  select coalesce(array_agg(distinct l order by l), '{}') into clean from unnest(coalesce(p_labels, '{}')) l;
  if exists (select 1 from unnest(clean) l where l not in (select o.key from public.label_options o)) then
    raise exception 'unknown label' using errcode = '22023';
  end if;
  if cardinality(clean) = 0 and not coalesce(p_unsure, false) then
    raise exception 'pick at least one label, or say you can''t tell' using errcode = '22023';
  end if;
  if now() - asg.assigned_at < interval '2 seconds' or coalesce(p_listen_ms, 0) < 1500 then
    raise exception 'listen to the clip first' using errcode = '22023';
  end if;
  update public.profiles p set label_seq = p.label_seq + 1 where p.id = uid returning p.label_seq into n;
  insert into public.labels (user_id, clip_id, labels, other_text, unsure, listen_ms, attempt, seq)
  values (uid, p_clip, clean,
          case when 'something else' = any (clean) then nullif(btrim(left(p_other, 200)), '') end,
          coalesce(p_unsure, false), least(greatest(p_listen_ms, 0), 600000), asg.attempt, n);
  update public.assignments a set state = 'done' where a.user_id = uid and a.clip_id = p_clip and a.attempt = asg.attempt;
  if asg.attempt = 1 then
    update public.clips c set votes = c.votes + 1, agreement = private.clip_agreement(p_clip),
                              crowd_conf = private.clip_consensus(p_clip)
    where c.id = p_clip;
  end if;
  return query
    select count(*), count(*) filter (where c.created_at >= date_trunc('day', now()))
    from private.contributions c where c.user_id = uid;
end $$;

create or replace function public.skip_clip(p_clip uuid)
returns void
language plpgsql security definer set search_path = '' as $$
begin
  if auth.uid() is null then raise exception 'sign in first' using errcode = '28000'; end if;
  update public.assignments a set state = 'skipped'
  where a.user_id = auth.uid() and a.clip_id = p_clip and a.state = 'open';
  if found then update public.clips c set skips = c.skips + 1 where c.id = p_clip; end if;
end $$;

-- trust, from three kinds of evidence, smoothed toward 0.5 by `trust_prior` pseudo-answers:
--   majority  Jaccard of each first answer with the labels most other people chose on that clip (>= 4 others)
--   controls  Jaccard with the labels people gave the control clip in the Ear Check kits (counts double)
--   retests   Jaccard of a person's first answer with their own hidden retest (counts double)
-- and x0.7 for someone whose typical answer comes faster than the clip can be heard (median listen < 2.5 s).
create function private.refresh_trust()
returns void
language plpgsql security definer set search_path = '' as $$
declare
  prior numeric := private.setting('trust_prior', 5);
begin
  with first as (
    select l.id, l.user_id, l.clip_id, l.labels from public.labels l where l.attempt = 1 and not l.unsure
  ), maj as (
    select f.user_id, private.jaccard(f.labels, coalesce((
             select array_agg(x) from (
               select x from first o, unnest(o.labels) x where o.clip_id = f.clip_id and o.user_id <> f.user_id
               group by x having 2 * count(*) >= (select count(*) from first o2 where o2.clip_id = f.clip_id and o2.user_id <> f.user_id)
             ) m), '{}')) as j
    from first f
    where (select count(*) from first o2 where o2.clip_id = f.clip_id and o2.user_id <> f.user_id) >= 4
  ), ctrl as (
    select f.user_id, private.jaccard(f.labels, array(
             select distinct x from jsonb_each(s.info -> 'people') e(who, v), jsonb_array_elements_text(e.v) x)) as j
    from first f join private.clip_sources s on s.clip_id = f.clip_id
    where s.kind = 'control' and jsonb_typeof(s.info -> 'people') = 'object'
  ), self as (
    select a.user_id, private.jaccard(a.labels, b.labels) as j
    from public.labels a join public.labels b on b.user_id = a.user_id and b.clip_id = a.clip_id and b.attempt = 2
    where a.attempt = 1
  ), ev as (
    select user_id, sum(j) as s, count(*) as n from maj group by user_id
    union all select user_id, 2 * sum(j), 2 * count(*) from ctrl group by user_id
    union all select user_id, 2 * sum(j), 2 * count(*) from self group by user_id
  ), agg as (
    select user_id, sum(s) as s, sum(n) as n from ev group by user_id
  ), speed as (
    select user_id, percentile_cont(0.5) within group (order by listen_ms) as med from public.labels group by user_id
  )
  update public.profiles p
  set trust = least(1, greatest(0, ((coalesce(a.s, 0) + prior * 0.5) / (coalesce(a.n, 0) + prior))
                                   * case when sp.med < 2500 then 0.7 else 1 end))::real,
      trust_evidence = coalesce(a.n, 0)::integer
  from public.profiles q left join agg a on a.user_id = q.id left join speed sp on sp.user_id = q.id
  where p.id = q.id and (a.user_id is not null or sp.user_id is not null);

  update public.clips c set crowd_conf = private.clip_consensus(c.id) where c.votes >= 2;
end $$;

revoke all on function private.setting(text, numeric), private.clip_consensus(uuid), private.refresh_trust()
  from public, anon, authenticated;

-- exports carry the attempt and the labeller's trust (the fusion learns its own reliabilities from first attempts)
drop function public.admin_export_labels(timestamptz);
create function public.admin_export_labels(p_since timestamptz default '-infinity')
returns table (label_id bigint, source_uid text, kind text, labeller text, username text, labels text[], other_text text,
               unsure boolean, listen_ms integer, attempt smallint, trust real, created_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select l.id, s.source_uid, s.kind, coalesce(p.fusion_name, 'crowd:' || left(p.id::text, 8)), p.username, l.labels,
         l.other_text, l.unsure, l.listen_ms, l.attempt, p.trust, l.created_at
  from public.labels l
  join private.clip_sources s on s.clip_id = l.clip_id
  join public.profiles p on p.id = l.user_id
  where l.created_at >= p_since
  order by l.id;
$$;

create extension if not exists pg_cron;
select cron.schedule('refresh-trust', '*/15 * * * *', 'select private.refresh_trust()');

-- new clips take the quorum from settings unless a caller asks for another
create or replace function public.admin_add_clips(p_clips jsonb)
returns integer
language plpgsql security definer set search_path = '' as $$
declare
  item jsonb;
  cid uuid;
  added integer := 0;
  u real;
  lo smallint := private.setting('min_votes', 12)::smallint;
  hi smallint := private.setting('max_votes', 20)::smallint;
begin
  perform private.require_service_role();
  for item in select * from jsonb_array_elements(p_clips) loop
    u := least(1, greatest(0, coalesce((item ->> 'uncertainty')::real, (item ->> 'priority')::real, 0.5)));
    select s.clip_id into cid from private.clip_sources s where s.source_uid = item ->> 'source_uid';
    if found then
      update public.clips c set uncertainty = u, priority = u,
                                target_votes = coalesce((item ->> 'target_votes')::smallint, c.target_votes),
                                max_votes = greatest(coalesce((item ->> 'target_votes')::smallint, c.target_votes), c.max_votes),
                                active = true
      where c.id = cid;
      update private.clip_sources s set kind = coalesce(item ->> 'kind', s.kind), info = coalesce(item -> 'info', s.info)
      where s.clip_id = cid;
      continue;
    end if;
    insert into public.clips (audio_path, duration_s, target_votes, max_votes, priority, uncertainty, batch)
    values (item ->> 'audio_path', coalesce((item ->> 'duration_s')::real, 6),
            coalesce((item ->> 'target_votes')::smallint, lo),
            greatest(coalesce((item ->> 'target_votes')::smallint, lo), hi), u, u, coalesce(item ->> 'batch', 'default'))
    returning id into cid;
    insert into private.clip_sources (clip_id, source_uid, kind, info)
    values (cid, item ->> 'source_uid', item ->> 'kind', coalesce(item -> 'info', '{}'));
    added := added + 1;
  end loop;
  return added;
end $$;
