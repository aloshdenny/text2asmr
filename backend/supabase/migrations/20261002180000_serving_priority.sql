-- Which clip to serve next, and never running dry.
--
-- Every clip carries a serving score, `need`, kept up to date by a trigger:
--   0.5 x judges still missing   (1 - votes / target_votes, floored at 0)  -- fewest judges first
-- + 0.3 x uncertainty            (machines and earlier listeners unsure: 1 = some class at P 0.5)
-- + 0.2 x disagreement           (1 - mean pairwise Jaccard of the crowd's answers; the uncertainty until 2 answers)
-- divided by (1 + 0.25 x judges beyond the target) and by (1 + skips).
-- An unjudged, unsure clip scores 1; a clip with its quorum and agreement keeps a little score, so once everything
-- has its quorum, listeners still get the most disputed clips. A listener only runs out after hearing every clip.

alter table public.clips add column uncertainty real not null default 0.5 check (uncertainty between 0 and 1);
alter table public.clips add column agreement real check (agreement between 0 and 1);
alter table public.clips add column skips integer not null default 0;
alter table public.clips add column need real not null default 1;
create index clips_by_need on public.clips (need desc) where active;

alter table private.clip_sources drop constraint clip_sources_kind_check,
  add constraint clip_sources_kind_check check (kind in ('unsure', 'confident', 'unlabelled', 'control'));

create function private.set_need()
returns trigger
language plpgsql as $$
begin
  new.need := (0.5 * greatest(0, 1 - new.votes::real / new.target_votes)
             + 0.3 * new.uncertainty
             + 0.2 * coalesce(1 - new.agreement, new.uncertainty))
            / (1 + 0.25 * greatest(0, new.votes - new.target_votes))
            / (1 + new.skips);
  return new;
end $$;
create trigger clips_need before insert or update of votes, target_votes, uncertainty, agreement, skips on public.clips
  for each row execute function private.set_need();
update public.clips set uncertainty = least(1, greatest(0, priority)) where priority between 0 and 1;

-- Mean pairwise Jaccard of the answers a clip has (answers with "can't tell" abstain).
create function private.clip_agreement(p_clip uuid)
returns real
language sql stable security definer set search_path = '' as $$
  select avg(case when cardinality(a.labels) + cardinality(b.labels) = 0 then 1
                  else (select count(*) from unnest(a.labels) x where x = any (b.labels))::real
                       / (select count(distinct y) from unnest(a.labels || b.labels) y) end)::real
  from public.labels a join public.labels b on a.clip_id = b.clip_id and a.id < b.id
  where a.clip_id = p_clip and not a.unsure and not b.unsure;
$$;

-- The best clip for this listener: highest need among clips they have never been given, a little less for each
-- listener currently holding it (it will have their answer soon), a little jitter so people arriving together get
-- different clips. Ranked within the 300 neediest, which the index serves directly.
create or replace function public.next_clip()
returns table (clip_id uuid, audio_path text, duration_s real, expires_at timestamptz)
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
  pick public.clips;
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
  if not found then return; end if;

  insert into public.assignments (user_id, clip_id) values (uid, pick.id);
  return query select pick.id, pick.audio_path, pick.duration_s, now() + interval '20 minutes';
end $$;

-- submit_label also refreshes the clip's agreement (and so its need)
create or replace function public.submit_label(p_clip uuid, p_labels text[], p_other text default null,
                                               p_unsure boolean default false, p_listen_ms integer default 0)
returns table (total bigint, today bigint)
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
  asg public.assignments;
  clean text[];
begin
  if uid is null then raise exception 'sign in first' using errcode = '28000'; end if;
  select * into asg from public.assignments a where a.user_id = uid and a.clip_id = p_clip for update;
  if not found or asg.state <> 'open' then
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
  insert into public.labels (user_id, clip_id, labels, other_text, unsure, listen_ms)
  values (uid, p_clip, clean,
          case when 'something else' = any (clean) then nullif(btrim(left(p_other, 200)), '') end,
          coalesce(p_unsure, false), least(greatest(p_listen_ms, 0), 600000));
  update public.assignments a set state = 'done' where a.user_id = uid and a.clip_id = p_clip;
  update public.clips c set votes = c.votes + 1, agreement = private.clip_agreement(p_clip) where c.id = p_clip;
  return query
    select count(*), count(*) filter (where c.created_at >= date_trunc('day', now()))
    from private.contributions c where c.user_id = uid;
end $$;

-- a skipped clip (broken, uncomfortable, unclear) sinks for everyone, not just the person who passed on it
create or replace function public.skip_clip(p_clip uuid)
returns void
language plpgsql security definer set search_path = '' as $$
begin
  if auth.uid() is null then raise exception 'sign in first' using errcode = '28000'; end if;
  update public.assignments a set state = 'skipped'
  where a.user_id = auth.uid() and a.clip_id = p_clip and a.state = 'open';
  if found then update public.clips c set skips = c.skips + 1 where c.id = p_clip; end if;
end $$;

-- admin_add_clips takes the clip's uncertainty (older callers sent it as "priority")
create or replace function public.admin_add_clips(p_clips jsonb)
returns integer
language plpgsql security definer set search_path = '' as $$
declare
  item jsonb;
  cid uuid;
  added integer := 0;
  u real;
begin
  perform private.require_service_role();
  for item in select * from jsonb_array_elements(p_clips) loop
    u := least(1, greatest(0, coalesce((item ->> 'uncertainty')::real, (item ->> 'priority')::real, 0.5)));
    select s.clip_id into cid from private.clip_sources s where s.source_uid = item ->> 'source_uid';
    if found then
      update public.clips c set uncertainty = u, priority = u,
                                target_votes = coalesce((item ->> 'target_votes')::smallint, c.target_votes), active = true
      where c.id = cid;
      update private.clip_sources s set kind = coalesce(item ->> 'kind', s.kind), info = coalesce(item -> 'info', s.info)
      where s.clip_id = cid;
      continue;
    end if;
    insert into public.clips (audio_path, duration_s, target_votes, priority, uncertainty, batch)
    values (item ->> 'audio_path', coalesce((item ->> 'duration_s')::real, 6),
            coalesce((item ->> 'target_votes')::smallint, 3), u, u, coalesce(item ->> 'batch', 'default'))
    returning id into cid;
    insert into private.clip_sources (clip_id, source_uid, kind, info)
    values (cid, item ->> 'source_uid', item ->> 'kind', coalesce(item -> 'info', '{}'));
    added := added + 1;
  end loop;
  return added;
end $$;

drop function public.admin_queue_stats();
create function public.admin_queue_stats()
returns table (batch text, kind text, clips bigint, complete bigint, votes bigint, mean_need real)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select c.batch, s.kind, count(*), count(*) filter (where c.votes >= c.target_votes), sum(c.votes), avg(c.need)::real
  from public.clips c join private.clip_sources s on s.clip_id = c.id
  where c.active group by 1, 2 order by 1, 2;
$$;

revoke all on function private.set_need(), private.clip_agreement(uuid) from public, anon, authenticated;
