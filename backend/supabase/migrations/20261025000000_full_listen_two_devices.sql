-- Two rules for everyone who labels:
--
-- 1. Hear the whole clip first. An answer is accepted only once the clip has played through (the browser's played
--    ranges add up to its length, give or take 0.6 s of encoder padding) and at least that long has passed since it
--    was served, so a faked play time or a sped-up player is refused too. Was: 1.5 s and 2 s.
-- 2. One account, at most two devices. A device is the browser's random id as the /sb proxy scrambles it (x-ab-device,
--    see 20261020000000_device_marks.sql). The first two devices an account answers from hold its slots; a third is
--    refused until a slot frees up: after 30 days unseen, or when an admin resets the account. The site calls
--    claim_device right after sign-in and signs a refused device straight out. A request with no device id (the
--    header stripped, or a browser that blocks storage) cannot answer. Requests that do not come through the API at
--    all (no request headers: SQL run by the project itself) are not checked.

create table private.account_devices (
  user_id uuid not null references auth.users (id) on delete cascade,
  device text not null,
  first_seen timestamptz not null default now(),
  last_seen timestamptz not null default now(),
  primary key (user_id, device)
);

-- people already labelling keep the two devices they used most recently
insert into private.account_devices (user_id, device, first_seen, last_seen)
select user_id, value, first_seen, last_seen
from (select m.*, row_number() over (partition by m.user_id order by m.last_seen desc) r
      from private.user_marks m where m.kind = 'device' and m.last_seen > now() - interval '30 days') x
where r <= 2
on conflict do nothing;

create function private.check_device(p_user uuid)
returns void
language plpgsql security definer set search_path = '' as $$
declare
  h text := nullif(current_setting('request.headers', true), '');
  d text;
begin
  if p_user is null or h is null then return; end if;
  d := h::json ->> 'x-ab-device';
  if d is null or d !~ '^[A-Za-z0-9_-]{16,64}$' then
    raise exception 'this browser could not be recognised: allow the site to store data, then reload' using errcode = 'P0003';
  end if;
  perform pg_advisory_xact_lock(hashtextextended(p_user::text, 7));
  update private.account_devices a set last_seen = now() where a.user_id = p_user and a.device = d;
  if found then return; end if;
  delete from private.account_devices a where a.user_id = p_user and a.last_seen < now() - interval '30 days';
  if (select count(*) from private.account_devices a where a.user_id = p_user) >= 2 then
    raise exception 'this account is already in use on two other devices' using errcode = 'P0004';
  end if;
  insert into private.account_devices (user_id, device) values (p_user, d);
end $$;
revoke all on function private.check_device(uuid) from public, anon, authenticated;

-- the site asks right after sign-in, so a third device is turned away at the door rather than at its first answer
create function public.claim_device()
returns void
language plpgsql security definer set search_path = '' as $$
begin
  if (select auth.uid()) is null then raise exception 'sign in first' using errcode = '28000'; end if;
  perform private.check_device((select auth.uid()));
end $$;
revoke all on function public.claim_device() from public, anon;
grant execute on function public.claim_device() to authenticated;

-- frees an account's device slots, e.g. after someone changes phone (service role)
create function public.admin_reset_devices(p_username text)
returns integer
language plpgsql security definer set search_path = '' as $$
declare
  n integer;
begin
  perform private.require_service_role();
  delete from private.account_devices a using public.profiles p
  where p.id = a.user_id and p.username = lower(btrim(p_username)) and p.deleted_at is null;
  get diagnostics n = row_count;
  return n;
end $$;
revoke all on function public.admin_reset_devices(text) from public, anon, authenticated;
grant execute on function public.admin_reset_devices(text) to service_role;

-- answering: the device rule, then the whole clip
create or replace function public.submit_label(p_clip uuid, p_labels text[], p_other text default null,
                                               p_unsure boolean default false, p_listen_ms integer default 0)
returns table (total bigint, today bigint)
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
  asg public.assignments;
  clean text[];
  n integer;
  dur double precision;
begin
  if uid is null then raise exception 'sign in first' using errcode = '28000'; end if;
  perform private.check_device(uid);
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
  select coalesce(c.duration_s, 6) into dur from public.clips c where c.id = p_clip;
  if now() - asg.assigned_at < make_interval(secs => greatest(dur - 0.6, 1.5))
     or coalesce(p_listen_ms, 0) < greatest(dur * 1000 - 600, 1500) then
    raise exception 'listen to the whole clip first' using errcode = '22023';
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

create or replace function public.edit_answer(p_clip uuid, p_spans jsonb, p_other text default null)
returns void
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := (select auth.uid());
  clean jsonb := private.clean_spans(p_clip, coalesce(p_spans, '[]'));
  labs text[] := private.span_labels(clean, p_other);
begin
  if uid is null then raise exception 'sign in first' using errcode = '28000'; end if;
  perform private.check_device(uid);
  if cardinality(labs) = 0 then
    raise exception 'mark at least one sound, or say what you hear' using errcode = '22023';
  end if;
  update public.labels l
  set labels = labs, spans = clean,
      other_text = case when 'something else' = any (labs) then nullif(btrim(left(p_other, 200)), '') end
  where l.user_id = uid and l.clip_id = p_clip and l.attempt = 1 and l.created_at > now() - interval '24 hours';
  if not found then
    raise exception 'only answers from the last 24 hours can be changed' using errcode = 'P0001';
  end if;
  update public.clips c set agreement = private.clip_agreement(p_clip), crowd_conf = private.clip_consensus(p_clip)
  where c.id = p_clip;
end $$;
