-- T2A crowd labelling: community listeners label short ASMR clips for the ASMR-CLAP ontology.
--
-- Every clip is heard by several people (target_votes), and nobody sees anyone else's answer, how many answers a
-- clip already has, what any model thinks, or where the clip came from: the browser only ever gets an opaque clip
-- id and a short-lived signed URL for the one clip assigned to it. A few clips people already labelled are mixed
-- in as controls, indistinguishable from the rest. Labels flow back into scripts/fuse_labels.py, where each
-- listener is one more labeller whose reliability is learned per class.
--
-- Labellers write only through the RPCs below (security definer); tables are read-only or invisible to them.

create schema if not exists private;
revoke all on schema private from public, anon, authenticated;

-- ---------------------------------------------------------------------------------------------------------------
-- label menu (same 20 labels and hints as the Ear Check kits)
create table public.label_options (
  key text primary key,
  grp text not null check (grp in ('Voice', 'Triggers', 'Other')),
  hint text not null,
  sort smallint not null unique
);
insert into public.label_options (key, grp, hint, sort) values
  ('breathing', 'Voice', 'audible breaths in or out, even quiet ones between whispered words', 1),
  ('kissing', 'Voice', 'kisses: short wet smacks of the lips', 2),
  ('oral sounds', 'Voice', 'licking, lip smacks, wet mouth sounds (not kisses)', 3),
  ('moaning', 'Voice', 'moans or sighs with voice in them', 4),
  ('whispering', 'Voice', 'whispered words', 5),
  ('normal speech', 'Voice', 'talking in a normal speaking voice', 6),
  ('tapping', 'Triggers', 'fingertips or nails tapping a surface: short, separate taps', 7),
  ('scratching', 'Triggers', 'nails dragged over a textured surface: continuous rough scraping', 8),
  ('crinkling', 'Triggers', 'plastic, foil or a wrapper squeezed: sharp crackly sounds', 9),
  ('brushing', 'Triggers', 'a brush stroked over the mic or a surface: soft swishing', 10),
  ('liquid', 'Triggers', 'water or gel: pouring, dripping, sloshing, bubbles', 11),
  ('spraying', 'Triggers', 'a spray bottle or mist: short hissy bursts of spray', 12),
  ('microphone touching', 'Triggers', 'fingers on the microphone itself: muffled rubs and thumps', 13),
  ('sticky', 'Triggers', 'tacky surfaces pulling apart: tape, slime, sticky fingers', 14),
  ('fabric rustling', 'Triggers', 'cloth moving: shirts, blankets, gloves', 15),
  ('paper rustling', 'Triggers', 'paper: pages turning, crumpled paper, cardboard (not cutting)', 16),
  ('cutting', 'Triggers', 'scissors or a knife: snips and slices (including cutting paper)', 17),
  ('background music', 'Other', 'music playing under everything else', 18),
  ('silence / room tone', 'Other', 'nothing but quiet room noise', 19),
  ('something else', 'Other', 'a sound that is not on this list: type what you hear', 20);

-- ---------------------------------------------------------------------------------------------------------------
-- tables
-- A profile outlives its sign-in: deleting an account removes the auth user and takes the profile off every public
-- list, but its labels stay in the dataset (so profiles.id deliberately has no foreign key to auth.users).
create table public.profiles (
  id uuid primary key,                                  -- the auth user it was made for
  username text not null unique check (username ~ '^[a-z0-9_]{3,20}$'),
  display_name text check (char_length(display_name) between 1 and 40),
  avatar_url text check (avatar_url ~ '^(https://|avatars/)'),  -- an https URL, or an object in the 'avatars' bucket
  fusion_name text unique,                              -- labeller name in fuse_labels.py for people who labelled Ear Check
                                                        -- kits before the site ('adi', ...); others export as crowd:<id>
  imported boolean not null default false,              -- made for a kit labeller: they claim it by setting a password
  adult_confirmed_at timestamptz not null default now(), -- set when the 18+ box is ticked (clips include intimate vocal sounds)
  created_at timestamptz not null default now(),
  deleted_at timestamptz
);

create table public.clips (
  id uuid primary key default gen_random_uuid(),
  audio_path text not null unique,                      -- object name in the private 'clips' bucket, opaque
  duration_s real not null default 6 check (duration_s > 0 and duration_s <= 30),
  target_votes smallint not null default 3 check (target_votes between 1 and 20),
  votes smallint not null default 0,
  priority real not null default 0,                     -- higher is served sooner (low fused confidence)
  active boolean not null default true,
  batch text not null default 'default',
  created_at timestamptz not null default now()
);
create index clips_queue on public.clips (priority desc) where active;

-- where a clip came from and what is already known about it: never visible to labellers
create table private.clip_sources (
  clip_id uuid primary key references public.clips (id) on delete cascade,
  source_uid text not null unique,                      -- uid in the CLAP pools (fuse_labels.py)
  kind text not null check (kind in ('unsure', 'unlabelled', 'control')),
  info jsonb not null default '{}'
);

create table public.assignments (
  user_id uuid not null references auth.users (id) on delete cascade,
  clip_id uuid not null references public.clips (id) on delete cascade,
  assigned_at timestamptz not null default now(),
  expires_at timestamptz not null default now() + interval '20 minutes',
  state text not null default 'open' check (state in ('open', 'done', 'skipped')),
  primary key (user_id, clip_id)
);
create index assignments_open_by_clip on public.assignments (clip_id) where state = 'open';
create index assignments_open_by_user on public.assignments (user_id) where state = 'open';

create table public.labels (
  id bigint generated always as identity primary key,
  user_id uuid not null references public.profiles (id),
  clip_id uuid not null references public.clips (id) on delete cascade,
  labels text[] not null,
  other_text text check (char_length(other_text) <= 200),
  unsure boolean not null default false,
  listen_ms integer not null default 0,
  created_at timestamptz not null default now(),
  unique (user_id, clip_id)
);
create index labels_by_user_time on public.labels (user_id, created_at);
create index labels_by_time on public.labels (created_at);

-- labels people gave in the Ear Check kits before the site existed: they count toward the leaderboard and the
-- contribution graph (the fusion already reads them from judges/human/*.jsonl, so they are not exported again)
create table public.imported_labels (
  id bigint generated always as identity primary key,
  user_id uuid not null references public.profiles (id),
  source_uid text not null,
  labels text[] not null,
  other_text text,
  kit text not null,
  created_at timestamptz not null,
  unique (user_id, kit, source_uid)
);
create index imported_by_user_time on public.imported_labels (user_id, created_at);

-- every contribution, for counting
create view private.contributions with (security_barrier) as
  select user_id, created_at from public.labels
  union all
  select user_id, created_at from public.imported_labels;

-- ---------------------------------------------------------------------------------------------------------------
-- access: the menu and public profile fields are readable by anyone; a labeller can read their own assignments
-- and labels; clips and sources are reachable only through the functions below
alter table public.label_options enable row level security;
alter table public.profiles enable row level security;
alter table public.clips enable row level security;
alter table public.assignments enable row level security;
alter table public.labels enable row level security;
alter table public.imported_labels enable row level security;
alter table private.clip_sources enable row level security;

revoke all on public.label_options, public.profiles, public.clips, public.assignments, public.labels, public.imported_labels
  from anon, authenticated;
grant select on public.label_options to anon, authenticated;
grant select (id, username, display_name, avatar_url, created_at) on public.profiles to anon, authenticated;
grant insert (id, username, display_name, avatar_url) on public.profiles to authenticated;
grant update (username, display_name, avatar_url) on public.profiles to authenticated;
grant select on public.assignments, public.labels, public.imported_labels to authenticated;

create policy "menu is public" on public.label_options for select using (true);
create policy "profiles are public" on public.profiles for select using (deleted_at is null);
create policy "create own profile" on public.profiles for insert to authenticated
  with check (id = (select auth.uid()));
create policy "edit own profile" on public.profiles for update to authenticated
  using (id = (select auth.uid()) and deleted_at is null) with check (id = (select auth.uid()));
create policy "own assignments" on public.assignments for select to authenticated using (user_id = (select auth.uid()));
create policy "own labels" on public.labels for select to authenticated using (user_id = (select auth.uid()));
create policy "own imported labels" on public.imported_labels for select to authenticated using (user_id = (select auth.uid()));

-- ---------------------------------------------------------------------------------------------------------------
-- labelling

-- The next clip for the signed-in labeller: an open assignment first (reload, second tab), else a clip that still
-- needs answers and that this person has never been given. Clips already in progress go first so they finish
-- their quorum; then priority; random within ties.
create function public.next_clip()
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

  select c.* into pick
  from public.clips c
  where c.active
    and c.votes + (select count(*) from public.assignments o
                   where o.clip_id = c.id and o.state = 'open' and o.expires_at > now()) < c.target_votes
    and not exists (select 1 from public.assignments m where m.clip_id = c.id and m.user_id = uid)
  order by (c.votes > 0) desc, c.priority desc, random()
  limit 1
  for update of c skip locked;
  if not found then return; end if;

  insert into public.assignments (user_id, clip_id) values (uid, pick.id);
  return query select pick.id, pick.audio_path, pick.duration_s, now() + interval '20 minutes';
end $$;

-- Save the labeller's answer for their assigned clip. Labels must come from the menu; "can't tell" may come with
-- or without a best guess. A clip must have been heard (>= 1.5 s of playback, >= 2 s since it was assigned).
create function public.submit_label(p_clip uuid, p_labels text[], p_other text default null,
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
  update public.clips c set votes = c.votes + 1 where c.id = p_clip;
  return query
    select count(*), count(*) filter (where c.created_at >= date_trunc('day', now()))
    from private.contributions c where c.user_id = uid;
end $$;

-- Pass on the assigned clip (broken audio, uncomfortable, ...): it goes back to the pool for others, never to
-- this person again, and does not count as a contribution.
create function public.skip_clip(p_clip uuid)
returns void
language plpgsql security definer set search_path = '' as $$
begin
  if auth.uid() is null then raise exception 'sign in first' using errcode = '28000'; end if;
  update public.assignments a set state = 'skipped'
  where a.user_id = auth.uid() and a.clip_id = p_clip and a.state = 'open';
end $$;

-- Lets storage hand out a signed URL only for the clip currently assigned to the caller.
create function private.can_hear(p_object text)
returns boolean
language sql stable security definer set search_path = '' as $$
  select exists (
    select 1 from public.assignments a join public.clips c on c.id = a.clip_id
    where a.user_id = auth.uid() and a.state = 'open' and c.audio_path = p_object);
$$;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('clips', 'clips', false, 2097152, array['audio/mpeg']),
       ('guide', 'guide', true, 2097152, array['audio/mpeg', 'application/json'])
on conflict (id) do nothing;

create policy "labellers hear their assigned clip" on storage.objects for select to authenticated
  using (bucket_id = 'clips' and private.can_hear(name));

-- ---------------------------------------------------------------------------------------------------------------
-- contributions: public, by username; site labels and imported kit labels both count; deleted accounts are hidden

create function public.leaderboard(p_days integer default null, p_limit integer default 50)
returns table (rank bigint, username text, display_name text, avatar_url text, labelled bigint, last_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select rank() over (order by count(*) desc), p.username, p.display_name, p.avatar_url, count(*), max(c.created_at)
  from private.contributions c join public.profiles p on p.id = c.user_id
  where p.deleted_at is null and (p_days is null or c.created_at >= now() - make_interval(days => p_days))
  group by p.id
  order by count(*) desc, max(c.created_at)
  limit least(greatest(coalesce(p_limit, 50), 1), 200);
$$;

-- Clips labelled per day over the last year, for the contribution graph (days in the viewer's time zone).
create function public.contributions(p_username text, p_tz text default 'UTC')
returns table (day date, labelled bigint)
language sql stable security definer set search_path = '' as $$
  select (c.created_at at time zone p_tz)::date, count(*)
  from private.contributions c join public.profiles p on p.id = c.user_id
  where p.username = lower(p_username) and p.deleted_at is null and c.created_at >= now() - interval '371 days'
  group by 1 order by 1;
$$;

create function public.profile_stats(p_username text)
returns table (username text, display_name text, avatar_url text, joined timestamptz, labelled bigint, rank bigint)
language sql stable security definer set search_path = '' as $$
  with counts as (
    select c.user_id, count(*) as n, min(c.created_at) as first_at
    from private.contributions c join public.profiles q on q.id = c.user_id and q.deleted_at is null
    group by c.user_id
  ), ranked as (
    select k.user_id, k.n, k.first_at, rank() over (order by k.n desc) as r from counts k
  )
  select p.username, p.display_name, p.avatar_url, least(p.created_at, r.first_at), coalesce(r.n, 0), r.r
  from public.profiles p left join ranked r on r.user_id = p.id
  where p.username = lower(p_username) and p.deleted_at is null;
$$;

create function public.site_stats()
returns table (labels bigint, labellers bigint, clips bigint, clips_complete bigint)
language sql stable security definer set search_path = '' as $$
  select (select count(*) from private.contributions),
         (select count(distinct c.user_id) from private.contributions c),
         (select count(*) from public.clips c where c.active),
         (select count(*) from public.clips c where c.active and c.votes >= c.target_votes);
$$;

-- ---------------------------------------------------------------------------------------------------------------
-- accounts

-- Before signing in or up: an imported kit labeller (account made for them, no password yet) is told to set a
-- password instead. Only those few accounts are ever reported; for every other email the answer is 'ok', so the
-- site cannot be used to find out who has an account.
create function public.account_status(p_email text)
returns text
language sql stable security definer set search_path = '' as $$
  select case when exists (
    select 1 from auth.users u join public.profiles p on p.id = u.id
    where lower(u.email) = lower(btrim(p_email)) and p.imported and p.deleted_at is null
      and coalesce(u.encrypted_password, '') = '') then 'needs_password' else 'ok' end;
$$;

-- Delete the signed-in account: the sign-in goes, the profile leaves every public list, the labels stay.
create function public.delete_account()
returns void
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
begin
  if uid is null then raise exception 'sign in first' using errcode = '28000'; end if;
  update public.profiles p
  set deleted_at = now(), username = 'deleted_' || substr(md5(p.id::text || clock_timestamp()::text), 1, 12),
      display_name = null, avatar_url = null
  where p.id = uid and p.deleted_at is null;
  delete from auth.users u where u.id = uid;
end $$;

-- Make the account for someone who labelled Ear Check kits before the site (run by the import script as postgres):
-- an email sign-in without a password, confirmed, plus their profile. Returns the existing id when it already exists.
create function private.import_contributor(p_email text, p_username text, p_display_name text, p_fusion_name text)
returns uuid
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid;
begin
  select u.id into uid from auth.users u where lower(u.email) = lower(p_email);
  if uid is null then
    uid := gen_random_uuid();
    -- GoTrue reads these token columns as strings: '' not NULL, or sign-in fails for the account
    insert into auth.users (instance_id, id, aud, role, email, encrypted_password, email_confirmed_at,
                            raw_app_meta_data, raw_user_meta_data, created_at, updated_at, confirmation_token,
                            recovery_token, email_change_token_new, email_change, email_change_token_current,
                            phone_change, phone_change_token, reauthentication_token)
    values ('00000000-0000-0000-0000-000000000000', uid, 'authenticated', 'authenticated', lower(p_email), '', now(),
            '{"provider": "email", "providers": ["email"]}', '{}', now(), now(), '', '', '', '', '', '', '', '');
    insert into auth.identities (provider_id, user_id, identity_data, provider, last_sign_in_at, created_at, updated_at)
    values (uid::text, uid, jsonb_build_object('sub', uid::text, 'email', lower(p_email), 'email_verified', true),
            'email', now(), now(), now());
  end if;
  insert into public.profiles (id, username, display_name, fusion_name, imported)
  values (uid, p_username, p_display_name, p_fusion_name, true)
  on conflict (id) do update set fusion_name = excluded.fusion_name, imported = true;
  return uid;
end $$;
revoke all on function private.import_contributor(text, text, text, text) from public, anon, authenticated, service_role;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('avatars', 'avatars', true, 524288, array['image/webp', 'image/png', 'image/jpeg'])
on conflict (id) do nothing;
create policy "upload own avatar" on storage.objects for insert to authenticated
  with check (bucket_id = 'avatars' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "replace own avatar" on storage.objects for update to authenticated
  using (bucket_id = 'avatars' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "remove own avatar" on storage.objects for delete to authenticated
  using (bucket_id = 'avatars' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "list own avatars" on storage.objects for select to authenticated
  using (bucket_id = 'avatars' and (storage.foldername(name))[1] = (select auth.uid())::text);

-- ---------------------------------------------------------------------------------------------------------------
-- admin (service role only): the sync scripts in backend/sync
--
-- Gated inside the function rather than by revoking EXECUTE: on the Supabase Postgres 17.6 image (fixed by 17.11)
-- calling a function the role may not execute crashed the server, so on an older project a revoked function
-- reachable through the API would let anyone restart the database.

create function private.require_service_role()
returns void
language plpgsql stable security definer set search_path = '' as $$
begin
  if coalesce(auth.role(), '') = 'service_role' then return; end if;
  -- a direct database session (psql, migrations) carries no API claims
  if nullif(current_setting('request.jwt.claims', true), '') is null and session_user in ('postgres', 'supabase_admin') then return; end if;
  raise exception 'service role only' using errcode = '42501';
end $$;

-- [{audio_path, source_uid, kind, info, priority, target_votes, duration_s, batch}] -> clips added (existing
-- source_uids get their priority / active / target refreshed instead)
create function public.admin_add_clips(p_clips jsonb)
returns integer
language plpgsql security definer set search_path = '' as $$
declare
  item jsonb;
  cid uuid;
  added integer := 0;
begin
  perform private.require_service_role();
  for item in select * from jsonb_array_elements(p_clips) loop
    select s.clip_id into cid from private.clip_sources s where s.source_uid = item ->> 'source_uid';
    if found then
      update public.clips c set priority = coalesce((item ->> 'priority')::real, c.priority),
                                target_votes = coalesce((item ->> 'target_votes')::smallint, c.target_votes),
                                active = true
      where c.id = cid;
      continue;
    end if;
    insert into public.clips (audio_path, duration_s, target_votes, priority, batch)
    values (item ->> 'audio_path', coalesce((item ->> 'duration_s')::real, 6),
            coalesce((item ->> 'target_votes')::smallint, 3), coalesce((item ->> 'priority')::real, 0),
            coalesce(item ->> 'batch', 'default'))
    returning id into cid;
    insert into private.clip_sources (clip_id, source_uid, kind, info)
    values (cid, item ->> 'source_uid', item ->> 'kind', coalesce(item -> 'info', '{}'));
    added := added + 1;
  end loop;
  return added;
end $$;

create function public.admin_known_sources(p_source_uids text[])
returns setof text
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select s.source_uid from private.clip_sources s where s.source_uid = any (p_source_uids);
$$;

create function public.admin_set_active(p_source_uids text[], p_active boolean)
returns integer
language sql security definer set search_path = '' as $$
  select private.require_service_role();
  with u as (
    update public.clips c set active = p_active
    from private.clip_sources s where s.clip_id = c.id and s.source_uid = any (p_source_uids)
    returning 1)
  select count(*)::integer from u;
$$;

create function public.admin_export_labels(p_since timestamptz default '-infinity')
returns table (label_id bigint, source_uid text, kind text, labeller text, username text, labels text[], other_text text,
               unsure boolean, listen_ms integer, created_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select l.id, s.source_uid, s.kind, coalesce(p.fusion_name, 'crowd:' || left(p.id::text, 8)), p.username, l.labels, l.other_text, l.unsure, l.listen_ms, l.created_at
  from public.labels l
  join private.clip_sources s on s.clip_id = l.clip_id
  join public.profiles p on p.id = l.user_id
  where l.created_at >= p_since
  order by l.id;
$$;

create function public.admin_queue_stats()
returns table (batch text, kind text, clips bigint, complete bigint, votes bigint)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select c.batch, s.kind, count(*), count(*) filter (where c.votes >= c.target_votes), sum(c.votes)
  from public.clips c join private.clip_sources s on s.clip_id = c.id
  where c.active group by 1, 2 order by 1, 2;
$$;

-- Every API function stays executable (see the note above the admin section); each checks its caller itself:
-- labeller functions need a signed-in user (auth.uid()), admin functions the service role.
grant usage on schema private to anon, authenticated, service_role;
revoke all on all functions in schema private from public;
grant execute on function private.can_hear(text), private.require_service_role() to anon, authenticated, service_role;
