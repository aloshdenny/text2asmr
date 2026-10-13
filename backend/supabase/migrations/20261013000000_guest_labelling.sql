-- Guests: label without creating an account.
--
-- A visitor ticks the 18+ box and gets a Supabase anonymous user (auth.signInAnonymously), and start_guest() gives them
-- a profile (guest_<12 hex>), so serving, check clips, retests and per-person trust work exactly as for everyone else.
-- Guests stay off the leaderboard and off public profile pages, but their labels count toward the dataset like anyone's.
-- Linking an email later keeps the same user id, so the labels come along; the first profile save made once the user
-- is no longer anonymous (choosing a username) clears the guest flag.

alter table public.profiles add column guest boolean not null default false;
grant select (guest) on public.profiles to anon, authenticated;

-- The guest profile for the signed-in anonymous user; does nothing when they already have one (a second tab, a reload).
create function public.start_guest()
returns void
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := (select auth.uid());
begin
  if uid is null then
    raise exception 'sign in first' using errcode = '42501';
  end if;
  if coalesce((auth.jwt() ->> 'is_anonymous')::boolean, false) is not true then
    raise exception 'only guest sessions start as guests' using errcode = '42501';
  end if;
  insert into public.profiles (id, username, guest, onboarded)
  values (uid, 'guest_' || substr(replace(uid::text, '-', ''), 1, 12), true, true)
  on conflict (id) do nothing;
end $$;
revoke all on function public.start_guest() from public, anon;
grant execute on function public.start_guest() to authenticated;

-- A guest who has linked an email (no longer anonymous) becomes a full labeller the first time they save their profile.
-- Reads auth.users, not the JWT: the access token keeps its is_anonymous claim until the next refresh, so right after
-- linking an email the token still says anonymous while the account no longer is.
create function private.clear_guest()
returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  if old.guest and exists (select 1 from auth.users u where u.id = new.id and not u.is_anonymous) then
    new.guest := false;
  end if;
  return new;
end $$;
create trigger profiles_clear_guest before update on public.profiles
  for each row execute function private.clear_guest();

-- The board: people with accounts only.
create or replace function public.leaderboard(p_days integer default null, p_limit integer default 50)
returns table (rank bigint, username text, display_name text, avatar_url text, labelled bigint, last_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select row_number() over (order by count(*) desc, max(c.created_at), p.username), p.username, p.display_name, p.avatar_url,
         count(*), max(c.created_at)
  from private.contributions c join public.profiles p on p.id = c.user_id
  where p.deleted_at is null and not p.guest and (p_days is null or c.created_at >= now() - make_interval(days => p_days))
  group by p.id
  order by count(*) desc, max(c.created_at), p.username
  limit least(greatest(coalesce(p_limit, 50), 1), 200);
$$;

-- A profile's numbers: public for people with accounts; a guest sees only their own (no rank: guests are not ranked).
create or replace function public.profile_stats(p_username text)
returns table (username text, display_name text, avatar_url text, joined timestamptz, labelled bigint, rank bigint)
language sql stable security definer set search_path = '' as $$
  with counts as (
    select c.user_id, q.username, q.guest, count(*) as n, min(c.created_at) as first_at, max(c.created_at) as last_at
    from private.contributions c join public.profiles q on q.id = c.user_id and q.deleted_at is null
    group by c.user_id, q.username, q.guest
  ), ranked as (
    select k.user_id, k.n, k.first_at,
           case when not k.guest then row_number() over (partition by k.guest order by k.n desc, k.last_at, k.username) end as r
    from counts k
  )
  select p.username, p.display_name, p.avatar_url, least(p.created_at, r.first_at), coalesce(r.n, 0), r.r
  from public.profiles p left join ranked r on r.user_id = p.id
  where p.username = lower(p_username) and p.deleted_at is null and (not p.guest or p.id = (select auth.uid()));
$$;

create or replace function public.contributions(p_username text, p_tz text default 'UTC')
returns table (day date, labelled bigint)
language sql stable security definer set search_path = '' as $$
  select (c.created_at at time zone p_tz)::date, count(*)
  from private.contributions c join public.profiles p on p.id = c.user_id
  where p.username = lower(p_username) and p.deleted_at is null and (not p.guest or p.id = (select auth.uid()))
    and c.created_at >= now() - interval '371 days'
  group by 1 order by 1;
$$;
