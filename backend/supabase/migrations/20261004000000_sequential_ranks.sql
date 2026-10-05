-- Ranks were rank(): everyone with the same count shared a number and the next numbers were skipped (76, 76, 76,
-- 79), which reads like a bug at the bottom of the board where many people have one or two labels. Ranks are now
-- 1, 2, 3, ... in the board's own order: more labels first, then whoever reached that count first, then username.

create or replace function public.leaderboard(p_days integer default null, p_limit integer default 50)
returns table (rank bigint, username text, display_name text, avatar_url text, labelled bigint, last_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select row_number() over (order by count(*) desc, max(c.created_at), p.username), p.username, p.display_name, p.avatar_url,
         count(*), max(c.created_at)
  from private.contributions c join public.profiles p on p.id = c.user_id
  where p.deleted_at is null and (p_days is null or c.created_at >= now() - make_interval(days => p_days))
  group by p.id
  order by count(*) desc, max(c.created_at), p.username
  limit least(greatest(coalesce(p_limit, 50), 1), 200);
$$;

create or replace function public.profile_stats(p_username text)
returns table (username text, display_name text, avatar_url text, joined timestamptz, labelled bigint, rank bigint)
language sql stable security definer set search_path = '' as $$
  with counts as (
    select c.user_id, q.username, count(*) as n, min(c.created_at) as first_at, max(c.created_at) as last_at
    from private.contributions c join public.profiles q on q.id = c.user_id and q.deleted_at is null
    group by c.user_id, q.username
  ), ranked as (
    select k.user_id, k.n, k.first_at, row_number() over (order by k.n desc, k.last_at, k.username) as r from counts k
  )
  select p.username, p.display_name, p.avatar_url, least(p.created_at, r.first_at), coalesce(r.n, 0), r.r
  from public.profiles p left join ranked r on r.user_id = p.id
  where p.username = lower(p_username) and p.deleted_at is null;
$$;
