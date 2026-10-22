-- Referral links: asmrboard.vercel.app/r/<handle>, one per creator (their Instagram handle). The browser keeps the
-- handle until the visitor has a profile (guest or member), then set_referral credits it once, and only to a profile
-- made in the last day, so people who were already labelling are not counted. A guest who signs up keeps the credit
-- (same profile). Private: no column grant; admin_referrals reports people and labels per handle.

alter table public.profiles add column referred_by text check (referred_by ~ '^[a-z0-9._]{1,30}$');

create function public.set_referral(p_ref text)
returns void
language plpgsql security definer set search_path = '' as $$
declare
  r text := lower(trim(p_ref));
begin
  if (select auth.uid()) is null then raise exception 'sign in first' using errcode = '28000'; end if;
  if r is null or r !~ '^[a-z0-9._]{1,30}$' then return; end if;
  update public.profiles p set referred_by = r
  where p.id = (select auth.uid()) and p.referred_by is null and p.created_at > now() - interval '1 day';
end $$;
revoke all on function public.set_referral(text) from public, anon;
grant execute on function public.set_referral(text) to authenticated;

create function public.admin_referrals()
returns table (ref text, people bigint, members bigint, guests bigint, labelling bigint, labels bigint,
               first_at timestamptz, last_label_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select p.referred_by, count(*), count(*) filter (where not p.guest), count(*) filter (where p.guest),
         count(*) filter (where l.n > 0), coalesce(sum(l.n), 0)::bigint, min(p.created_at), max(l.last_at)
  from public.profiles p
  left join lateral (select count(*) as n, max(x.created_at) as last_at from public.labels x where x.user_id = p.id) l on true
  where p.referred_by is not null and p.deleted_at is null
  group by p.referred_by
  order by coalesce(sum(l.n), 0) desc, count(*) desc;
$$;
revoke all on function public.admin_referrals() from public, anon, authenticated;
grant execute on function public.admin_referrals() to service_role;
