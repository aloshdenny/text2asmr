-- Thin classes first. People have labelled spraying, cutting, sticky, brushing, paper rustling, liquid, background music
-- and room tone far less than the vocal classes (9 to 43 labels each against 130 to 230), and those are the classes no
-- machine judge hears reliably. Clips whose title / search class (info.title_class, page turning counting as paper
-- rustling) or whose unsure classes (info.unsure) include one of them carry a boost, and next_clip serves them before
-- everything else: thin clips already started are topped up to breadth_votes, then the neediest fresh thin clip, and
-- only then the rest as before. The class list is a table (private.thin_classes); new clips get their boost when they
-- are queued; admin_refresh_boost() recomputes every clip after the list or the 'thin_boost' setting changes.

create table private.thin_classes (label text primary key);
revoke all on private.thin_classes from public, anon, authenticated;
insert into private.thin_classes (label) values
  ('spraying'), ('cutting'), ('sticky'), ('brushing'), ('paper rustling'), ('liquid'), ('background music'), ('silence / room tone');

alter table public.clips add column boost real not null default 0;

create function private.thin_boost(p_info jsonb)
returns real
language sql stable set search_path = '' as $$
  select case when exists (
    select 1 from private.thin_classes t
    where t.label = case p_info ->> 'title_class' when 'page turning' then 'paper rustling' else p_info ->> 'title_class' end
       or (jsonb_typeof(p_info -> 'unsure') = 'array' and p_info -> 'unsure' ? t.label)
  ) then private.setting('thin_boost', 0.3)::real else 0 end;
$$;

create function private.set_clip_boost()
returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  update public.clips set boost = private.thin_boost(new.info) where id = new.clip_id;
  return new;
end $$;
create trigger clip_sources_boost after insert or update of info on private.clip_sources
  for each row execute function private.set_clip_boost();

create function public.admin_refresh_boost()
returns integer
language plpgsql security definer set search_path = '' as $$
declare
  n integer;
begin
  perform private.require_service_role();
  update public.clips c set boost = private.thin_boost(s.info)
  from private.clip_sources s where s.clip_id = c.id and c.boost is distinct from private.thin_boost(s.info);
  get diagnostics n = row_count;
  return n;
end $$;
revoke all on function public.admin_refresh_boost() from public, anon, authenticated;
grant execute on function public.admin_refresh_boost() to service_role;

update public.clips c set boost = private.thin_boost(s.info) from private.clip_sources s where s.clip_id = c.id;

-- serving: the breadth-first order of 20261017000000, run twice -- thin clips only, then any clip
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
  breadth integer := private.setting('breadth_votes', 3)::integer;
  ctrl_rate numeric;
  thin_only boolean;
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
