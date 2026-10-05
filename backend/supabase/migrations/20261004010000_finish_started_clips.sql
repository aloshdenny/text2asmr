-- Serving fix: next_clip ranked candidates by need alone, and a clip nobody has answered (need ~0.74) always beat one
-- with an answer (~0.70), so each answer went to a new clip and no clip collected the judges the consensus needs.
-- In-progress clips now come first (after retests and controls), fullest first.

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

  -- otherwise finish what others started: a clip that has judges but not yet its quorum and that they have not
  -- heard, most judges first, so each clip reaches min_votes before fresh ones are opened (ranking by need alone put
  -- every answer on a new clip: 340 of 342 clips had a single judge)
  if not got then
    select c.* into pick from public.clips c
    where c.active and c.votes > 0 and c.votes < c.target_votes
      and not exists (select 1 from public.assignments m where m.user_id = uid and m.clip_id = c.id)
    order by c.votes desc, c.need desc, random()
    limit 1
    for update of c skip locked;
    if found then got := true; end if;
  end if;

  -- else the neediest fresh clip they have never been given
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
