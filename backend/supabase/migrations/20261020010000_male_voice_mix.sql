-- Male voices in the mix. Labellers said they never got "daddy" clips: the male-voice corpus (aoxo/t2a-daddy) is 8% of
-- the active queue (1,994 clips) and under 3% of the boosted clips that are served first, so 106 of its clips had ever
-- been served. Each clip now knows its voice (clips.voice, from info.voice: 'male' for t2a-daddy, 'female' for
-- t2a-mommy, null for trigger recordings), and a `male_share` of draws (0.3) looks among male-voice clips first, with
-- the usual order inside them (started clips, then boosted, then neediest), falling back to every clip when none is left.

alter table public.clips add column voice text check (voice in ('male', 'female'));
create index clips_active_male on public.clips (need desc) where active and voice = 'male';

create function private.set_clip_voice()
returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  update public.clips c set voice = case when new.info ->> 'voice' in ('male', 'female') then new.info ->> 'voice' end
  where c.id = new.clip_id and c.voice is distinct from (case when new.info ->> 'voice' in ('male', 'female') then new.info ->> 'voice' end);
  return new;
end $$;
create trigger clip_sources_voice after insert or update of info on private.clip_sources
  for each row execute function private.set_clip_voice();
revoke all on function private.set_clip_voice() from public, anon, authenticated;

-- merge keys into queued clips' info: [{"source_uid": ..., "info": {...}}] (the voice and boost triggers follow)
create function public.admin_merge_info(p_rows jsonb)
returns integer
language plpgsql security definer set search_path = '' as $$
declare
  n integer;
begin
  perform private.require_service_role();
  update private.clip_sources s set info = s.info || r.info
  from jsonb_to_recordset(p_rows) as r(source_uid text, info jsonb)
  where s.source_uid = r.source_uid and not (s.info @> r.info);
  get diagnostics n = row_count;
  return n;
end $$;
revoke all on function public.admin_merge_info(jsonb) from public, anon, authenticated;
grant execute on function public.admin_merge_info(jsonb) to service_role;

insert into private.settings (key, value) values ('male_share', 0.3) on conflict (key) do nothing;

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
  gap integer;
  breadth integer := private.setting('breadth_votes', 3)::integer;
  ctrl_rate numeric;
  thin_only boolean;
  audited boolean;
  male_only boolean;
  passes boolean[] := case when random() < private.setting('male_share', 0.3) then array[true, false] else array[false] end;
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
  audited := exists (select 1 from private.labeller_quality q where q.user_id = uid and q.status in ('watch', 'weak'));
  gap := least(private.setting('repeat_gap', 300)::integer,
               greatest(private.setting('repeat_gap_min', 30)::integer, n / 2));
  if audited then
    rep := private.setting('audit_repeat_rate', 0.12);
    gap := private.setting('audit_repeat_gap', 15)::integer;
  end if;

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

  -- a control: often while we know little about this person (or they are on watch), rarely once they have answered many
  if not got then
    select greatest(private.setting('control_rate_min', 0.05),
                    private.setting('control_rate_max', 0.25) - 0.01 * count(*)) into ctrl_rate
    from public.labels l join private.clip_sources s on s.clip_id = l.clip_id
    where l.user_id = uid and s.kind = 'control';
    if audited then ctrl_rate := greatest(ctrl_rate, private.setting('audit_control_rate', 0.3)); end if;
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

  -- male voices get their share of the draws (`male_share`): first look among male-voice clips only, then anywhere
  foreach male_only in array passes loop
  exit when got;
  foreach thin_only in array array[true, false] loop
    exit when got;

    -- started clips up to `breadth` judges, fullest first
    select c.* into pick from public.clips c
    where c.active and (c.boost > 0 or not thin_only) and c.votes > 0 and c.votes < least(c.target_votes, breadth)
      and (c.voice = 'male' or not male_only)
      and not exists (select 1 from public.assignments m where m.user_id = uid and m.clip_id = c.id)
    order by c.votes desc, c.need + c.boost desc, random()
    limit 1
    for update of c skip locked;
    if found then got := true; exit; end if;

    -- the neediest clip they have never been given (fresh clips rank first; started ones keep filling to target)
    with candidates as (
      select c.id, c.need + c.boost as score from public.clips c
      where c.active and (c.boost > 0 or not thin_only) and (c.voice = 'male' or not male_only)
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
  end loop;
  if not got then return; end if;

  insert into public.assignments (user_id, clip_id, attempt) values (uid, pick.id, att);
  return query select pick.id, pick.audio_path, pick.duration_s, now() + interval '20 minutes';
end $$;
