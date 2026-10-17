-- Two fixes for a small crowd.
--
-- 1. Breadth first. next_clip topped up every started clip to its full quorum (target_votes, 12) before opening a fresh
--    one. With three active labellers that meant every label went to the ~360 clips already started, and none of the
--    10,000+ thin-class clips queued since 2026-10-08 was heard once. Started clips are now topped up only to
--    `breadth_votes` judges (setting, default 3: enough for a majority); after that the neediest fresh clip comes first,
--    and clips keep collecting judges up to target_votes through the need ranking as before.
--
-- 2. The people ring counts labels. Counting clips with at least one human label stood still while people topped up
--    started clips (742 for a week with 74 new labels). The snapshot now also carries human_labels -- every answer
--    people gave on the site and in the Ear Check kits -- and is refreshed hourly instead of daily.

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

  -- then bring started clips up to `breadth` judges, fullest first, so no clip is left with a lone answer
  if not got then
    select c.* into pick from public.clips c
    where c.active and c.votes > 0 and c.votes < least(c.target_votes, breadth)
      and not exists (select 1 from public.assignments m where m.user_id = uid and m.clip_id = c.id)
    order by c.votes desc, c.need desc, random()
    limit 1
    for update of c skip locked;
    if found then got := true; end if;
  end if;

  -- else the neediest clip they have never been given (fresh clips rank first; started ones keep filling to target)
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

-- the people ring: every answer people gave (site labels that name sounds, and Ear Check kit labels)
alter table private.progress_snapshot add column human_labels bigint not null default 0;

create or replace function private.refresh_progress()
returns void
language sql security definer set search_path = '' as $$
  with real as (
    select count(*) as items,
           count(*) filter (where exists (select 1 from public.labels l where l.clip_id = c.id and not l.unsure)
                              or exists (select 1 from public.imported_labels i where i.source_uid = s.source_uid)) as labelled
    from public.clips c join private.clip_sources s on s.clip_id = c.id
    where c.core
  ), seed as (
    select count(*) as labelled
    from public.imported_labels i join public.profiles p on p.id = i.user_id
    where p.synthetic and i.kit = 'synthetic'
  ), answers as (
    select (select count(*) from public.labels l where not l.unsure)
         + (select count(*) from public.imported_labels i join public.profiles p on p.id = i.user_id where not p.synthetic) as n
  )
  insert into private.progress_snapshot (id, foundation_items, foundation_human_labelled, corpus_recordings, ai_labelled_recordings,
                                         human_labels, computed_at)
  select true, r.items, least(r.items, r.labelled + (select labelled from seed)),
         coalesce((select value::bigint from private.settings where key = 'corpus_recordings'), 0),
         coalesce((select value::bigint from private.settings where key = 'ai_labelled_recordings'), 0),
         (select n from answers), now()
  from real r
  on conflict (id) do update set foundation_items = excluded.foundation_items,
                                 foundation_human_labelled = excluded.foundation_human_labelled,
                                 corpus_recordings = excluded.corpus_recordings,
                                 ai_labelled_recordings = excluded.ai_labelled_recordings,
                                 human_labels = excluded.human_labels,
                                 computed_at = excluded.computed_at;
$$;

drop function public.dataset_progress();
create function public.dataset_progress()
returns table (foundation_items bigint, foundation_human_labelled bigint, corpus_recordings bigint, ai_labelled_recordings bigint,
               human_labels bigint, computed_at timestamptz)
language plpgsql security definer set search_path = '' as $$
begin
  -- never counted, or the hourly job has not run for a while: count now (once), then serve the snapshot
  if not exists (select 1 from private.progress_snapshot s where s.computed_at > now() - interval '2 hours') then
    perform private.refresh_progress();
  end if;
  return query select s.foundation_items, s.foundation_human_labelled, s.corpus_recordings, s.ai_labelled_recordings,
                      s.human_labels, s.computed_at
               from private.progress_snapshot s;
end $$;
grant execute on function public.dataset_progress() to anon, authenticated, service_role;

select cron.unschedule('refresh-progress');
select cron.schedule('refresh-progress', '5 * * * *', 'select private.refresh_progress()');
select private.refresh_progress();
