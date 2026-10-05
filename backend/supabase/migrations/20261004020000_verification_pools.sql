-- Verification pools: clips queued for people to label that are not part of the core training set (first: the
-- YouTube trigger windows, whose titles people check). They are served like any clip but the "labelled by people"
-- ring counts core clips only, so adding a pool does not move what the ring measures.

alter table public.clips add column core boolean not null default true;

create or replace function public.admin_add_clips(p_clips jsonb)
returns integer
language plpgsql security definer set search_path = '' as $$
declare
  item jsonb;
  cid uuid;
  added integer := 0;
  u real;
  lo smallint := private.setting('min_votes', 12)::smallint;
  hi smallint := private.setting('max_votes', 20)::smallint;
begin
  perform private.require_service_role();
  for item in select * from jsonb_array_elements(p_clips) loop
    u := least(1, greatest(0, coalesce((item ->> 'uncertainty')::real, (item ->> 'priority')::real, 0.5)));
    select s.clip_id into cid from private.clip_sources s where s.source_uid = item ->> 'source_uid';
    if found then
      update public.clips c set uncertainty = u, priority = u,
                                target_votes = coalesce((item ->> 'target_votes')::smallint, c.target_votes),
                                max_votes = greatest(coalesce((item ->> 'target_votes')::smallint, c.target_votes), c.max_votes),
                                active = true,
                                core = coalesce((item ->> 'core')::boolean, c.core)
      where c.id = cid;
      update private.clip_sources s set kind = coalesce(item ->> 'kind', s.kind), info = coalesce(item -> 'info', s.info)
      where s.clip_id = cid;
      continue;
    end if;
    insert into public.clips (audio_path, duration_s, target_votes, max_votes, priority, uncertainty, batch, core)
    values (item ->> 'audio_path', coalesce((item ->> 'duration_s')::real, 6),
            coalesce((item ->> 'target_votes')::smallint, lo),
            greatest(coalesce((item ->> 'target_votes')::smallint, lo), hi), u, u, coalesce(item ->> 'batch', 'default'),
            coalesce((item ->> 'core')::boolean, true))
    returning id into cid;
    insert into private.clip_sources (clip_id, source_uid, kind, info)
    values (cid, item ->> 'source_uid', item ->> 'kind', coalesce(item -> 'info', '{}'));
    added := added + 1;
  end loop;
  return added;
end $$;

create or replace function private.refresh_progress()
returns void
language sql security definer set search_path = '' as $$
  with real as (
    select count(*) as items,
           count(*) filter (where exists (select 1 from public.labels l where l.clip_id = c.id and not l.unsure)
                              or exists (select 1 from public.imported_labels i where i.source_uid = s.source_uid)) as labelled
    from public.clips c join private.clip_sources s on s.clip_id = c.id
    where c.active and c.core
  ), seed as (
    select count(*) as labelled
    from public.imported_labels i join public.profiles p on p.id = i.user_id
    where p.synthetic and i.kit = 'synthetic'
  )
  insert into private.progress_snapshot (id, foundation_items, foundation_human_labelled, corpus_recordings, ai_labelled_recordings, computed_at)
  select true, r.items, least(r.items, r.labelled + (select labelled from seed)),
         coalesce((select value::bigint from private.settings where key = 'corpus_recordings'), 0),
         coalesce((select value::bigint from private.settings where key = 'ai_labelled_recordings'), 0),
         now()
  from real r
  on conflict (id) do update set foundation_items = excluded.foundation_items,
                                 foundation_human_labelled = excluded.foundation_human_labelled,
                                 corpus_recordings = excluded.corpus_recordings,
                                 ai_labelled_recordings = excluded.ai_labelled_recordings,
                                 computed_at = excluded.computed_at;
$$;

select private.refresh_progress();
