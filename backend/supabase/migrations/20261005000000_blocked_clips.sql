-- Blocked clips: audio that must not be served or trained on -- recordings whose titles put minors, school / teen
-- settings, age play or incest in a sexual scene. A blocked clip is taken down (active = false, audio deleted by the
-- caller) and can never come back: reloads skip it, a refresh never re-activates anything, and the people ring does
-- not count it. Its rows stay, so the decision is visible and reversible by hand.

alter table public.clips add column blocked boolean not null default false;

create or replace function public.admin_reload_candidates(p_limit integer default 2000)
returns table (clip_id uuid, audio_path text, source_uid text, kind text, votes integer, need real)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select c.id, c.audio_path, s.source_uid, s.kind, c.votes::integer, c.need
  from public.clips c join private.clip_sources s on s.clip_id = c.id
  where not c.active and not c.blocked and c.votes < c.target_votes and s.kind <> 'confident'
  order by (c.votes > 0) desc, c.need desc
  limit least(greatest(coalesce(p_limit, 2000), 1), 20000);
$$;

create or replace function private.refresh_progress()
returns void
language sql security definer set search_path = '' as $$
  with real as (
    select count(*) as items,
           count(*) filter (where exists (select 1 from public.labels l where l.clip_id = c.id and not l.unsure)
                              or exists (select 1 from public.imported_labels i where i.source_uid = s.source_uid)) as labelled
    from public.clips c join private.clip_sources s on s.clip_id = c.id
    where c.core and not c.blocked
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

-- block clips by source uid: returns each newly blocked clip's audio object so the caller can delete it
create function public.admin_block_clips(p_source_uids text[])
returns table (clip_id uuid, audio_path text)
language plpgsql security definer set search_path = '' as $$
begin
  perform private.require_service_role();
  return query
    update public.clips c set blocked = true, active = false
    from private.clip_sources s
    where s.clip_id = c.id and s.source_uid = any(p_source_uids) and not c.blocked
    returning c.id, c.audio_path;
end $$;

grant execute on function public.admin_block_clips(text[]) to service_role;
select private.refresh_progress();
