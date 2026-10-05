-- Working set: Supabase Storage (1 GB on the free plan) holds only the audio people still need to hear. A clip is
-- taken down (active = false, then its audio deleted; its rows and labels stay) when nobody needs to label it --
-- confident fused labels and no judge yet, or its quorum reached -- and reloaded from the research server's copy
-- when it is needed again (backend/sync/working_set.py). The people ring counts core clips whether or not their
-- audio is up right now: taking audio down does not shrink the training set.

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

-- clips nobody needs to hear now, quorum reached first; never one somebody has open
create function public.admin_takedown_candidates(p_limit integer default 5000)
returns table (clip_id uuid, audio_path text, source_uid text, kind text, votes integer, reason text)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select c.id, c.audio_path, s.source_uid, s.kind, c.votes::integer,
         case when c.votes >= c.target_votes then 'quorum' else 'confident' end
  from public.clips c join private.clip_sources s on s.clip_id = c.id
  where c.active
    and not exists (select 1 from public.assignments a where a.clip_id = c.id and a.state = 'open' and a.expires_at > now())
    and (c.votes >= c.target_votes or (s.kind = 'confident' and c.votes = 0))
  order by (c.votes >= c.target_votes) desc, c.need
  limit least(greatest(coalesce(p_limit, 5000), 1), 20000);
$$;

-- taken-down clips people still need: started ones first, then by need
create function public.admin_reload_candidates(p_limit integer default 2000)
returns table (clip_id uuid, audio_path text, source_uid text, kind text, votes integer, need real)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select c.id, c.audio_path, s.source_uid, s.kind, c.votes::integer, c.need
  from public.clips c join private.clip_sources s on s.clip_id = c.id
  where not c.active and c.votes < c.target_votes and s.kind <> 'confident'
  order by (c.votes > 0) desc, c.need desc
  limit least(greatest(coalesce(p_limit, 2000), 1), 20000);
$$;

create function public.admin_set_active(p_clips uuid[], p_active boolean)
returns integer
language plpgsql security definer set search_path = '' as $$
declare n integer;
begin
  perform private.require_service_role();
  update public.clips c set active = p_active where c.id = any(p_clips) and c.active <> p_active;
  get diagnostics n = row_count;
  return n;
end $$;

create function public.admin_storage_bytes(p_bucket text default 'clips')
returns bigint
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select coalesce(sum((o.metadata ->> 'size')::bigint), 0)::bigint from storage.objects o where o.bucket_id = p_bucket;
$$;

grant execute on function public.admin_takedown_candidates(integer), public.admin_reload_candidates(integer),
  public.admin_set_active(uuid[], boolean), public.admin_storage_bytes(text) to service_role;

select private.refresh_progress();
