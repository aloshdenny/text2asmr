-- With taken-down clips (working_set.py) a refresh must not switch a clip back on: its audio may be gone. Refreshing
-- a known source now updates its numbers, kind and info but leaves `active` alone; only working_set.py --reload,
-- which uploads the audio first, brings a clip back.

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
