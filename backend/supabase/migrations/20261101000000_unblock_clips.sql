-- Unblock clips by source uid. 2026-11-01 the content filter narrowed to minor-coded material only (minors, school / teen
-- settings, age play, parent-child incest); clips blocked only for adult sibling / cousin / aunt / step-family roleplay
-- come back. An unblocked clip stays inactive with no audio: working_set.py --reload uploads it again when people
-- still need it, like any taken-down clip. Returns how many clips were unblocked.

create function public.admin_unblock_clips(p_source_uids text[])
returns integer
language plpgsql security definer set search_path = '' as $$
declare n integer;
begin
  perform private.require_service_role();
  update public.clips c set blocked = false
  from private.clip_sources s
  where s.clip_id = c.id and s.source_uid = any(p_source_uids) and c.blocked;
  get diagnostics n = row_count;
  return n;
end $$;

grant execute on function public.admin_unblock_clips(text[]) to service_role;
