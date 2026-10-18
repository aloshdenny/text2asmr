-- Serve the new labels first. Clapping and the NSFW group (20261018000000) have no people's answers yet and no machine
-- judge that hears them, so they join the thin classes. A clip's title_tags (backend/sync/title_tags.py: the sounds its
-- recording's title promises) now count toward the boost like its title class and unsure classes do, and a clip
-- tagged with one of these brand-new labels gets the larger 'new_label_boost' (0.5) so it reaches people before the
-- older thin classes (0.3).

insert into private.thin_classes (label) values
  ('clapping'), ('sex slapping'), ('wet sounds'), ('squirting'), ('spanking'), ('vibrator'), ('gagging'),
  ('whimpering'), ('growling'), ('grunting'), ('bed creaking')
on conflict do nothing;

create table private.new_labels (label text primary key);
revoke all on private.new_labels from public, anon, authenticated;
insert into private.new_labels (label)
  select key from public.label_options where added_at > '2026-10-17';

create or replace function private.thin_boost(p_info jsonb)
returns real
language sql stable set search_path = '' as $$
  select case
    when jsonb_typeof(p_info -> 'title_tags') = 'array'
         and exists (select 1 from private.new_labels n where p_info -> 'title_tags' ? n.label)
      then private.setting('new_label_boost', 0.5)::real
    when exists (
      select 1 from private.thin_classes t
      where t.label = case p_info ->> 'title_class' when 'page turning' then 'paper rustling' else p_info ->> 'title_class' end
         or (jsonb_typeof(p_info -> 'unsure') = 'array' and p_info -> 'unsure' ? t.label)
         or (jsonb_typeof(p_info -> 'title_tags') = 'array' and p_info -> 'title_tags' ? t.label))
      then private.setting('thin_boost', 0.3)::real
    else 0 end;
$$;

-- tag clips already queued: [{"source_uid": ..., "title_tags": [...]}]; the clip_sources trigger recomputes the boost
create function public.admin_tag_clips(p_rows jsonb)
returns integer
language plpgsql security definer set search_path = '' as $$
declare
  n integer;
begin
  perform private.require_service_role();
  update private.clip_sources s set info = s.info || jsonb_build_object('title_tags', r.title_tags)
  from jsonb_to_recordset(p_rows) as r(source_uid text, title_tags jsonb)
  where s.source_uid = r.source_uid and s.info -> 'title_tags' is distinct from r.title_tags;
  get diagnostics n = row_count;
  return n;
end $$;
revoke all on function public.admin_tag_clips(jsonb) from public, anon, authenticated;
grant execute on function public.admin_tag_clips(jsonb) to service_role;
