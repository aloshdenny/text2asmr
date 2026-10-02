-- The two rings measure different things, each against its own total:
--   people: clips in the core training set (every active clip in the labelling queue) that at least one person has
--           labelled, on the site or in the Ear Check kits -- counted live
--   AI:     recordings on Hugging Face (t2a-mommy, t2a-daddy, t2a-audios-v1, asmr-yt-chapters) that the CLAP-ASMR
--           pipeline has labelled -- both numbers set by the pipeline through admin_set_setting

delete from private.settings where key in ('corpus_items', 'ai_labelled_items');
insert into private.settings (key, value) values ('corpus_recordings', 0), ('ai_labelled_recordings', 0)
on conflict (key) do nothing;

drop function public.dataset_progress();
create function public.dataset_progress()
returns table (foundation_items bigint, foundation_human_labelled bigint, corpus_recordings bigint, ai_labelled_recordings bigint)
language sql stable security definer set search_path = '' as $$
  select count(*),
         count(*) filter (where exists (select 1 from public.labels l where l.clip_id = c.id and not l.unsure)
                            or exists (select 1 from public.imported_labels i where i.source_uid = s.source_uid)),
         (select value::bigint from private.settings where key = 'corpus_recordings'),
         (select value::bigint from private.settings where key = 'ai_labelled_recordings')
  from public.clips c join private.clip_sources s on s.clip_id = c.id
  where c.active;
$$;
