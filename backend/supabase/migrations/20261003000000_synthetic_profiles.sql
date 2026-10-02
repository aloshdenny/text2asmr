-- Synthetic (seed) profiles: placeholder labellers that make an empty board look lived in until organic activity
-- takes over. Their labels live in imported_labels under kit 'synthetic' with placeholder source uids, so they never
-- match a real clip, never reach exports or the fusion, and only count toward the leaderboard and the people ring.
-- The flag is not readable through the API. Remove them all with:
--   delete from public.imported_labels where kit = 'synthetic';
--   delete from public.profiles where synthetic;

alter table public.profiles add column synthetic boolean not null default false;

create or replace function public.dataset_progress()
returns table (foundation_items bigint, foundation_human_labelled bigint, corpus_recordings bigint, ai_labelled_recordings bigint)
language sql stable security definer set search_path = '' as $$
  with real as (
    select count(*) as items,
           count(*) filter (where exists (select 1 from public.labels l where l.clip_id = c.id and not l.unsure)
                              or exists (select 1 from public.imported_labels i where i.source_uid = s.source_uid)) as labelled
    from public.clips c join private.clip_sources s on s.clip_id = c.id
    where c.active
  ), seed as (
    select count(*) as labelled
    from public.imported_labels i join public.profiles p on p.id = i.user_id
    where p.synthetic and i.kit = 'synthetic'
  )
  select r.items,
         least(r.items, r.labelled + (select labelled from seed)),
         (select value::bigint from private.settings where key = 'corpus_recordings'),
         (select value::bigint from private.settings where key = 'ai_labelled_recordings')
  from real r;
$$;
