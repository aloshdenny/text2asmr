-- Home page rings: dataset_progress() recounted the human-labelled clips on every visit (3.6 s with 21k clips). The
-- counts now come from a snapshot taken once a day (pg_cron, 00:05 UTC), or on demand with admin_refresh_progress()
-- after a pipeline run. The site's /api/progress edge route caches the answer on Vercel's CDN on top of that.

create table private.progress_snapshot (
  id boolean primary key default true check (id),     -- one row
  foundation_items bigint not null,
  foundation_human_labelled bigint not null,
  corpus_recordings bigint not null,
  ai_labelled_recordings bigint not null,
  computed_at timestamptz not null default now()
);
revoke all on private.progress_snapshot from public, anon, authenticated;

-- the counting itself (as 20261003000000_synthetic_profiles.sql), written into the snapshot
create function private.refresh_progress()
returns void
language sql security definer set search_path = '' as $$
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

drop function public.dataset_progress();
create function public.dataset_progress()
returns table (foundation_items bigint, foundation_human_labelled bigint, corpus_recordings bigint, ai_labelled_recordings bigint,
               computed_at timestamptz)
language plpgsql security definer set search_path = '' as $$
begin
  -- never counted, or the daily job has not run for over a day: count now (once), then serve the snapshot
  if not exists (select 1 from private.progress_snapshot s where s.computed_at > now() - interval '26 hours') then
    perform private.refresh_progress();
  end if;
  return query select s.foundation_items, s.foundation_human_labelled, s.corpus_recordings, s.ai_labelled_recordings, s.computed_at
               from private.progress_snapshot s;
end $$;

create function public.admin_refresh_progress()
returns void
language plpgsql security definer set search_path = '' as $$
begin
  perform private.require_service_role();
  perform private.refresh_progress();
end $$;

revoke all on function private.refresh_progress() from public;
grant execute on function public.dataset_progress() to anon, authenticated, service_role;
grant execute on function public.admin_refresh_progress() to service_role;

select private.refresh_progress();
select cron.schedule('refresh-progress', '5 0 * * *', 'select private.refresh_progress()');
