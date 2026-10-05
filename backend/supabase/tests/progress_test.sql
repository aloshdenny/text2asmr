-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(10);

create temp table before as select * from public.dataset_progress();
grant select on before to anon, authenticated, service_role;

set local role anon;
set local request.jwt.claims = '{"role": "anon"}';
select is((select count(*) from public.dataset_progress()), 1::bigint, 'anyone can read the ring numbers');
select ok((select computed_at > now() - interval '26 hours' from public.dataset_progress()), 'from a snapshot under a day old');
select throws_ok('select * from private.progress_snapshot', '42501', null, 'the snapshot table itself is private');
select throws_ok('select public.admin_refresh_progress()', '42501', null, 'visitors cannot force a recount');
reset role;

-- a new clip does not move the rings until the next count ...
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select is(public.admin_add_clips('[{"audio_path": "p1.mp3", "source_uid": "pool:progress-test", "kind": "unsure"}]'::jsonb), 1, 'a clip is added');
select is((select foundation_items from public.dataset_progress()), (select foundation_items from before), 'served from the snapshot, not recounted');
-- ... which the pipeline can ask for
select lives_ok('select public.admin_refresh_progress()', 'the service role can recount');
reset role;
-- and a snapshot older than a day is recounted on the next read
update private.progress_snapshot set computed_at = now() - interval '2 days', foundation_items = 0;
select is((select foundation_items from public.dataset_progress()), (select foundation_items + 1 from before), 'a stale snapshot is recounted');

-- a verification-pool clip (core = false) is served but does not count toward the ring
select public.admin_refresh_progress();
create temp table mid as select * from public.dataset_progress();
select is(public.admin_add_clips('[{"audio_path": "p2.mp3", "source_uid": "pool:progress-verify", "kind": "unlabelled", "core": false}]'::jsonb),
          1, 'a verification-pool clip is added');
select public.admin_refresh_progress();
select is((select foundation_items from public.dataset_progress()), (select foundation_items from mid), 'and the ring does not count it');

select * from finish();
rollback;
