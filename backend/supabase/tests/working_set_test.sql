-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(12);

update public.clips set active = false;
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select is(public.admin_add_clips('[
  {"audio_path": "w-conf.mp3", "source_uid": "pool:w-conf", "kind": "confident"},
  {"audio_path": "w-unsure.mp3", "source_uid": "pool:w-unsure", "kind": "unsure", "uncertainty": 1.0},
  {"audio_path": "w-weak.mp3", "source_uid": "pool:w-weak", "kind": "unlabelled", "core": false}
]'::jsonb), 3, 'three clips queued');
select public.admin_refresh_progress();
reset role;
create temp table before as select * from public.dataset_progress();
create temp table ids as select c.id, c.audio_path from public.clips c where c.audio_path like 'w-%';
grant select on before, ids to service_role, anon;

set local role service_role;
select is((select array_agg(audio_path) from public.admin_takedown_candidates(100) where audio_path like 'w-%'), array['w-conf.mp3'],
          'only the confident clip nobody has judged comes down');
select is(public.admin_set_active(array(select id from ids where audio_path = 'w-conf.mp3'), false), 1, 'it is taken down');
select is(public.admin_set_active(array(select id from ids where audio_path = 'w-conf.mp3'), false), 0, 'taking it down again changes nothing');
select public.admin_refresh_progress();
select is((select foundation_items from public.dataset_progress()), (select foundation_items from before),
          'the ring still counts a core clip whose audio is down');
select ok(public.admin_storage_bytes() >= 0, 'storage usage is readable');
-- re-queueing a taken-down clip (push_clips refresh) must not switch it back on: its audio is gone
select public.admin_add_clips('[{"audio_path": "w-conf2.mp3", "source_uid": "pool:w-conf", "kind": "unsure", "uncertainty": 1.0}]'::jsonb);
select is((select count(*) from public.admin_reload_candidates(5000) where audio_path = 'w-conf.mp3'), 1::bigint,
          'refreshed to unsure, it stays down until a reload uploads it');
-- an unsure clip taken down for space comes back first; a confident one never does
select public.admin_set_active(array(select id from ids where audio_path = 'w-unsure.mp3'), false);
select is((select array_agg(audio_path order by audio_path) from public.admin_reload_candidates(5000) where audio_path like 'w-%'), array['w-conf.mp3', 'w-unsure.mp3'],
          'reload candidates: taken-down clips people still need, neediest first');
-- a blocked clip comes down and never comes back
select is((select count(*) from public.admin_block_clips(array['pool:w-unsure'])), 1::bigint, 'blocking takes the clip down');
select is((select count(*) from public.admin_reload_candidates(5000) where audio_path = 'w-unsure.mp3'), 0::bigint,
          'a blocked clip is never a reload candidate');
reset role;

set local role anon;
set local request.jwt.claims = '{"role": "anon"}';
select throws_ok('select * from public.admin_takedown_candidates(10)', '42501', null, 'visitors cannot list takedowns');
select throws_ok($$select public.admin_set_active('{}'::uuid[], false)$$, '42501', null, 'visitors cannot take clips down');
reset role;

select * from finish();
rollback;
