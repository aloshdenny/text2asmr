-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(9);

update private.settings set value = 0 where key in ('repeat_rate', 'control_rate_max', 'control_rate_min');
update public.clips set active = false;
insert into auth.users (id, email) values ('c1000000-0000-0000-0000-000000000000', 'fl@example.test');
insert into public.profiles (id, username, adult_confirmed_at) values ('c1000000-0000-0000-0000-000000000000', 'fl_one', now());
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select public.admin_add_clips('[
  {"audio_path": "fl-a.mp3", "source_uid": "pool:fl-a", "kind": "unsure", "priority": 0.9, "duration_s": 8},
  {"audio_path": "fl-b.mp3", "source_uid": "pool:fl-b", "kind": "unsure", "priority": 0.8, "duration_s": 8},
  {"audio_path": "fl-c.mp3", "source_uid": "pool:fl-c", "kind": "unsure", "priority": 0.7, "duration_s": 8}
]'::jsonb);
reset role;

-- the whole clip: an 8 s clip needs ~8 s of playback and ~8 s since it was served
set local request.headers = '{"x-ab-device": "deviceAAAAAAAAAAAAAAAA"}';
set local role authenticated;
set local request.jwt.claims = '{"sub": "c1000000-0000-0000-0000-000000000000", "role": "authenticated"}';
select public.next_clip();
reset role;
update public.assignments set assigned_at = now() - interval '30 seconds' where user_id = 'c1000000-0000-0000-0000-000000000000';
set local role authenticated;
set local request.jwt.claims = '{"sub": "c1000000-0000-0000-0000-000000000000", "role": "authenticated"}';
select throws_ok($$select * from public.submit_label((select clip_id from public.next_clip()), array['tapping'], null, false, 4000)$$,
                 '22023', 'listen to the whole clip first', 'half a clip is not enough');
select lives_ok($$select * from public.submit_label((select clip_id from public.next_clip()), array['tapping'], null, false, 7600)$$,
                'the whole clip (within encoder padding) is');
-- the next clip was served just now (inside this transaction): no time has passed, whatever the player reports
select throws_ok($$select * from public.submit_label((select clip_id from public.next_clip()), array['tapping'], null, false, 8000)$$,
                 '22023', 'listen to the whole clip first', 'a faked play time is refused until the clip could have played');
reset role;

-- devices: the first two hold the slots, a third is turned away
set local role authenticated;
set local request.jwt.claims = '{"sub": "c1000000-0000-0000-0000-000000000000", "role": "authenticated"}';
select lives_ok('select public.claim_device()', 'the first device is fine again');
set local request.headers = '{"x-ab-device": "deviceBBBBBBBBBBBBBBBB"}';
select lives_ok('select public.claim_device()', 'a second device is fine');
set local request.headers = '{"x-ab-device": "deviceCCCCCCCCCCCCCCCC"}';
select throws_ok('select public.claim_device()', 'P0004', 'this account is already in use on two other devices', 'a third is refused');
select throws_ok($$select * from public.submit_label((select clip_id from public.next_clip()), array['tapping'], null, false, 8000)$$,
                 'P0004', null, '... and cannot answer');
set local request.headers = '{"x-ab-ua": "Chrome Android"}';
select throws_ok('select public.claim_device()', 'P0003', null, 'no device id, no answers');
reset role;

-- a slot unseen for 30 days frees up
update private.account_devices set last_seen = now() - interval '31 days'
where user_id = 'c1000000-0000-0000-0000-000000000000' and device = 'deviceBBBBBBBBBBBBBBBB';
set local request.headers = '{"x-ab-device": "deviceCCCCCCCCCCCCCCCC"}';
set local role authenticated;
set local request.jwt.claims = '{"sub": "c1000000-0000-0000-0000-000000000000", "role": "authenticated"}';
select lives_ok('select public.claim_device()', 'a device unused for 30 days gives its slot up');
reset role;

select * from finish();
rollback;
