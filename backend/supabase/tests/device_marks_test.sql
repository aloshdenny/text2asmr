-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(10);

-- two guests on one phone, a third person elsewhere on the same network
insert into auth.users (id, email, is_anonymous) values
  ('d1000000-0000-0000-0000-000000000000', null, true), ('d2000000-0000-0000-0000-000000000000', null, true),
  ('d3000000-0000-0000-0000-000000000000', 'd3@example.test', false);
update private.settings set value = 0 where key in ('repeat_rate', 'control_rate_max', 'control_rate_min');
update public.clips set active = false;
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select public.admin_add_clips('[{"audio_path": "dm1.mp3", "source_uid": "pool:dm-1", "kind": "unsure", "priority": 0.9}]'::jsonb);
reset role;

-- a guest starts on the phone: the proxy's marks come in as request headers
set local request.headers = '{"x-ab-device": "phoneAAAAAAAAAAAAAAAAA", "x-ab-ip": "ipAAAAAAAAAAAAAAAAAAAA", "x-ab-net": "netAAAAAAAAAAAAAAAAAAA", "x-ab-ua": "Chrome Android"}';
set local role authenticated;
set local request.jwt.claims = '{"sub": "d1000000-0000-0000-0000-000000000000", "role": "authenticated", "is_anonymous": true}';
select lives_ok('select public.start_guest()', 'a guest starts');
reset role;
select is((select array_agg(kind order by kind) from private.user_marks where user_id = 'd1000000-0000-0000-0000-000000000000'),
          array['device', 'ip', 'net', 'ua'], 'starting records the device, IP, network and browser marks');

-- a second guest on the same phone answers a clip
set local role authenticated;
set local request.jwt.claims = '{"sub": "d2000000-0000-0000-0000-000000000000", "role": "authenticated", "is_anonymous": true}';
select public.start_guest();
select clip_id from public.next_clip();
reset role;
update public.assignments set assigned_at = now() - interval '10 seconds';
set local role authenticated;
set local request.jwt.claims = '{"sub": "d2000000-0000-0000-0000-000000000000", "role": "authenticated", "is_anonymous": true}';
select public.submit_label((select clip_id from public.next_clip()), array['tapping'], null, false, 7000);
reset role;
select is((select hits from private.user_marks where user_id = 'd2000000-0000-0000-0000-000000000000' and kind = 'device'), 2,
          'each answer counts another sighting');

-- the first guest answers too, so both accounts have labels
set local role authenticated;
set local request.jwt.claims = '{"sub": "d1000000-0000-0000-0000-000000000000", "role": "authenticated", "is_anonymous": true}';
select clip_id from public.next_clip();
reset role;
update public.assignments set assigned_at = now() - interval '10 seconds';
set local role authenticated;
set local request.jwt.claims = '{"sub": "d1000000-0000-0000-0000-000000000000", "role": "authenticated", "is_anonymous": true}';
select public.submit_label((select clip_id from public.next_clip()), array['crinkling'], null, false, 7000);
reset role;

-- a member on another device in the same network; junk headers are ignored
set local request.headers = '{"x-ab-device": "laptopBBBBBBBBBBBBBBBB", "x-ab-ip": "ipBBBBBBBBBBBBBBBBBBBB", "x-ab-net": "netAAAAAAAAAAAAAAAAAAA", "x-ab-ua": "Safari <script>"}';
set local request.jwt.claims = '{"sub": "d3000000-0000-0000-0000-000000000000", "role": "authenticated"}';
insert into public.profiles (id, username, adult_confirmed_at) values ('d3000000-0000-0000-0000-000000000000', 'dm_member', now());
select is((select count(*) from private.user_marks where user_id = 'd3000000-0000-0000-0000-000000000000' and kind = 'ua'), 0::bigint,
          'a malformed mark is not stored');

-- an import by the service role is not marked against the person it is for
set local request.jwt.claims = '{"role": "service_role"}';
insert into public.labels (user_id, clip_id, labels, listen_ms)
  values ('d3000000-0000-0000-0000-000000000000', (select clip_id from private.clip_sources where source_uid = 'pool:dm-1'), array['liquid'], 6000);
select is((select max(hits) from private.user_marks where user_id = 'd3000000-0000-0000-0000-000000000000'), 1,
          'labels written by someone else leave no mark');
reset request.jwt.claims;

-- the groups
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select is((select accounts from public.admin_shared_marks() where kind = 'device' and mark = 'phoneAAA'), 2, 'the phone is shared by two accounts');
select is((select accounts from public.admin_shared_marks() where kind = 'net' and mark = 'netAAAAA'), 3, 'the network by three');
select is((select count(*) from public.admin_shared_marks() where kind = 'device' and mark = 'laptopBB'), 0::bigint, 'a device of one is not a group');
reset role;

select private.refresh_quality();
select ok('shares a device with other accounts' = any ((select flags from private.labeller_quality where user_id = 'd1000000-0000-0000-0000-000000000000')::text[]),
          'sharing a device is a warning sign');
select ok(not ('shares a device with other accounts' = any ((select flags from private.labeller_quality where user_id = 'd3000000-0000-0000-0000-000000000000')::text[])),
          'sharing only a network is not');

select * from finish();
rollback;
