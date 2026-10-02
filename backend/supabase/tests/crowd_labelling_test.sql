-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(30);

-- four people: a, b, c finished their profile; d signed in but did not
insert into auth.users (id, email) values
  ('00000000-0000-0000-0000-00000000000a', 'a@example.test'),
  ('00000000-0000-0000-0000-00000000000b', 'b@example.test'),
  ('00000000-0000-0000-0000-00000000000c', 'c@example.test'),
  ('00000000-0000-0000-0000-00000000000d', 'd@example.test');
insert into public.profiles (id, username, adult_confirmed_at) values
  ('00000000-0000-0000-0000-00000000000a', 'alice', now()),
  ('00000000-0000-0000-0000-00000000000b', 'bob', now()),
  ('00000000-0000-0000-0000-00000000000c', 'cara', now());

-- two clips, each to be heard by two people; the unsure one first
set local role service_role;
select is(public.admin_add_clips('[
  {"audio_path": "t1.mp3", "source_uid": "pool:one", "kind": "unsure", "priority": 0.9, "target_votes": 2, "info": {"fused": {"tapping": 0.5}}},
  {"audio_path": "t2.mp3", "source_uid": "pool:two", "kind": "control", "priority": 0.1, "target_votes": 2, "info": {"people": ["moaning"]}}
]'::jsonb), 2, 'service role adds clips');
select is(public.admin_add_clips('[{"audio_path": "t1b.mp3", "source_uid": "pool:one", "kind": "unsure", "priority": 0.95}]'::jsonb),
          0, 're-adding a source refreshes it instead of duplicating');
reset role;

-- anonymous visitors: public stats only
set local role anon;
select throws_ok('select * from public.next_clip()', '28000', 'sign in first', 'anon cannot get clips');
select throws_ok('select count(*) from public.clips', '42501', null, 'anon cannot read clips');
select lives_ok('select * from public.leaderboard()', 'anon can read the leaderboard');
select lives_ok('select * from public.site_stats()', 'anon can read site stats');
select throws_ok('select adult_confirmed_at from public.profiles', '42501', null, 'private profile fields stay private');
reset role;

-- d has no profile yet
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000d", "role": "authenticated"}';
select throws_ok('select * from public.next_clip()', 'P0001', 'finish your profile first', 'profile needed before labelling');

-- alice gets the unsure clip, and the same one again on reload
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000a", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 't1.mp3', 'highest-priority clip first');
select is((select audio_path from public.next_clip()), 't1.mp3', 'an open assignment comes back on reload');
select throws_ok('select count(*) from public.clips', '42501', null, 'labellers cannot read the clip table');
select throws_ok('select * from private.clip_sources', '42501', null, 'labellers cannot read clip sources');
select throws_ok('select public.admin_export_labels()', '42501', null, 'labellers cannot export labels');
select ok(private.can_hear('t1.mp3'), 'storage lets alice hear her assigned clip');
select ok(not private.can_hear('t2.mp3'), '... and not a clip she was not given');
select throws_ok($$select * from public.submit_label((select clip_id from public.next_clip()), array['tapping'], null, false, 4000)$$,
                 '22023', 'listen to the clip first', 'answers within 2 s of assignment are refused');
reset role;
update public.assignments set assigned_at = now() - interval '10 seconds';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000a", "role": "authenticated"}';
select throws_ok($$select * from public.submit_label((select clip_id from public.next_clip()), array['tapping'], null, false, 500)$$,
                 '22023', 'listen to the clip first', 'answers without 1.5 s of playback are refused');
select throws_ok($$select * from public.submit_label((select clip_id from public.next_clip()), array['banjo'], null, false, 4000)$$,
                 '22023', 'unknown label', 'labels must come from the menu');
select throws_ok($$select * from public.submit_label((select clip_id from public.next_clip()), array[]::text[], null, false, 4000)$$,
                 '22023', null, 'an empty answer needs "can''t tell"');
select is((select total from public.submit_label((select clip_id from public.next_clip()),
                                                  array['tapping', 'whispering', 'tapping', 'something else'], '  wooden comb  ', false, 4000)),
          1::bigint, 'alice''s first label counts');
select is((select labels from public.labels), array['something else', 'tapping', 'whispering'], 'labels are de-duplicated and sorted');
select is((select other_text from public.labels), 'wooden comb', 'free text kept (trimmed) with "something else"');

-- bob is the second judge of clip 1; cara cannot get clip 1 (quorum of 2 reached)
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000b", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 't1.mp3', 'a clip in progress is served to the next judge first');
select is((select count(*) from public.labels), 0::bigint, 'bob cannot see alice''s labels');
reset role;
update public.assignments set assigned_at = now() - interval '10 seconds';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000b", "role": "authenticated"}';
select lives_ok($$select * from public.submit_label((select clip_id from public.next_clip()), array[]::text[], null, true, 3000)$$,
                'bob can say he cannot tell');
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000c", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 't2.mp3', 'cara gets the other clip once clip 1 has its two answers');
select lives_ok($$select public.skip_clip((select clip_id from public.next_clip()))$$, 'cara can pass on a clip');
select is((select count(*) from public.next_clip()), 0::bigint, '... and never gets it back; nothing else is left for her');
reset role;

-- contributions and export
select is((select array_agg(username order by rank, username) from public.leaderboard()), array['alice', 'bob'], 'leaderboard lists labellers');
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select is((select count(*) from public.admin_export_labels() where source_uid = 'pool:one'), 2::bigint, 'export carries the source uid');
reset role;

select * from finish();
rollback;
