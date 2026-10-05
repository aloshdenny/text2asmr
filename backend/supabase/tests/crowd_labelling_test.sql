-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(64);

-- serving draws (hidden retests, controls) are random: off for the deterministic part, forced on at the end
update private.settings set value = 0 where key in ('repeat_rate', 'control_rate_max', 'control_rate_min');

-- everyone already on the board sits out too (rolled back), so the board holds only the test's people
update public.profiles set deleted_at = now() where deleted_at is null;

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

-- whatever is already queued sits out (this transaction is rolled back), so only the test's clips are served
update public.clips set active = false;

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
select is((select audio_path from public.next_clip()), 't1.mp3', 'never dry: past its quorum, the unsure clip still takes more listeners');
reset role;
select ok((select skips from public.clips where audio_path = 't2.mp3') = 1, 'a skip is counted on the clip');
select ok((select need from public.clips where audio_path = 't2.mp3') < 0.3, '... and sinks it for everyone');
select ok((select need from public.clips where audio_path = 't1.mp3') between 0.3 and 0.6, 'quorum reached: need falls to its uncertainty and disagreement');
update public.assignments set assigned_at = now() - interval '10 seconds';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000c", "role": "authenticated"}';
select lives_ok($$select * from public.submit_label((select clip_id from public.next_clip()), array['tapping', 'whispering'], null, false, 3000)$$,
                'a third judge answers');
reset role;
select is((select round(agreement::numeric, 2) from public.clips where audio_path = 't1.mp3'), 0.67, 'agreement is the mean Jaccard of the answers');
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000c", "role": "authenticated"}';
select is((select count(*) from public.next_clip()), 0::bigint, 'a listener runs out only after hearing every clip');
reset role;

-- contributions and export
select is((select array_agg(username order by rank, username) from public.leaderboard() where username in ('alice', 'bob', 'kitty')), array['alice', 'bob'], 'leaderboard lists labellers');
select is((select max(rank) - min(rank) from public.leaderboard(null, 200) where username in ('alice', 'bob')), 1::bigint,
          'tied labellers get consecutive ranks, not a shared one');
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select is((select count(*) from public.admin_export_labels() where source_uid = 'pool:one'), 3::bigint, 'export carries the source uid');
select is((select labeller from public.admin_export_labels() where username = 'alice'), 'crowd:00000000', 'site listeners export as crowd:<id>');
reset role;
set local request.jwt.claims = '';

-- a kit labeller from before the site: a profile waiting for whoever signs up with their email
select ok(private.import_contributor('Kit@Example.test', 'kitty', 'Kit', 'test_kit_labeller') is not null, 'import makes a claimable profile');
select is(private.import_contributor('kit@example.test', 'kitty', 'Kit', 'test_kit_labeller'),
          (select id from public.profiles where username = 'kitty'), 'importing again reuses it');
select ok(not exists (select 1 from auth.users where email = 'kit@example.test'), 'no sign-in is made for them');
insert into public.imported_labels (user_id, source_uid, labels, kit, created_at)
select (select id from public.profiles where username = 'kitty'), 'pool:old' || g, array['tapping'], 'kit-2', now() - interval '3 days'
from generate_series(1, 3) g;
select is((select array_agg(username order by rank, username) from public.leaderboard() where username in ('alice', 'bob', 'kitty')), array['kitty', 'alice', 'bob'],
          'imported kit labels count on the leaderboard');
set local role anon;
select is(public.account_status(' KIT@example.test '), 'unclaimed', 'sign-in tells a kit labeller to sign up');
select is(public.account_status('a@example.test'), 'ok', 'ordinary accounts are not reported');
reset role;
-- signing up with the address, unconfirmed: nothing moves yet; once confirmed, the profile and its labels move over
insert into auth.users (id, email) values ('00000000-0000-0000-0000-0000000000e1', 'kit@example.test');
select ok(exists (select 1 from public.profiles where username = 'kitty' and claim_email is not null), 'an unconfirmed sign-up claims nothing');
update auth.users set email_confirmed_at = now() where id = '00000000-0000-0000-0000-0000000000e1';
select is((select id from public.profiles where username = 'kitty'), '00000000-0000-0000-0000-0000000000e1'::uuid, 'confirmed: the profile is theirs');
select is((select count(*) from public.imported_labels where user_id = '00000000-0000-0000-0000-0000000000e1'), 3::bigint, 'with its kit labels');
select is(public.account_status('kit@example.test'), 'ok', 'and the address is no longer reported');
select is((select sum(labelled) from public.contributions('kitty')), 3::numeric, 'the contribution graph follows');

-- alice deletes her account: off the lists, her label stays (and still exports)
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000a", "role": "authenticated"}';
select lives_ok('select public.delete_account()', 'alice deletes her account');
reset role;
select ok(not exists (select 1 from auth.users where id = '00000000-0000-0000-0000-00000000000a'), 'her sign-in is gone');
select is((select array_agg(username order by rank, username) from public.leaderboard() where username in ('alice', 'bob', 'kitty')), array['kitty', 'bob'], 'she is off the leaderboard');
select is((select count(*) from public.labels where user_id = '00000000-0000-0000-0000-00000000000a'), 1::bigint, 'her label stays');

-- a third judge on clip 1 settles part of it: tapping and whispering agreed, "something else" split
select is((select round(crowd_conf::numeric, 2) from public.clips where audio_path = 't1.mp3'), 0.67, 'trust-weighted consensus on the clip');
select is((select need < 0.5 from public.clips where audio_path = 't1.mp3'), true, 'quorum reached and partly settled: need drops');

-- controls go first to people we know little about
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select is(public.admin_add_clips('[{"audio_path": "t3.mp3", "source_uid": "pool:three", "kind": "unsure", "uncertainty": 1.0}]'::jsonb), 1, 'a new clip');
reset role;
set local request.jwt.claims = '';
select is((select target_votes::int || '/' || max_votes::int from public.clips where audio_path = 't3.mp3'), '12/20', 'new clips want 12 to 20 judges');
update private.settings set value = 1 where key in ('control_rate_max', 'control_rate_min');
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000b", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 't2.mp3', 'a control is served ahead of the neediest clip');
reset role;
update private.settings set value = 0 where key in ('control_rate_max', 'control_rate_min');

-- hidden retests: after the gap, a clip comes back as a second attempt that does not count toward the quorum
update private.settings set value = 1 where key = 'repeat_rate';
update private.settings set value = 0 where key = 'repeat_gap';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000c", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 't1.mp3', 'a clip comes back as a hidden retest');
reset role;
update public.assignments set assigned_at = now() - interval '10 seconds';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000c", "role": "authenticated"}';
select lives_ok($$select * from public.submit_label((select clip_id from public.next_clip()), array['tapping'], null, false, 3000)$$, 'the retest is answered');
reset role;
select is((select votes from public.clips where audio_path = 't1.mp3'), 3::smallint, 'a retest does not count toward the quorum');
select is((select count(*) from public.labels l join public.profiles p on p.id = l.user_id where p.username = 'cara' and l.attempt = 2), 1::bigint,
          'it is kept as a second attempt');
select lives_ok('select private.refresh_trust()', 'trust refresh runs');
select ok((select trust between 0 and 1 and trust_evidence > 0 from public.profiles where username = 'cara'), 'cara has a trust score from her retest');

select * from finish();
rollback;
