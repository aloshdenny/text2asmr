-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(38);

insert into auth.users (id, email) values
  ('00000000-0000-0000-0000-0000000000e1', 'e1@example.test'),
  ('00000000-0000-0000-0000-0000000000e2', 'e2@example.test');
insert into public.profiles (id, username, adult_confirmed_at) values
  ('00000000-0000-0000-0000-0000000000e1', 'erin', now()),
  ('00000000-0000-0000-0000-0000000000e2', 'finn', now());

-- scratch tables the labeller's calls write into (temp tables die with the session)
create temp table served (audio_a text);
create temp table answer (correct boolean, was_ai boolean, fooled_pct integer, others integer, my_correct integer, my_total integer);
create temp table pair (kind text, audio_a text, audio_b text);
grant all on served, answer, pair to authenticated;

-- whatever is already live sits out (this transaction is rolled back)
update public.clips set active = false;
update private.rai_items set active = false;

-- helpers
select ok((select bool_and(g between 8 and 17) and min(g) < 11 and max(g) > 14
           from (select private.interlude_gap() g from generate_series(1, 400)) s), 'gaps are 8-17 labels, spread out');
select is(private.hot_sound('{"fused": {"moaning": 0.6, "whispering": 0.93, "tapping": 1.0}}'), 'whispering',
          'a clip''s main sound is its most likely vocal one');
select is(private.hot_sound('{"fused": {"tapping": 1.0}}'), null, 'no vocal sound, no pairing');

-- two Real-or-AI items (one real, one AI); no confident clips yet, so only Real or AI can come up
set local role service_role;
select is(public.admin_add_rai_items('[
  {"audio_path": "r-real.mp3", "is_ai": false, "voice": "ex01", "source": "ex01_whisper_001"},
  {"audio_path": "r-ai.mp3", "is_ai": true, "voice": "ex01", "model": "t2a-v1.2", "source": "ex01_whisper_002"}
]'::jsonb), 2, 'service role adds Real-or-AI items');
reset role;

-- anonymous visitors get nothing
set local role anon;
select throws_ok('select * from public.next_interlude()', '28000', 'sign in first', 'anon cannot get an interlude');
select throws_ok('select * from private.rai_items', '42501', null, 'anon cannot read the items');
reset role;

-- erin's first call only schedules her first interlude
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000e1", "role": "authenticated"}';
select is((select count(*) from public.next_interlude()), 0::bigint, 'no interlude on the first call');
select throws_ok('select * from private.rai_items', '42501', null, 'labellers cannot read the items');
select throws_ok('select * from private.rai_answers', '42501', null, 'labellers cannot read the answers');
select throws_ok('select * from private.hot_pairs', '42501', null, 'labellers cannot read the pairs');
select throws_ok('select interlude_next_seq from public.profiles', '42501', null, 'labellers cannot see when the next one is due');
select throws_ok('select * from public.admin_export_rai()', '42501', null, 'labellers cannot export answers');
select throws_ok($$select public.admin_add_rai_items('[]')$$, '42501', null, 'labellers cannot add items');
reset role;
select ok((select interlude_next_seq between label_seq + 8 and label_seq + 17 from public.profiles where username = 'erin'),
          'her first interlude is due 8-17 labels from now');

-- not due yet: nothing; due: a Real-or-AI clip
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000e1", "role": "authenticated"}';
select is((select count(*) from public.next_interlude()), 0::bigint, 'nothing before it is due');
reset role;
update public.profiles set label_seq = interlude_next_seq where username = 'erin';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000e1", "role": "authenticated"}';
select is((select kind from public.next_interlude()), 'rai', 'a due interlude comes up (only Real or AI is available)');
select ok((select audio_a in ('r-real.mp3', 'r-ai.mp3') and audio_b is null from public.next_interlude()),
          'one clip, the same one again on reload');
select ok(private.can_hear((select audio_a from public.next_interlude())), 'storage lets her hear it');
select throws_ok('select * from public.answer_rai(true, 4000)', '22023', 'listen to the clip first', 'too-quick guesses are refused');
reset role;
update private.interludes set served_at = now() - interval '10 seconds';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000e2", "role": "authenticated"}';
select ok(not private.can_hear('r-real.mp3') and not private.can_hear('r-ai.mp3'), 'nobody else can hear her interlude');
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000e1", "role": "authenticated"}';
insert into served select audio_a from public.next_interlude();
insert into answer select * from public.answer_rai(true, 4000);
select is((select correct from answer), (select audio_a = 'r-ai.mp3' from served), 'the guess is scored against the truth');
select is((select was_ai from answer), (select audio_a = 'r-ai.mp3' from served), 'the truth is revealed after the guess');
select ok((select fooled_pct is null and others = 0 and my_total = 1 from answer), 'no "fooled" share before enough answers; score 1 answered');
select is((select count(*) from public.next_interlude()), 0::bigint, 'answered: the next one waits for another gap');
select ok(not private.can_hear((select audio_a from served)), 'and the clip is no longer playable');
select throws_ok('select * from public.answer_rai(false, 4000)', 'P0001', 'nothing to answer', 'one answer per interlude');
reset role;

-- Which is hotter: Real or AI runs out, two confident moaning clips (and a tapping one that must not be paired)
update private.rai_items set active = false;
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select is(public.admin_add_clips('[
  {"audio_path": "h1.mp3", "source_uid": "pool:h1", "kind": "confident", "info": {"fused": {"moaning": 0.97, "breathing": 0.4}}},
  {"audio_path": "h2.mp3", "source_uid": "pool:h2", "kind": "confident", "info": {"fused": {"moaning": 0.95}}},
  {"audio_path": "h3.mp3", "source_uid": "pool:h3", "kind": "confident", "info": {"fused": {"tapping": 0.99}}},
  {"audio_path": "h4.mp3", "source_uid": "pool:h4", "kind": "unsure", "info": {"fused": {"moaning": 0.99}}}
]'::jsonb), 4, 'confident clips added (and an unsure one that must not be paired)');
reset role;
update public.profiles set label_seq = interlude_next_seq where username = 'erin';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000e1", "role": "authenticated"}';
insert into pair select * from public.next_interlude();
select is((select kind from pair), 'hot', 'Which is hotter comes up');
select ok((select array[audio_a, audio_b] <@ array['h1.mp3', 'h2.mp3'] and audio_a <> audio_b from pair),
          'two different clips sharing their main vocal sound');
select ok(private.can_hear('h1.mp3') and private.can_hear('h2.mp3') and not private.can_hear('h3.mp3') and not private.can_hear('h4.mp3'),
          'storage serves just the pair');
select throws_ok($$select * from public.answer_hot('c', 2000, 2000)$$, '22023', 'pick a or b', 'only a or b');
select throws_ok($$select * from public.answer_hot('a', 2000, 2000)$$, '22023', 'listen to both clips first', 'too-quick picks are refused');
reset role;
update private.interludes set served_at = now() - interval '10 seconds';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000e1", "role": "authenticated"}';
select ok((select agree_pct is null and judges = 0 from public.answer_hot('a', 2000, 2000)), 'first judge of the pair');
reset role;
select is((select c.audio_path from private.hot_votes v join public.clips c on c.id = v.winner
           where v.user_id = '00000000-0000-0000-0000-0000000000e1'), (select audio_a from pair),
          'the vote goes to the clip shown as A');

-- the only possible pair is judged: no Which-is-hotter for her any more (and nothing else is left), so nothing
update public.profiles set label_seq = interlude_next_seq where username = 'erin';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000e1", "role": "authenticated"}';
select is((select count(*) from public.next_interlude()), 0::bigint, 'a pair is never judged twice by the same person');
reset role;

-- skipping: finn gets the pair, skips it, and the next waits for another gap
update public.profiles set label_seq = 20, interlude_next_seq = 20 where username = 'finn';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000e2", "role": "authenticated"}';
select is((select kind from public.next_interlude()), 'hot', 'another person gets the pair');
select lives_ok('select public.skip_interlude()', 'an interlude can be skipped');
select is((select count(*) from public.next_interlude()), 0::bigint, 'skipped: nothing until the next gap');
reset role;

select * from finish();
rollback;
