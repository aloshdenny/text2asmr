-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(13);

-- six people answer the same 20 clips. Each clip holds one of five sound sets; three listen carefully, one answers
-- with another clip's sounds (a random clicker with realistic-looking answers), one ticks whispering every time, and
-- one answers carefully but 2.5 s after the clip was served.
insert into auth.users (id, email) values
  ('00000000-0000-0000-0000-0000000000a1', 'qa1@example.test'), ('00000000-0000-0000-0000-0000000000a2', 'qa2@example.test'),
  ('00000000-0000-0000-0000-0000000000a3', 'qa3@example.test'), ('00000000-0000-0000-0000-0000000000a4', 'qa4@example.test'),
  ('00000000-0000-0000-0000-0000000000a5', 'qa5@example.test'), ('00000000-0000-0000-0000-0000000000a6', 'qa6@example.test');
insert into public.profiles (id, username, adult_confirmed_at) values
  ('00000000-0000-0000-0000-0000000000a1', 'qa_careful1', now()), ('00000000-0000-0000-0000-0000000000a2', 'qa_careful2', now()),
  ('00000000-0000-0000-0000-0000000000a3', 'qa_careful3', now()), ('00000000-0000-0000-0000-0000000000a4', 'qa_random', now()),
  ('00000000-0000-0000-0000-0000000000a5', 'qa_whisper', now()), ('00000000-0000-0000-0000-0000000000a6', 'qa_rusher', now());

set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select public.admin_add_clips((select jsonb_agg(jsonb_build_object('audio_path', 'q' || i || '.mp3', 'source_uid', 'qa:' || i,
                                                                   'kind', 'unsure', 'priority', 0.5))
                               from generate_series(1, 20) i));
reset role;

create temp table truth (i int primary key, clip uuid, labels text[]);
insert into truth
  select i, (select clip_id from private.clip_sources where source_uid = 'qa:' || i),
         (array[array['whispering', 'whispering'], array['tapping', 'tapping'], array['crinkling', 'tapping'],
                array['breathing', 'whispering'], array['brushing', 'brushing']])[1 + i % 5 : 1 + i % 5][1:2]
  from generate_series(1, 20) i;
update truth set labels = array(select distinct x from unnest(labels) x order by x);

create temp table answers (who uuid, i int, labels text[], answer_s numeric);
insert into answers
  select u.id, t.i, t.labels, 9 from truth t,
         (values ('00000000-0000-0000-0000-0000000000a1'::uuid), ('00000000-0000-0000-0000-0000000000a2'),
                 ('00000000-0000-0000-0000-0000000000a3')) u(id)
  union all select '00000000-0000-0000-0000-0000000000a4', t.i, (select labels from truth o where o.i = 1 + (t.i + 6) % 20), 9 from truth t
  union all select '00000000-0000-0000-0000-0000000000a5', t.i, array['whispering'], 9 from truth t
  union all select '00000000-0000-0000-0000-0000000000a6', t.i, t.labels, 2.5 from truth t;

insert into public.assignments (user_id, clip_id, assigned_at, state, attempt)
  select a.who, t.clip, now() - interval '1 hour' + a.i * interval '1 minute', 'done', 1 from answers a join truth t on t.i = a.i;
insert into public.labels (user_id, clip_id, labels, listen_ms, created_at, attempt, seq)
  select a.who, t.clip, a.labels, 6000, now() - interval '1 hour' + a.i * interval '1 minute' + a.answer_s * interval '1 second', 1, a.i
  from answers a join truth t on t.i = a.i;

select lives_ok('select private.refresh_quality()', 'quality refreshes');
-- a second pass leaves the weak out of everyone else's references
select lives_ok('select private.refresh_quality()', 'and refreshes again');

select is((select status from private.labeller_quality where user_id = '00000000-0000-0000-0000-0000000000a1'), 'ok',
          'a careful listener is ok');
select ok((select follow from private.labeller_quality where user_id = '00000000-0000-0000-0000-0000000000a1') > 0.8,
          '... and their answers follow the audio');
select is((select status from private.labeller_quality where user_id = '00000000-0000-0000-0000-0000000000a4'), 'weak',
          'answers that belong to other clips are weak');
select ok((select follow from private.labeller_quality where user_id = '00000000-0000-0000-0000-0000000000a4') < 0.1,
          '... because they do not follow the audio');
select is((select status from private.labeller_quality where user_id = '00000000-0000-0000-0000-0000000000a5'), 'weak',
          'whispering on every clip is weak, though a fifth of the clips are whispers');
select ok((select 'same label every time' = any (flags) from private.labeller_quality where user_id = '00000000-0000-0000-0000-0000000000a5'),
          '... and flagged for the same label every time');
select is((select status from private.labeller_quality where user_id = '00000000-0000-0000-0000-0000000000a6'), 'watch',
          'careful answers given too fast are on watch');
select is((select flags from private.labeller_quality where user_id = '00000000-0000-0000-0000-0000000000a6'), array['rushing'],
          '... for rushing only');

-- the weak stop counting
update private.settings set value = 1 where key = 'hide_weak_from_board';
insert into private.settings (key, value) select 'hide_weak_from_board', 1 where not exists (select 1 from private.settings where key = 'hide_weak_from_board');
select is((select count(*) from public.leaderboard(null, 200) where username in ('qa_random', 'qa_whisper')), 0::bigint,
          'with hide_weak_from_board on, weak labellers leave the board');
select is((select count(*) from public.leaderboard(null, 200) where username = 'qa_careful1'), 1::bigint, '... and the rest stay');

set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select is((select status from public.admin_labeller_quality() where username = 'qa_random'), 'weak', 'the review list shows it');
reset role;

select * from finish();
rollback;
