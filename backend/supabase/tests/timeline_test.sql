-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(28);

update private.settings set value = 0 where key in ('repeat_rate', 'control_rate_max', 'control_rate_min', 'male_share');
update public.clips set active = false, gold = false;
insert into auth.users (id, email) values
  ('a1100000-0000-0000-0000-000000000000', 'tl1@example.test'), ('a2200000-0000-0000-0000-000000000000', 'tl2@example.test'),
  ('a3300000-0000-0000-0000-000000000000', 'tl3@example.test'), ('a4400000-0000-0000-0000-000000000000', 'tl4@example.test');
insert into public.profiles (id, username, adult_confirmed_at) values
  ('a1100000-0000-0000-0000-000000000000', 'tl_women', now()), ('a2200000-0000-0000-0000-000000000000', 'tl_men', now()),
  ('a3300000-0000-0000-0000-000000000000', 'tl_both', now()), ('a4400000-0000-0000-0000-000000000000', 'tl_spans', now());

-- preferences
set local role authenticated;
set local request.jwt.claims = '{"sub": "a1100000-0000-0000-0000-000000000000", "role": "authenticated"}';
select throws_ok($$select public.set_preferences(p_voice => 'robots')$$, '22023', null, 'voices are female, male or both');
select lives_ok($$select public.set_preferences('speech', 'nsfw', 'female')$$, 'a labeller answers the three questions');
select is((select voice_pref from public.labelling_state()), 'female', '... and reads it back');
select throws_ok($$select voice_pref from public.profiles$$, '42501', null, 'nobody can read preferences from profiles');
set local request.jwt.claims = '{"sub": "a2200000-0000-0000-0000-000000000000", "role": "authenticated"}';
select public.set_preferences('speech', 'both', 'male');
set local request.jwt.claims = '{"sub": "a3300000-0000-0000-0000-000000000000", "role": "authenticated"}';
select public.set_preferences('both', 'both', 'both');
reset role;

-- a needy male-voice clip and a less needy female-voice one
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select public.admin_add_clips('[
  {"audio_path": "tm.mp3", "source_uid": "pool:tl-male", "kind": "unsure", "priority": 0.9, "info": {"voice": "male"}},
  {"audio_path": "tf.mp3", "source_uid": "pool:tl-female", "kind": "unsure", "priority": 0.2, "info": {"voice": "female"}}
]'::jsonb);
reset role;
set local role authenticated;
set local request.jwt.claims = '{"sub": "a1100000-0000-0000-0000-000000000000", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 'tf.mp3', 'women only: the male clip is skipped though needier');
set local request.jwt.claims = '{"sub": "a2200000-0000-0000-0000-000000000000", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 'tm.mp3', 'men only: the male clip');
reset role;
update public.clips c set active = false from private.clip_sources s where s.clip_id = c.id and s.source_uid = 'pool:tl-male';
delete from public.assignments where user_id = 'a2200000-0000-0000-0000-000000000000';
set local role authenticated;
set local request.jwt.claims = '{"sub": "a2200000-0000-0000-0000-000000000000", "role": "authenticated"}';
select is((select count(*) from public.next_clip()), 0::bigint, 'men only never gets a female-voice clip');
-- SFW only: never an adult clip, even the neediest; men only after asking for NSFW
set local request.jwt.claims = '{"sub": "a1100000-0000-0000-0000-000000000000", "role": "authenticated"}';
select public.set_preferences(p_content => 'sfw');
reset role;
delete from public.assignments where user_id = 'a1100000-0000-0000-0000-000000000000';
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select public.admin_add_clips('[{"audio_path": "tt.mp3", "source_uid": "yt:tl-trigger:1", "kind": "unsure", "priority": 0.05}]'::jsonb);
reset role;
update public.clips c set active = true from private.clip_sources s where s.clip_id = c.id and s.source_uid = 'pool:tl-male';
set local role authenticated;
set local request.jwt.claims = '{"sub": "a1100000-0000-0000-0000-000000000000", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 'tt.mp3', 'SFW only: the trigger recording, not the needier voice clips');
select results_eq('select sound_pref, content_pref, voice_pref from public.labelling_state()', $$values ('speech', 'sfw', 'female')$$,
                  'all three preferences read back');
reset role;
select ok((select nsfw from public.clips c join private.clip_sources s on s.clip_id = c.id where s.source_uid = 'pool:tl-male'),
          'voice-corpus clips count as NSFW');
update public.clips c set active = false from private.clip_sources s where s.clip_id = c.id and s.source_uid in ('pool:tl-male', 'yt:tl-trigger:1');

-- the gold set: picked per label, served first, and it opens the timeline labeller when complete
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select public.admin_add_clips('[
  {"audio_path": "g1.mp3", "source_uid": "pool:tl-gold-1", "kind": "unsure", "priority": 0.01, "info": {"fused": {"tapping": 0.8}}},
  {"audio_path": "g2.mp3", "source_uid": "pool:tl-gold-2", "kind": "unsure", "priority": 0.01, "info": {"title_tags": ["squirting"]}}
]'::jsonb);
select is((select sum(picked) from public.admin_pick_gold(1)), 2::bigint, 'one clip per label where the queue has evidence');
reset role;
select is((select array_agg(s.info ->> 'gold_for' order by s.source_uid) from private.clip_sources s join public.clips c on c.id = s.clip_id where c.gold),
          array['tapping', 'squirting'], 'each gold clip remembers its label');
select is((select max(target_votes) from public.clips where gold), 20::smallint, 'gold clips collect up to 20 answers');
set local role authenticated;
set local request.jwt.claims = '{"sub": "a3300000-0000-0000-0000-000000000000", "role": "authenticated"}';
select ok((select audio_path from public.next_clip()) in ('g1.mp3', 'g2.mp3'), 'gold clips come before needier ones');
select results_eq('select mode, gold_done, gold_total from public.labelling_state()', $$values ('tags', 0, 2)$$,
                  'the tag list stays until the gold set is complete');
select public.set_preferences(p_beta => true);
select is((select mode from public.labelling_state()), 'timeline', 'a labeller can opt into the timeline beta');
select public.set_preferences(p_beta => false);
reset role;
update public.clips set votes = 12 where gold;
set local role authenticated;
set local request.jwt.claims = '{"sub": "a3300000-0000-0000-0000-000000000000", "role": "authenticated"}';
select is((select mode from public.labelling_state()), 'timeline', 'with 12 answers on every gold clip, everyone gets the timeline');
reset role;
update private.settings set value = -1 where key = 'timeline_ui';
set local role authenticated;
set local request.jwt.claims = '{"sub": "a3300000-0000-0000-0000-000000000000", "role": "authenticated"}';
select is((select mode from public.labelling_state()), 'tags', 'timeline_ui -1 switches it off for everyone');
reset role;
update private.settings set value = 0 where key = 'timeline_ui';

-- timeline answers
update public.clips set active = false;
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select public.admin_add_clips('[{"audio_path": "sp.mp3", "source_uid": "pool:tl-spans", "kind": "unsure", "priority": 0.9, "duration_s": 6}]'::jsonb);
reset role;
set local role authenticated;
set local request.jwt.claims = '{"sub": "a4400000-0000-0000-0000-000000000000", "role": "authenticated"}';
select audio_path from public.next_clip();
reset role;
update public.assignments set assigned_at = now() - interval '10 seconds';
set local role authenticated;
set local request.jwt.claims = '{"sub": "a4400000-0000-0000-0000-000000000000", "role": "authenticated"}';
select throws_ok($$select * from public.submit_spans((select clip_id from public.next_clip()),
  '[{"label": "moaning", "start_ms": 100, "end_ms": 9000, "lane": 0}]'::jsonb, null, 4000)$$, '22023', null, 'a sound must end inside the clip');
select throws_ok($$select * from public.submit_spans((select clip_id from public.next_clip()),
  '[{"label": "moaning", "start_ms": 0, "end_ms": 900, "lane": 0}, {"label": "tapping", "start_ms": 0, "end_ms": 900, "lane": 0}]'::jsonb, null, 4000)$$,
  '22023', null, 'one sound per lane');
select throws_ok($$select * from public.submit_spans((select clip_id from public.next_clip()),
  (select jsonb_agg(jsonb_build_object('label', 'moaning', 'start_ms', 0, 'end_ms', 500, 'lane', i)) from generate_series(0, 6) i), null, 4000)$$,
  '22023', 'up to 6 sounds per clip', 'up to 6 sounds');
select throws_ok($$select * from public.submit_spans((select clip_id from public.next_clip()), '[]'::jsonb, '  ', 4000)$$,
  '22023', null, 'an empty answer is refused');
select is((select total from public.submit_spans((select clip_id from public.next_clip()),
  '[{"label": "normal speech", "start_ms": 0, "end_ms": 2500, "lane": 3}, {"label": "sex slapping", "start_ms": 1200, "end_ms": 5000, "lane": 4},
    {"label": "moaning", "start_ms": 2900, "end_ms": 6000, "lane": 0}]'::jsonb, 'a creaky chair', 4000)), 1::bigint, 'a timeline answer is saved');
select is((select labels from public.labels where user_id = 'a4400000-0000-0000-0000-000000000000'),
          array['moaning', 'normal speech', 'sex slapping', 'something else'], 'its labels are the sounds marked, plus the free text');
select is((select jsonb_array_length(spans) from public.labels where user_id = 'a4400000-0000-0000-0000-000000000000'), 3,
          'the spans are kept with their times');
select ok(private.can_hear('sp.mp3'), 'the clip can be heard again after answering');
select lives_ok($$select public.edit_answer((select clip_id from public.my_recent_answers(5) limit 1),
  '[{"label": "moaning", "start_ms": 2950, "end_ms": 6000, "lane": 0}]'::jsonb, null)$$, 'a recent answer can be changed');
select is((select labels from public.labels where user_id = 'a4400000-0000-0000-0000-000000000000'), array['moaning'], '... and is');
reset role;

select * from finish();
rollback;
