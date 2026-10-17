-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(11);

update private.settings set value = 0 where key in ('repeat_rate', 'control_rate_max', 'control_rate_min');
update public.clips set active = false;
insert into auth.users (id, email) values
  ('00000000-0000-0000-0000-0000000000f1', 'f1@example.test'),
  ('00000000-0000-0000-0000-0000000000f2', 'f2@example.test');
insert into public.profiles (id, username, adult_confirmed_at) values
  ('00000000-0000-0000-0000-0000000000f1', 'gale', now()),
  ('00000000-0000-0000-0000-0000000000f2', 'hugo', now());

-- a dull clip (low need) and an uncertain one (high need), each wanting 12 judges
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select is(public.admin_add_clips('[
  {"audio_path": "s-dull.mp3", "source_uid": "pool:s-dull", "kind": "unsure", "uncertainty": 0.1, "target_votes": 12},
  {"audio_path": "s-hot.mp3", "source_uid": "pool:s-hot", "kind": "unsure", "uncertainty": 0.95, "target_votes": 12}
]'::jsonb), 2, 'two clips queued');
reset role;

-- the clip ids, looked up here: labellers cannot read the clip table
create temp table ids as select id, audio_path from public.clips where audio_path like 's-%';
grant select on ids to authenticated;

-- gale answers the dull clip
insert into public.assignments (user_id, clip_id, assigned_at)
select '00000000-0000-0000-0000-0000000000f1', id, now() - interval '10 seconds' from public.clips where audio_path = 's-dull.mp3';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000f1", "role": "authenticated"}';
select lives_ok($$select * from public.submit_label((select id from ids where audio_path = 's-dull.mp3'), array['tapping'], null, false, 4000)$$,
                'gale answers the dull clip');

-- hugo gets the clip gale started, though the untouched one is needier
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000f2", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 's-dull.mp3', 'a started clip is finished before a fresh one is opened');
reset role;

-- breadth first: a started clip with `breadth_votes` (3) judges gives way to a fresh one, though it wants 12
delete from public.assignments where user_id = '00000000-0000-0000-0000-0000000000f2';
update public.clips set votes = 3 where audio_path = 's-dull.mp3';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000f2", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 's-hot.mp3', 'a clip with three judges gives way to the neediest fresh one');
reset role;
update public.clips set votes = 1 where audio_path = 's-dull.mp3';

-- once a clip has its quorum it is no longer preferred
delete from public.assignments where user_id = '00000000-0000-0000-0000-0000000000f2';
update public.clips set target_votes = 1 where audio_path = 's-dull.mp3';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000f2", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 's-hot.mp3', 'a clip at quorum gives way to the neediest fresh one');
reset role;

-- thin classes first: queued with a thin title class, a clip carries a boost
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select public.admin_add_clips('[
  {"audio_path": "s-thin.mp3", "source_uid": "pool:s-thin", "kind": "unsure", "uncertainty": 0.2, "target_votes": 12,
   "info": {"title_class": "spraying"}},
  {"audio_path": "s-thin2.mp3", "source_uid": "pool:s-thin2", "kind": "unsure", "uncertainty": 0.2, "target_votes": 12,
   "info": {"title_class": "page turning"}}
]'::jsonb);
reset role;
select ok((select boost > 0 from public.clips where audio_path = 's-thin.mp3'), 'a spraying clip is boosted');
select ok((select boost > 0 from public.clips where audio_path = 's-thin2.mp3'), 'page turning counts as paper rustling');

-- an ordinary started clip (s-dull, 1 judge) and a needier ordinary fresh one (s-hot) both give way to a thin clip
delete from public.assignments where user_id = '00000000-0000-0000-0000-0000000000f2';
update public.clips set target_votes = 12, votes = 1 where audio_path = 's-dull.mp3';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000f2", "role": "authenticated"}';
select ok((select audio_path from public.next_clip()) in ('s-thin.mp3', 's-thin2.mp3'), 'a thin-class clip comes before ordinary ones');
reset role;

-- among thin clips, one already started is topped up before a fresh one is opened
delete from public.assignments where user_id = '00000000-0000-0000-0000-0000000000f2';
update public.clips set votes = 1 where audio_path = 's-thin2.mp3';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000f2", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 's-thin2.mp3', 'a started thin clip is topped up first');
reset role;

-- taking a class off the list and recomputing removes its boost
delete from private.thin_classes where label = 'spraying';
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select ok(public.admin_refresh_boost() >= 1, 'boosts are recomputed');
reset role;
select is((select boost from public.clips where audio_path = 's-thin.mp3'), 0::real, 'and spraying is no longer boosted');

select * from finish();
rollback;
