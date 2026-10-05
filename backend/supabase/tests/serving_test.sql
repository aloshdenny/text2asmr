-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(4);

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

-- once a clip has its quorum it is no longer preferred
delete from public.assignments where user_id = '00000000-0000-0000-0000-0000000000f2';
update public.clips set target_votes = 1 where audio_path = 's-dull.mp3';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000f2", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 's-hot.mp3', 'a clip at quorum gives way to the neediest fresh one');
reset role;

select * from finish();
rollback;
