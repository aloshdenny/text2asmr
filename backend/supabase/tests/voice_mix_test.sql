-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(6);

update private.settings set value = 0 where key in ('repeat_rate', 'control_rate_max', 'control_rate_min');
update public.clips set active = false;
insert into auth.users (id, email) values
  ('f1000000-0000-0000-0000-000000000000', 'vm1@example.test'), ('f2000000-0000-0000-0000-000000000000', 'vm2@example.test'),
  ('f3000000-0000-0000-0000-000000000000', 'vm3@example.test');
insert into public.profiles (id, username, adult_confirmed_at) values
  ('f1000000-0000-0000-0000-000000000000', 'vm_one', now()), ('f2000000-0000-0000-0000-000000000000', 'vm_two', now()),
  ('f3000000-0000-0000-0000-000000000000', 'vm_three', now());

-- a needy female-voice clip and a barely needed male-voice one
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select public.admin_add_clips('[
  {"audio_path": "vf.mp3", "source_uid": "pool:vm-female", "kind": "unsure", "priority": 0.95, "info": {"voice": "female"}},
  {"audio_path": "vm.mp3", "source_uid": "pool:vm-male", "kind": "unsure", "priority": 0.05}
]'::jsonb);
select is(public.admin_merge_info('[{"source_uid": "pool:vm-male", "info": {"voice": "male"}}]'::jsonb), 1, 'a queued clip learns its voice');
reset role;
select is((select c.voice from public.clips c join private.clip_sources s on s.clip_id = c.id where s.source_uid = 'pool:vm-male'), 'male',
          '... and the clip carries it');

-- every draw a male draw: the male clip comes first despite its low need
insert into private.settings (key, value) values ('male_share', 1) on conflict (key) do update set value = 1;
set local role authenticated;
set local request.jwt.claims = '{"sub": "f1000000-0000-0000-0000-000000000000", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 'vm.mp3', 'a male draw serves a male-voice clip');
reset role;

-- no male draws: neediest first, as before
update private.settings set value = 0 where key = 'male_share';
set local role authenticated;
set local request.jwt.claims = '{"sub": "f2000000-0000-0000-0000-000000000000", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 'vf.mp3', 'otherwise the neediest clip comes first');
reset role;

-- a male draw with no male-voice clip left falls back to the rest
update private.settings set value = 1 where key = 'male_share';
update public.clips c set active = false from private.clip_sources s where s.clip_id = c.id and s.source_uid = 'pool:vm-male';
set local role authenticated;
set local request.jwt.claims = '{"sub": "f3000000-0000-0000-0000-000000000000", "role": "authenticated"}';
select is((select audio_path from public.next_clip()), 'vf.mp3', 'with no male-voice clip left, a male draw takes any clip');
reset role;

select is(private.setting('male_share', 0.3), 1::numeric, 'the share is a setting');

select * from finish();
rollback;
