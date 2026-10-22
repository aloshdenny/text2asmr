-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(7);

update private.settings set value = 0 where key in ('repeat_rate', 'control_rate_max', 'control_rate_min');
update public.clips set active = false;
insert into auth.users (id, email, is_anonymous) values
  ('e1000000-0000-0000-0000-000000000000', null, true),
  ('e2000000-0000-0000-0000-000000000000', 'gs-member@example.test', false);
insert into public.profiles (id, username, adult_confirmed_at) values ('e2000000-0000-0000-0000-000000000000', 'gs_member', now());

-- a needy adult clip (a voice recording) and a barely needed SFW one
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select public.admin_add_clips('[
  {"audio_path": "gs-adult.mp3", "source_uid": "pool:gs-adult", "kind": "unsure", "priority": 0.95, "info": {"voice": "female"}},
  {"audio_path": "gs-sfw.mp3", "source_uid": "pool:gs-sfw", "kind": "unsure", "priority": 0.05}
]'::jsonb);
reset role;

-- a guest starts out SFW, and cannot choose otherwise
set local role authenticated;
set local request.jwt.claims = '{"sub": "e1000000-0000-0000-0000-000000000000", "role": "authenticated", "is_anonymous": true}';
select public.start_guest();
reset role;
select is((select content_pref from public.profiles where id = 'e1000000-0000-0000-0000-000000000000'), 'sfw',
          'a guest profile reads SFW');
set local role authenticated;
set local request.jwt.claims = '{"sub": "e1000000-0000-0000-0000-000000000000", "role": "authenticated", "is_anonymous": true}';
select public.set_preferences('speech', 'nsfw', 'female');
reset role;
select ok((select content_pref = 'sfw' and sound_pref is null and voice_pref is null
           from public.profiles where id = 'e1000000-0000-0000-0000-000000000000'),
          'a guest cannot set preferences');

-- serving: the SFW clip despite the adult one being needier, and nothing once only adult clips are left
set local role authenticated;
set local request.jwt.claims = '{"sub": "e1000000-0000-0000-0000-000000000000", "role": "authenticated", "is_anonymous": true}';
select is((select audio_path from public.next_clip()), 'gs-sfw.mp3', 'a guest gets the SFW clip');
reset role;
delete from public.assignments where user_id = 'e1000000-0000-0000-0000-000000000000';
update public.clips c set active = false from private.clip_sources s where s.clip_id = c.id and s.source_uid = 'pool:gs-sfw';
set local role authenticated;
set local request.jwt.claims = '{"sub": "e1000000-0000-0000-0000-000000000000", "role": "authenticated", "is_anonymous": true}';
select is((select count(*) from public.next_clip()), 0::bigint, 'and never an adult one');
reset role;

-- a member who chose NSFW gets the adult clip
set local role authenticated;
set local request.jwt.claims = '{"sub": "e2000000-0000-0000-0000-000000000000", "role": "authenticated"}';
select public.set_preferences('speech', 'nsfw', 'female');
select is((select audio_path from public.next_clip()), 'gs-adult.mp3', 'a member who chose NSFW gets it');
reset role;

-- the guest signs up: no longer SFW-only, and asked afresh
update auth.users set email = 'gs-linked@example.test', is_anonymous = false where id = 'e1000000-0000-0000-0000-000000000000';
set local role authenticated;
set local request.jwt.claims = '{"sub": "e1000000-0000-0000-0000-000000000000", "role": "authenticated", "is_anonymous": true}';
update public.profiles set username = 'gs_linked' where id = 'e1000000-0000-0000-0000-000000000000';
reset role;
select ok((select not guest and content_pref is null from public.profiles where id = 'e1000000-0000-0000-0000-000000000000'),
          'signing up clears the SFW default so preferences are asked');
set local role authenticated;
set local request.jwt.claims = '{"sub": "e1000000-0000-0000-0000-000000000000", "role": "authenticated"}';
select public.set_preferences('both', 'both', 'both');
reset role;
select is((select content_pref from public.profiles where id = 'e1000000-0000-0000-0000-000000000000'), 'both',
          'and a member''s choice sticks');

select * from finish();
rollback;
