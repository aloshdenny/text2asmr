-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(10);

-- the menu, as anyone sees it
set local role anon;
select is((select count(*) from public.label_options), 31::bigint, 'the menu has 31 labels');
select is((select count(*) from public.label_options where grp = 'NSFW'), 10::bigint, 'ten of them are in the NSFW group');
select is((select grp from public.label_options where key = 'clapping'), 'Triggers', 'plain clapping is a trigger');
select is((select title from public.label_options where key = 'sex slapping'), 'clapping😉', 'sex slapping shows as clapping😉');
select is((select count(*) from public.label_options where title is null), 30::bigint, 'every other label shows its key');
select is((select count(*) from public.label_options where added_at = '2026-10-02 00:00:00+00'), 20::bigint,
          'the first twenty keep their launch date');
select ok((select min(added_at) from public.label_options where sort > 20) > '2026-10-17', 'the new ones are dated when they were added');
reset role;

select throws_ok($$insert into public.label_options (key, grp, hint, sort) values ('banjo', 'Music', 'a banjo', 99)$$,
                 '23514', null, 'groups are limited to Voice, Triggers, NSFW and Other');

-- a labeller can tick the new labels
update private.settings set value = 0 where key in ('repeat_rate', 'control_rate_max', 'control_rate_min');
update public.clips set active = false;
insert into auth.users (id, email) values ('00000000-0000-0000-0000-0000000000e1', 'e1@example.test');
insert into public.profiles (id, username, adult_confirmed_at) values ('00000000-0000-0000-0000-0000000000e1', 'erin', now());
set local role service_role;
select public.admin_add_clips('[{"audio_path": "m1.mp3", "source_uid": "pool:menu-one", "kind": "unsure", "priority": 0.9}]'::jsonb);
reset role;
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000e1", "role": "authenticated"}';
select audio_path from public.next_clip();
reset role;
update public.assignments set assigned_at = now() - interval '10 seconds';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000e1", "role": "authenticated"}';
select is((select total from public.submit_label((select clip_id from public.next_clip()),
                                                  array['squirting', 'sex slapping', 'clapping', 'moaning'], null, false, 4000)),
          1::bigint, 'new labels are accepted');
select is((select labels from public.labels where user_id = '00000000-0000-0000-0000-0000000000e1'),
          array['clapping', 'moaning', 'sex slapping', 'squirting'], 'and stored by key');
reset role;

select * from finish();
rollback;
