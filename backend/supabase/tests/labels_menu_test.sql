-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(14);

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

-- clips whose titles promise a new sound are served before the older thin classes
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select is(public.admin_add_clips('[
  {"audio_path": "b1.mp3", "source_uid": "pool:boost-new", "kind": "unlabelled", "priority": 0.5, "info": {"title_tags": ["squirting", "spanking"]}},
  {"audio_path": "b2.mp3", "source_uid": "pool:boost-old", "kind": "unsure", "priority": 0.5, "info": {"unsure": ["brushing"]}},
  {"audio_path": "b3.mp3", "source_uid": "pool:boost-none", "kind": "unsure", "priority": 0.5, "info": {"unsure": ["whispering"]}}
]'::jsonb), 3, 'three clips queued');
select is(public.admin_tag_clips('[{"source_uid": "pool:boost-old", "title_tags": ["growling"]}]'::jsonb), 1, 'title tags can be added to a queued clip');
reset role;
select results_eq($$select s.source_uid, c.boost from public.clips c join private.clip_sources s on s.clip_id = c.id
                    where s.source_uid like 'pool:boost-%' order by s.source_uid$$,
                  $$values ('pool:boost-new', 0.5::real), ('pool:boost-none', 0::real), ('pool:boost-old', 0.5::real)$$,
                  'a new-label title tag boosts 0.5, a thin unsure class 0.3, anything else 0');
select is(private.thin_boost('{"unsure": ["brushing"]}'::jsonb), 0.3::real, 'the older thin classes keep 0.3');

select * from finish();
rollback;
