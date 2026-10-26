-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(9);

insert into auth.users (id, email) values
  ('f1000000-0000-0000-0000-000000000000', 'xb-one@example.test'),
  ('f2000000-0000-0000-0000-000000000000', 'xb-two@example.test'),
  ('f3000000-0000-0000-0000-000000000000', 'xb-three@example.test'),
  ('f4000000-0000-0000-0000-000000000000', 'xb-four@example.test');
insert into public.profiles (id, username) values
  ('f1000000-0000-0000-0000-000000000000', 'xb_one'), ('f2000000-0000-0000-0000-000000000000', 'xb_two'),
  ('f3000000-0000-0000-0000-000000000000', 'xb_three'), ('f4000000-0000-0000-0000-000000000000', 'xb_four');
set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select public.admin_add_clips('[
  {"audio_path": "xb-ok.mp3", "source_uid": "pool:xb-ok", "kind": "unsure"},
  {"audio_path": "xb-bad.mp3", "source_uid": "pool:xb-bad", "kind": "unsure"}
]'::jsonb);
reset role;

-- answers on both clips, then the second one is blocked
insert into public.labels (user_id, clip_id, labels, other_text, unsure)
select u, c.id, l, t, s
from (values
  ('f1000000-0000-0000-0000-000000000000'::uuid, 'xb-ok.mp3', array['tapping'], null, false),
  ('f2000000-0000-0000-0000-000000000000'::uuid, 'xb-ok.mp3', array['tapping', 'something else'], '', false),
  ('f3000000-0000-0000-0000-000000000000'::uuid, 'xb-ok.mp3', array['something else'], null, false),
  ('f4000000-0000-0000-0000-000000000000'::uuid, 'xb-ok.mp3', array['scratching'], null, true),
  ('f1000000-0000-0000-0000-000000000000'::uuid, 'xb-bad.mp3', array['whispering'], null, false),
  ('f2000000-0000-0000-0000-000000000000'::uuid, 'xb-bad.mp3', array['something else'], 'a train', false)
) v (u, path, l, t, s)
join public.clips c on c.audio_path = v.path;
update public.clips set blocked = true, active = false where audio_path = 'xb-bad.mp3';

-- a "Which is hotter?" vote on a pair with the blocked clip, and one on a clean pair
insert into public.clips (audio_path, duration_s) values ('xb-ok2.mp3', 6);
insert into private.hot_pairs (a, b, sound)
select least(c1.id, c2.id), greatest(c1.id, c2.id), 'whispering'
from public.clips c1, public.clips c2 where c1.audio_path = 'xb-ok.mp3' and c2.audio_path in ('xb-bad.mp3', 'xb-ok2.mp3');
insert into private.clip_sources (clip_id, source_uid, kind)
select id, 'pool:xb-ok2', 'confident' from public.clips where audio_path = 'xb-ok2.mp3';
insert into private.hot_votes (user_id, pair_id, winner)
select 'f1000000-0000-0000-0000-000000000000', h.id, h.a from private.hot_pairs h
join public.clips c on c.id in (h.a, h.b) where c.audio_path in ('xb-bad.mp3', 'xb-ok2.mp3');

set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select is((select count(*) from public.admin_export_labels() where source_uid = 'pool:xb-bad'), 0::bigint,
          'answers on a blocked clip are not exported');
select is((select count(*) from public.admin_export_labels() where source_uid = 'pool:xb-ok'), 2::bigint,
          'the clean clip keeps its two answers that say something');
select is((select labels from public.admin_export_labels() where username = 'xb_one' and source_uid = 'pool:xb-ok'),
          array['tapping'], 'a plain answer is untouched');
select is((select labels from public.admin_export_labels() where username = 'xb_two' and source_uid = 'pool:xb-ok'),
          array['tapping'], 'a bare "something else" is dropped from an answer');
select is((select count(*) from public.admin_export_labels() where username = 'xb_three'), 0::bigint,
          'an answer that was only a bare "something else" is skipped');
select is((select count(*) from public.admin_export_labels() where username = 'xb_four'), 0::bigint,
          'a can''t tell answer that also names sounds is skipped');
select is((select count(*) from public.admin_export_hot() where labeller = 'crowd:f1000000'), 1::bigint,
          'hot votes on a pair with a blocked clip are not exported');
select is((select count(*) from public.admin_export_hot() where labeller = 'crowd:f1000000' and 'pool:xb-ok2' in (a_uid, b_uid)),
          1::bigint, 'the clean pair is');
reset role;
select is((select count(*) from public.labels l join public.clips c on c.id = l.clip_id where c.audio_path in ('xb-ok.mp3', 'xb-bad.mp3')),
          6::bigint, 'the rows themselves stay');

select * from finish();
rollback;
