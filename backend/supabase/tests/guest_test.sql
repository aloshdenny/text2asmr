-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(11);

insert into auth.users (id, email, is_anonymous) values
  ('00000000-0000-0000-0000-0000000000a1', null, true),
  ('00000000-0000-0000-0000-0000000000a2', 'full@example.com', false);
insert into public.profiles (id, username) values ('00000000-0000-0000-0000-0000000000a2', 'full_member');

-- a guest session gets a profile, once
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000a1", "role": "authenticated", "is_anonymous": true}';
select lives_ok('select public.start_guest()', 'a guest session starts');
select lives_ok('select public.start_guest()', 'starting again (a second tab) is harmless');
select is((select username from public.profiles where id = '00000000-0000-0000-0000-0000000000a1'), 'guest_000000000000',
          'the guest gets a guest_ username');
select is((select guest from public.profiles where id = '00000000-0000-0000-0000-0000000000a1'), true, 'and is flagged as a guest');
reset role;

-- give both people a label so they would show up on the board
insert into public.imported_labels (user_id, source_uid, labels, kit, created_at) values
  ('00000000-0000-0000-0000-0000000000a1', 'guest-test:1', array['tapping'], 'guest-test', now()),
  ('00000000-0000-0000-0000-0000000000a2', 'guest-test:2', array['tapping'], 'guest-test', now());

set local role anon;
set local request.jwt.claims = '{"role": "anon"}';
select is((select count(*) from public.leaderboard(null, 200) where username = 'guest_000000000000'), 0::bigint,
          'guests are not on the leaderboard');
select is((select count(*) from public.profile_stats('guest_000000000000')), 0::bigint, 'nobody else sees a guest''s profile');
select is((select count(*) from public.leaderboard(null, 200) where username = 'full_member'), 1::bigint, 'members still are');
reset role;

set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000a1", "role": "authenticated", "is_anonymous": true}';
select is((select labelled from public.profile_stats('guest_000000000000')), 1::bigint, 'a guest sees their own count');
reset role;

-- a signed-up member cannot turn themselves into a guest
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000a2", "role": "authenticated", "is_anonymous": false}';
select throws_ok('select public.start_guest()', '42501', null, 'only anonymous sessions start as guests');
reset role;

-- the guest links an email (no longer anonymous) and picks a username: now a member, on the board. The session token
-- still says anonymous until it refreshes, so the flag must follow the account, not the token.
update auth.users set email = 'linked@example.com', is_anonymous = false where id = '00000000-0000-0000-0000-0000000000a1';
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-0000000000a1", "role": "authenticated", "is_anonymous": true}';
update public.profiles set username = 'quiet_guest' where id = '00000000-0000-0000-0000-0000000000a1';
select is((select guest from public.profiles where id = '00000000-0000-0000-0000-0000000000a1'), false,
          'saving the profile after linking an email clears the guest flag');
reset role;
set local role anon;
set local request.jwt.claims = '{"role": "anon"}';
select is((select count(*) from public.leaderboard(null, 200) where username = 'quiet_guest'), 1::bigint,
          'and their labels now count on the board');
reset role;

select * from finish();
rollback;
