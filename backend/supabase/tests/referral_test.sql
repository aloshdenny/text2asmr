-- supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(8);

insert into auth.users (id, email, is_anonymous) values
  ('d1000000-0000-0000-0000-000000000000', null, true),
  ('d2000000-0000-0000-0000-000000000000', 'rf-old@example.test', false);
insert into public.profiles (id, username, created_at) values ('d2000000-0000-0000-0000-000000000000', 'rf_old', now() - interval '3 days');

-- a new guest arriving from a creator's link is credited to them, once
set local role authenticated;
set local request.jwt.claims = '{"sub": "d1000000-0000-0000-0000-000000000000", "role": "authenticated", "is_anonymous": true}';
select public.start_guest();
select lives_ok($$select public.set_referral('AcelynAura')$$, 'a new guest records the referral');
select public.set_referral('someone.else');
select public.set_referral('not a handle!');
reset role;
select is((select referred_by from public.profiles where id = 'd1000000-0000-0000-0000-000000000000'), 'acelynaura',
          'the first handle sticks, lowercased; later or malformed ones are ignored');

-- someone who was already labelling is not counted
set local role authenticated;
set local request.jwt.claims = '{"sub": "d2000000-0000-0000-0000-000000000000", "role": "authenticated"}';
select public.set_referral('acelynaura');
select throws_ok('select referred_by from public.profiles', '42501', null, 'labellers cannot read referrals');
select throws_ok('select * from public.admin_referrals()', '42501', null, 'nor the report');
reset role;
select is((select referred_by from public.profiles where id = 'd2000000-0000-0000-0000-000000000000'), null,
          'an older profile gets no credit');

-- the guest labels, then signs up: the credit and their labels stay with the creator
insert into public.labels (user_id, clip_id, labels)
select 'd1000000-0000-0000-0000-000000000000', c.id, array['tapping'] from public.clips c limit 1;
update auth.users set email = 'rf-linked@example.test', is_anonymous = false where id = 'd1000000-0000-0000-0000-000000000000';
set local role authenticated;
set local request.jwt.claims = '{"sub": "d1000000-0000-0000-0000-000000000000", "role": "authenticated", "is_anonymous": true}';
update public.profiles set username = 'rf_linked' where id = 'd1000000-0000-0000-0000-000000000000';
reset role;

set local role service_role;
set local request.jwt.claims = '{"role": "service_role"}';
select is((select people from public.admin_referrals() where ref = 'acelynaura'), 1::bigint, 'the report counts the person');
select is((select members from public.admin_referrals() where ref = 'acelynaura'), 1::bigint, 'as a member once they sign up');
select is((select labels from public.admin_referrals() where ref = 'acelynaura'),
          (select count(*) from public.labels where user_id = 'd1000000-0000-0000-0000-000000000000'), 'with their labels');
reset role;

select * from finish();
rollback;
