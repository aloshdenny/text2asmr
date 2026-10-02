-- Kit labellers claim their profile by signing up, not by a set-password email.
--
-- The first version made each of them a password-less sign-in and asked them to set a password by email, which
-- needs mail delivery to people outside the project. Now their profile waits with a claim_email and no sign-in at
-- all; when someone signs up with that email (and the address counts as confirmed), the profile -- with its kit
-- labels -- moves onto the new account, which then goes through the same onboarding as everyone else (username,
-- optional picture).

alter table public.profiles add column claim_email text unique check (claim_email = lower(claim_email));
alter table public.profiles add column onboarded boolean not null default true;
grant select (onboarded) on public.profiles to authenticated;
grant update (onboarded) on public.profiles to authenticated;

-- a claimed profile changes id to the new sign-in; its labels follow
alter table public.labels drop constraint labels_user_id_fkey,
  add constraint labels_user_id_fkey foreign key (user_id) references public.profiles (id) on update cascade;
alter table public.imported_labels drop constraint imported_labels_user_id_fkey,
  add constraint imported_labels_user_id_fkey foreign key (user_id) references public.profiles (id) on update cascade;

create function private.claim_profile()
returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  if new.email is null or new.email_confirmed_at is null then return new; end if;
  if tg_op = 'UPDATE' and old.email_confirmed_at is not null then return new; end if;
  if exists (select 1 from public.profiles p where p.id = new.id) then return new; end if;
  update public.profiles p set id = new.id, claim_email = null
  where p.claim_email = lower(new.email) and p.deleted_at is null;
  return new;
end $$;

create trigger claim_profile_on_signup after insert or update of email_confirmed_at on auth.users
  for each row execute function private.claim_profile();

-- Profiles made for kit labellers: no sign-in until they sign up with this email.
create or replace function private.import_contributor(p_email text, p_username text, p_display_name text, p_fusion_name text)
returns uuid
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid;
  existing uuid;
begin
  select u.id into existing from auth.users u where lower(u.email) = lower(p_email) and u.email_confirmed_at is not null;
  select p.id into uid from public.profiles p
  where p.fusion_name = p_fusion_name or p.claim_email = lower(p_email) or p.id = existing limit 1;
  if uid is null then
    insert into public.profiles (id, username, display_name, fusion_name, imported, onboarded, claim_email)
    values (coalesce(existing, gen_random_uuid()), p_username, p_display_name, p_fusion_name, true, false,
            case when existing is null then lower(p_email) end)
    returning id into uid;
  else
    update public.profiles p set fusion_name = p_fusion_name, imported = true where p.id = uid;
  end if;
  return uid;
end $$;
revoke all on function private.import_contributor(text, text, text, text) from public, anon, authenticated, service_role;
revoke all on function private.claim_profile() from public, anon, authenticated, service_role;

-- Sign-in help: an email whose kit labels are waiting to be claimed is told to sign up instead. As before, only
-- those few addresses are ever reported; everything else answers 'ok'.
create or replace function public.account_status(p_email text)
returns text
language sql stable security definer set search_path = '' as $$
  select case when exists (
    select 1 from public.profiles p where p.claim_email = lower(btrim(p_email)) and p.deleted_at is null)
  then 'unclaimed' else 'ok' end;
$$;

-- Accounts the first version made (password-less sign-ins nobody has used) become claimable profiles.
update public.profiles p set claim_email = lower(u.email), onboarded = false
from auth.users u
where u.id = p.id and p.imported and coalesce(u.encrypted_password, '') = '' and u.last_sign_in_at is null;
delete from auth.users u using public.profiles p where u.id = p.id and p.claim_email is not null;
