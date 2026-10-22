-- Guests label SFW clips only, with no preferences. The 18+ confirmation that covers adult clips is the one taken at
-- sign-up, so preferences and NSFW clips are for members. A guest's profile always reads SFW with no sound or voice
-- preference, whatever tries to set it (serving then keeps adult clips and "Which is hotter?" away from them); when a
-- guest becomes a member the preferences are cleared, so they are asked afresh after signing up.

create function private.guest_sfw()
returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  if new.guest then
    new.sound_pref := null;
    new.content_pref := 'sfw';
    new.voice_pref := null;
  elsif tg_op = 'UPDATE' and old.guest then
    new.sound_pref := null;
    new.content_pref := null;
    new.voice_pref := null;
  end if;
  return new;
end $$;
revoke all on function private.guest_sfw() from public, anon, authenticated;

-- named to fire after profiles_clear_guest, which may turn the guest flag off in the same update
create trigger profiles_guest_sfw before insert or update on public.profiles
  for each row execute function private.guest_sfw();

update public.profiles set content_pref = 'sfw' where guest;

-- adult clips a guest already has open go back to the pool
delete from public.assignments a
using public.clips c, public.profiles p
where c.id = a.clip_id and p.id = a.user_id and a.state = 'open' and c.nsfw and p.guest;
