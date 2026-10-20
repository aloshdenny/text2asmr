-- Preferences, the gold set and the timeline labeller (beta).
--
-- 1. Labels: "clapping😉" now reads "fucking" (the key stays "sex slapping"); singing joins Voice; "can't tell" and
--    "skip" leave the labeller (what someone cannot place goes under "something else"), so its hint says so.
-- 2. Preferences, asked once of everyone who labels: speech or non-speech sounds (or both); SFW, NSFW or both; and,
--    when NSFW is in, female, male or both voices. Speech means the voice corpora, non-speech the trigger recordings;
--    that one only orders the draws (the rest fills in when it runs out). SFW / NSFW and the voice are hard filters:
--    an SFW labeller never gets an adult clip or a "Which is hotter?" break, a women-only labeller never a male voice.
--    The voice corpora count as NSFW (clips.nsfw), except recordings titled SFW. Preferences are private: no column
--    grant, only the owner's own functions read or write them.
-- 3. The gold set: 12 clips per label (picked by admin_pick_gold from the AI probabilities, title tags and title
--    classes), each to be heard by `gold_votes` (12) people and up to 20. It is served before everything else. When
--    every gold clip has 12 answers the timeline labeller replaces the tag list for everyone (labelling_state);
--    before that, labellers can opt into it as a beta, and 'timeline_ui' forces it on (1) or off (-1).
-- 4. Timeline answers: up to 6 sounds per clip, each with a start and end in milliseconds and a lane (3 above the
--    waveform, 3 below), stored in labels.spans; the labels array stays the set of sounds, so everything that reads
--    labels keeps working. Answers from the last 24 hours can be revisited and changed (prev / next).

-- 1. labels
update public.label_options set title = 'fucking' where key = 'sex slapping';
insert into public.label_options (key, grp, hint, sort) values ('singing', 'Voice', 'singing or humming a tune', 34);
update public.label_options set hint = 'a sound that is not on this list, or one you cannot place: type what you hear'
  where key = 'something else';
insert into private.thin_classes (label) values ('singing') on conflict do nothing;
insert into private.new_labels (label) values ('singing') on conflict do nothing;

-- 2. preferences
alter table public.profiles
  add column sound_pref text check (sound_pref in ('speech', 'nonspeech', 'both')),
  add column content_pref text check (content_pref in ('sfw', 'nsfw', 'both')),
  add column voice_pref text check (voice_pref in ('female', 'male', 'both')),
  add column beta_timeline boolean not null default false;

create function public.set_preferences(p_sound text default null, p_content text default null, p_voice text default null,
                                       p_beta boolean default null)
returns void
language plpgsql security definer set search_path = '' as $$
begin
  if (select auth.uid()) is null then raise exception 'sign in first' using errcode = '28000'; end if;
  if p_sound is not null and p_sound not in ('speech', 'nonspeech', 'both') then
    raise exception 'sounds are speech, nonspeech or both' using errcode = '22023';
  end if;
  if p_content is not null and p_content not in ('sfw', 'nsfw', 'both') then
    raise exception 'clips are sfw, nsfw or both' using errcode = '22023';
  end if;
  if p_voice is not null and p_voice not in ('female', 'male', 'both') then
    raise exception 'voices are female, male or both' using errcode = '22023';
  end if;
  update public.profiles p
  set sound_pref = coalesce(p_sound, p.sound_pref), content_pref = coalesce(p_content, p.content_pref),
      voice_pref = coalesce(p_voice, p.voice_pref), beta_timeline = coalesce(p_beta, p.beta_timeline)
  where p.id = (select auth.uid());
end $$;
revoke all on function public.set_preferences(text, text, text, boolean) from public, anon;
grant execute on function public.set_preferences(text, text, text, boolean) to authenticated;

-- adult clips: the voice corpora, unless info.nsfw says otherwise (recordings titled SFW)
alter table public.clips add column nsfw boolean not null default false;
create or replace function private.set_clip_voice()
returns trigger
language plpgsql security definer set search_path = '' as $$
declare
  v text := case when new.info ->> 'voice' in ('male', 'female') then new.info ->> 'voice' end;
  x boolean := case when new.info ->> 'nsfw' in ('true', 'false') then (new.info ->> 'nsfw')::boolean else v is not null end;
begin
  update public.clips c set voice = v, nsfw = x
  where c.id = new.clip_id and (c.voice is distinct from v or c.nsfw is distinct from x);
  return new;
end $$;
update public.clips set nsfw = true where voice is not null;

-- may this person be served this clip (SFW / NSFW, and the voice they asked not to hear)
create function private.clip_allowed(p_voice text, p_nsfw boolean, p_content text, p_excl text)
returns boolean
language sql immutable as $$
  select (p_content is null or p_content = 'both' or (p_content = 'sfw' and not p_nsfw) or (p_content = 'nsfw' and p_nsfw))
     and (p_excl is null or p_voice is null or p_voice <> p_excl);
$$;

-- 3. the gold set
alter table public.clips add column gold boolean not null default false;
create index clips_gold_open on public.clips (votes desc) where active and gold;
insert into private.settings (key, value) values ('gold_votes', 12), ('timeline_ui', 0) on conflict (key) do nothing;

-- pick up to p_per_label gold clips for every label from what the queue knows about it (service role)
create function public.admin_pick_gold(p_per_label integer default 12)
returns table (label text, picked integer)
language plpgsql security definer set search_path = '' as $$
declare
  o record;
  ids uuid[];
begin
  perform private.require_service_role();
  for o in select l.key from public.label_options l where l.key <> 'something else' order by l.sort loop
    select array_agg(id) into ids from (
      select c.id
      from public.clips c join private.clip_sources s on s.clip_id = c.id
      cross join lateral (select greatest(
        coalesce((s.info -> 'fused' ->> o.key)::real, 0),
        case when s.info -> 'title_tags' ? o.key then 0.7 else 0 end,
        case when s.info ->> 'title_class' = o.key
                  or (o.key = 'paper rustling' and s.info ->> 'title_class' = 'page turning') then 0.7 else 0 end,
        case when s.info -> 'unsure' ? o.key then 0.5 else 0 end) as ev) e
      where c.active and not c.blocked and not c.gold and s.kind <> 'control' and e.ev >= 0.3
      -- likely positives first, at random among them (not only the easiest, most confident ones)
      order by (e.ev >= 0.5) desc, random()
      limit p_per_label) t;
    if ids is not null then
      update public.clips c set gold = true, target_votes = greatest(c.target_votes, 20) where c.id = any (ids);
      update private.clip_sources s set info = s.info || jsonb_build_object('gold_for', o.key) where s.clip_id = any (ids);
    end if;
    label := o.key; picked := coalesce(cardinality(ids), 0);
    return next;
  end loop;
end $$;
revoke all on function public.admin_pick_gold(integer) from public, anon, authenticated;
grant execute on function public.admin_pick_gold(integer) to service_role;

-- which labeller to show, the gold progress, and this person's preferences
create function public.labelling_state()
returns table (mode text, gold_done integer, gold_total integer, gold_votes integer, sound_pref text, content_pref text,
               voice_pref text, beta boolean)
language sql stable security definer set search_path = '' as $$
  with g as (
    select count(*) filter (where c.votes >= private.setting('gold_votes', 12)) as done, count(*) as total
    from public.clips c where c.gold and not c.blocked
  ), me as (
    select p.sound_pref, p.content_pref, p.voice_pref, p.beta_timeline from public.profiles p where p.id = (select auth.uid())
  )
  select case when private.setting('timeline_ui', 0) = 1 then 'timeline'
              when private.setting('timeline_ui', 0) = -1 then 'tags'
              when (select total from g) > 0 and (select done from g) >= (select total from g) then 'timeline'
              when coalesce((select beta_timeline from me), false) then 'timeline'
              else 'tags' end,
         (select done from g)::integer, (select total from g)::integer, private.setting('gold_votes', 12)::integer,
         (select sound_pref from me), (select content_pref from me), (select voice_pref from me),
         coalesce((select beta_timeline from me), false);
$$;
grant execute on function public.labelling_state() to authenticated;

-- 4. timeline answers
alter table public.labels add column spans jsonb;

-- validated spans for a clip: [{label, start_ms, end_ms, lane}], at most 6, inside the clip, one per lane
create function private.clean_spans(p_clip uuid, p_spans jsonb)
returns jsonb
language plpgsql stable security definer set search_path = '' as $$
declare
  dur integer := (select round(coalesce(c.duration_s, 6) * 1000)::integer from public.clips c where c.id = p_clip);
  e jsonb;
  i integer := 0;
  s integer;
  t integer;
  lane integer;
  lanes integer[] := '{}';
  res jsonb := '[]';
begin
  if p_spans is null or jsonb_typeof(p_spans) <> 'array' then raise exception 'spans must be a list' using errcode = '22023'; end if;
  if jsonb_array_length(p_spans) > 6 then raise exception 'up to 6 sounds per clip' using errcode = '22023'; end if;
  for e in select x from jsonb_array_elements(p_spans) x loop
    if not exists (select 1 from public.label_options o where o.key = e ->> 'label' and o.key <> 'something else') then
      raise exception 'unknown label' using errcode = '22023';
    end if;
    begin
      s := (e ->> 'start_ms')::integer; t := (e ->> 'end_ms')::integer; lane := coalesce((e ->> 'lane')::integer, i);
    exception when others then raise exception 'start and end must be whole milliseconds' using errcode = '22023';
    end;
    if s is null or t is null or s < 0 or t > dur + 250 or t - s < 20 then
      raise exception 'each sound needs a start and an end inside the clip' using errcode = '22023';
    end if;
    if lane < 0 or lane > 5 or lane = any (lanes) then raise exception 'one sound per lane' using errcode = '22023'; end if;
    lanes := lanes || lane;
    res := res || jsonb_build_object('label', e ->> 'label', 'start_ms', s, 'end_ms', least(t, dur), 'lane', lane);
    i := i + 1;
  end loop;
  return res;
end $$;

create function private.span_labels(p_spans jsonb, p_other text)
returns text[]
language sql immutable set search_path = '' as $$
  select array(select distinct x from (
    select e ->> 'label' as x from jsonb_array_elements(p_spans) e
    union all select 'something else' where nullif(btrim(coalesce(p_other, '')), '') is not null) t order by x);
$$;

create function public.submit_spans(p_clip uuid, p_spans jsonb, p_other text default null, p_listen_ms integer default 0)
returns table (total bigint, today bigint)
language plpgsql security definer set search_path = '' as $$
declare
  clean jsonb := private.clean_spans(p_clip, coalesce(p_spans, '[]'));
  labs text[] := private.span_labels(clean, p_other);
begin
  if cardinality(labs) = 0 then
    raise exception 'mark at least one sound, or say what you hear' using errcode = '22023';
  end if;
  return query select * from public.submit_label(p_clip, labs, p_other, false, p_listen_ms);
  update public.labels l set spans = clean
  where l.user_id = (select auth.uid()) and l.clip_id = p_clip
    and l.attempt = (select max(m.attempt) from public.labels m where m.user_id = (select auth.uid()) and m.clip_id = p_clip);
end $$;
revoke all on function public.submit_spans(uuid, jsonb, text, integer) from public, anon;
grant execute on function public.submit_spans(uuid, jsonb, text, integer) to authenticated;

-- change your own answer from the last 24 hours (prev / next)
create function public.edit_answer(p_clip uuid, p_spans jsonb, p_other text default null)
returns void
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := (select auth.uid());
  clean jsonb := private.clean_spans(p_clip, coalesce(p_spans, '[]'));
  labs text[] := private.span_labels(clean, p_other);
begin
  if uid is null then raise exception 'sign in first' using errcode = '28000'; end if;
  if cardinality(labs) = 0 then
    raise exception 'mark at least one sound, or say what you hear' using errcode = '22023';
  end if;
  update public.labels l
  set labels = labs, spans = clean,
      other_text = case when 'something else' = any (labs) then nullif(btrim(left(p_other, 200)), '') end
  where l.user_id = uid and l.clip_id = p_clip and l.attempt = 1 and l.created_at > now() - interval '24 hours';
  if not found then
    raise exception 'only answers from the last 24 hours can be changed' using errcode = 'P0001';
  end if;
  update public.clips c set agreement = private.clip_agreement(p_clip), crowd_conf = private.clip_consensus(p_clip)
  where c.id = p_clip;
end $$;
revoke all on function public.edit_answer(uuid, jsonb, text) from public, anon;
grant execute on function public.edit_answer(uuid, jsonb, text) to authenticated;

create function public.my_recent_answers(p_limit integer default 20)
returns table (clip_id uuid, audio_path text, duration_s real, labels text[], spans jsonb, other_text text, created_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select l.clip_id, c.audio_path, c.duration_s, l.labels, l.spans, l.other_text, l.created_at
  from public.labels l join public.clips c on c.id = l.clip_id
  where l.user_id = (select auth.uid()) and l.attempt = 1 and l.created_at > now() - interval '24 hours'
    and not c.blocked and c.audio_path is not null
  order by l.created_at desc
  limit least(greatest(coalesce(p_limit, 20), 1), 50);
$$;
revoke all on function public.my_recent_answers(integer) from public, anon;
grant execute on function public.my_recent_answers(integer) to authenticated;

revoke all on function private.clean_spans(uuid, jsonb), private.span_labels(jsonb, text) from public, anon, authenticated;
revoke all on function private.set_clip_voice() from public, anon, authenticated;

-- a labeller may hear again what they answered in the last 24 hours
create or replace function private.can_hear(p_object text)
returns boolean
language sql stable security definer set search_path = '' as $$
  select exists (
           select 1 from public.assignments a join public.clips c on c.id = a.clip_id
           where a.user_id = auth.uid() and a.state = 'open' and c.audio_path = p_object)
      or exists (
           select 1 from public.labels l join public.clips c on c.id = l.clip_id
           where l.user_id = auth.uid() and c.audio_path = p_object and l.created_at > now() - interval '24 hours')
      or exists (
           select 1 from private.interludes i join private.rai_items r on r.id = i.rai_item
           where i.user_id = auth.uid() and r.audio_path = p_object)
      or exists (
           select 1 from private.interludes i join private.hot_pairs h on h.id = i.hot_pair
           join public.clips c on c.id in (h.a, h.b)
           where i.user_id = auth.uid() and c.audio_path = p_object);
$$;

-- exports carry the spans
drop function public.admin_export_labels(timestamptz);
create function public.admin_export_labels(p_since timestamptz default '-infinity')
returns table (label_id bigint, source_uid text, kind text, labeller text, username text, labels text[], other_text text,
               unsure boolean, listen_ms integer, attempt smallint, trust real, quality text, spans jsonb,
               created_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select l.id, s.source_uid, s.kind, coalesce(p.fusion_name, 'crowd:' || left(p.id::text, 8)), p.username, l.labels,
         l.other_text, l.unsure, l.listen_ms, l.attempt, p.trust, coalesce(q.status, 'too early'), l.spans, l.created_at
  from public.labels l
  join private.clip_sources s on s.clip_id = l.clip_id
  join public.profiles p on p.id = l.user_id
  left join private.labeller_quality q on q.user_id = l.user_id
  where l.created_at >= p_since
  order by l.id;
$$;
revoke all on function public.admin_export_labels(timestamptz) from public, anon, authenticated;
grant execute on function public.admin_export_labels(timestamptz) to service_role;


-- serving: preferences first, then the gold set, then the usual order
create or replace function public.next_clip()
returns table (clip_id uuid, audio_path text, duration_s real, expires_at timestamptz)
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
  pick public.clips;
  got boolean := false;
  att smallint := 1;
  n integer;
  r double precision := random();
  rep numeric := private.setting('repeat_rate', 0.04);
  gap integer;
  breadth integer := private.setting('breadth_votes', 3)::integer;
  gold_votes integer := private.setting('gold_votes', 12)::integer;
  ctrl_rate numeric;
  thin_only boolean;
  audited boolean;
  spref text;
  cpref text;
  vpref text;
  excl text;                                   -- the voice this person asked not to hear
  male_passes boolean[];
  pass_male boolean[] := '{}';
  pass_sound text[] := '{}';
  sk text;
  mk boolean;
  i integer;
  male_only boolean;
  sound text;
begin
  if uid is null then raise exception 'sign in first' using errcode = '28000'; end if;
  if not exists (select 1 from public.profiles p where p.id = uid and p.deleted_at is null) then
    raise exception 'finish your profile first' using errcode = 'P0001';
  end if;
  return query
    select a.clip_id, c.audio_path, c.duration_s, a.expires_at
    from public.assignments a join public.clips c on c.id = a.clip_id
    where a.user_id = uid and a.state = 'open' and a.expires_at > now() and c.active
    order by a.assigned_at
    limit 1;
  if found then return; end if;

  select p.label_seq, p.sound_pref, p.content_pref, p.voice_pref into n, spref, cpref, vpref
  from public.profiles p where p.id = uid;
  excl := case when cpref = 'sfw' then null when vpref = 'male' then 'female' when vpref = 'female' then 'male' end;
  audited := exists (select 1 from private.labeller_quality q where q.user_id = uid and q.status in ('watch', 'weak'));
  gap := least(private.setting('repeat_gap', 300)::integer,
               greatest(private.setting('repeat_gap_min', 30)::integer, n / 2));
  if audited then
    rep := private.setting('audit_repeat_rate', 0.12);
    gap := private.setting('audit_repeat_gap', 15)::integer;
  end if;

  -- a hidden retest: something they answered at least `gap` labels ago, not yet retested
  if n >= gap and r < rep then
    select c.* into pick
    from public.labels l join public.clips c on c.id = l.clip_id
    where l.user_id = uid and l.attempt = 1 and not l.unsure and l.seq <= n - gap and c.active
      and private.clip_allowed(c.voice, c.nsfw, cpref, excl)
      and not exists (select 1 from public.assignments m where m.user_id = uid and m.clip_id = c.id and m.attempt = 2)
    order by random()
    limit 1
    for update of c skip locked;
    if found then got := true; att := 2; end if;
  end if;

  -- a control: often while we know little about this person (or they are on watch), rarely once they have answered many
  if not got then
    select greatest(private.setting('control_rate_min', 0.05),
                    private.setting('control_rate_max', 0.25) - 0.01 * count(*)) into ctrl_rate
    from public.labels l join private.clip_sources s on s.clip_id = l.clip_id
    where l.user_id = uid and s.kind = 'control';
    if audited then ctrl_rate := greatest(ctrl_rate, private.setting('audit_control_rate', 0.3)); end if;
    if r >= rep and r < rep + ctrl_rate then
      select c.* into pick
      from public.clips c join private.clip_sources s on s.clip_id = c.id
      where c.active and s.kind = 'control' and private.clip_allowed(c.voice, c.nsfw, cpref, excl)
        and not exists (select 1 from public.assignments m where m.user_id = uid and m.clip_id = c.id)
      order by c.need desc, random()
      limit 1
      for update of c skip locked;
      if found then got := true; end if;
    end if;
  end if;

  -- the passes: the kind of sound they prefer first, then any; within each, male voices first when the draw (or the
  -- person's preference) says so
  male_passes := case when cpref = 'sfw' or vpref = 'female' then array[false]
                      when vpref = 'male' then array[true, false]
                      when random() < private.setting('male_share', 0.3) then array[true, false]
                      else array[false] end;
  foreach sk in array (case when spref in ('speech', 'nonspeech') then array[spref, 'any'] else array['any'] end) loop
    foreach mk in array (case when sk = 'nonspeech' then array[false] else male_passes end) loop
      pass_male := pass_male || mk;
      pass_sound := pass_sound || sk;
    end loop;
  end loop;

  for i in 1 .. cardinality(pass_male) loop
    exit when got;
    male_only := pass_male[i];
    sound := pass_sound[i];

    -- the gold set: clips that need `gold_votes` listeners each before the timeline labeller opens, fullest first
    select c.* into pick from public.clips c
    where c.active and c.gold and c.votes < gold_votes
      and (c.voice = 'male' or not male_only) and (sound = 'any' or (sound = 'speech') = (c.voice is not null))
      and private.clip_allowed(c.voice, c.nsfw, cpref, excl)
      and not exists (select 1 from public.assignments m where m.user_id = uid and m.clip_id = c.id)
    order by c.votes desc, c.need desc, random()
    limit 1
    for update of c skip locked;
    if found then got := true; exit; end if;

    foreach thin_only in array array[true, false] loop
      exit when got;

      -- started clips up to `breadth` judges, fullest first
      select c.* into pick from public.clips c
      where c.active and (c.boost > 0 or not thin_only) and c.votes > 0 and c.votes < least(c.target_votes, breadth)
        and (c.voice = 'male' or not male_only) and (sound = 'any' or (sound = 'speech') = (c.voice is not null))
        and private.clip_allowed(c.voice, c.nsfw, cpref, excl)
        and not exists (select 1 from public.assignments m where m.user_id = uid and m.clip_id = c.id)
      order by c.votes desc, c.need + c.boost desc, random()
      limit 1
      for update of c skip locked;
      if found then got := true; exit; end if;

      -- the neediest clip they have never been given (fresh clips rank first; started ones keep filling to target)
      with candidates as (
        select c.id, c.need + c.boost as score from public.clips c
        where c.active and (c.boost > 0 or not thin_only)
          and (c.voice = 'male' or not male_only) and (sound = 'any' or (sound = 'speech') = (c.voice is not null))
          and private.clip_allowed(c.voice, c.nsfw, cpref, excl)
          and not exists (select 1 from public.assignments m where m.user_id = uid and m.clip_id = c.id)
        order by c.need + c.boost desc
        limit 300
      )
      select c.* into pick
      from candidates k join public.clips c on c.id = k.id
      order by k.score
               - 0.15 * (select count(*) from public.assignments o
                         where o.clip_id = k.id and o.state = 'open' and o.expires_at > now())
               + 0.05 * random() desc
      limit 1
      for update of c skip locked;
      if found then got := true; end if;
    end loop;
  end loop;
  if not got then return; end if;

  insert into public.assignments (user_id, clip_id, attempt) values (uid, pick.id, att);
  return query select pick.id, pick.audio_path, pick.duration_s, now() + interval '20 minutes';
end $$;

-- breaks: no "Which is hotter?" for SFW labellers, and never a voice they asked not to hear
create or replace function public.next_interlude()
 RETURNS TABLE(kind text, audio_a text, audio_b text)
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO ''
AS $$
declare
  uid uuid := auth.uid();
  seq integer;
  nxt integer;
  it private.interludes;
  item uuid;
  pair uuid;
  two uuid[];
  snd text;
  k text;
  want_ai boolean := random() < 0.5;
  cpref text;
  excl text;
begin
  if uid is null then raise exception 'sign in first' using errcode = '28000'; end if;
  select p.label_seq, p.interlude_next_seq, p.content_pref,
         case when p.content_pref = 'sfw' then null when p.voice_pref = 'male' then 'female' when p.voice_pref = 'female' then 'male' end
  into seq, nxt, cpref, excl
  from public.profiles p where p.id = uid and p.deleted_at is null
  for update;
  if not found then return; end if;

  select * into it from private.interludes i where i.user_id = uid;
  if not found then
    if nxt is null then
      update public.profiles p set interlude_next_seq = seq + private.interlude_gap() where p.id = uid;
      return;
    end if;
    if seq < nxt then return; end if;

    foreach k in array (case when random() < 0.5 then array['rai', 'hot'] else array['hot', 'rai'] end) loop
      continue when k = 'hot' and cpref = 'sfw';             -- "Which is hotter?" is adult content
      if k = 'rai' then
        -- an item they have not had: real or AI with even odds, the least-answered first
        select r.id into item from private.rai_items r
        where r.active and not exists (select 1 from private.rai_answers x where x.user_id = uid and x.item_id = r.id)
        order by (r.is_ai = want_ai) desc, (select count(*) from private.rai_answers x where x.item_id = r.id), random()
        limit 1;
        if item is not null then
          insert into private.interludes (user_id, kind, rai_item) values (uid, 'rai', item) returning * into it;
          exit;
        end if;
      else
        -- half the time a pair others have started judging (so pairs collect several votes), else a new pair
        if random() < 0.5 then
          select h.id into pair from private.hot_pairs h
          join public.clips c1 on c1.id = h.a join public.clips c2 on c2.id = h.b
          where c1.active and c2.active and h.votes < private.setting('hot_pair_votes', 5)
            and private.clip_allowed(c1.voice, true, null, excl) and private.clip_allowed(c2.voice, true, null, excl)
            and not exists (select 1 from private.hot_votes v where v.user_id = uid and v.pair_id = h.id)
          order by h.votes desc, random()
          limit 1;
        end if;
        if pair is null then
          foreach snd in array (select array_agg(x order by random())
                                from unnest(array['moaning', 'whispering', 'breathing', 'kissing', 'oral sounds']) x) loop
            select array_agg(y.clip_id) into two from (
              select s.clip_id from private.clip_sources s join public.clips c on c.id = s.clip_id
              where c.active and s.kind = 'confident' and s.hot_sound = snd and private.clip_allowed(c.voice, true, null, excl)
                and s.hot_p >= private.setting('hot_min_p', 0.9)
              order by random()
              limit 2) y;
            if coalesce(array_length(two, 1), 0) = 2 then
              insert into private.hot_pairs (a, b, sound) values (least(two[1], two[2]), greatest(two[1], two[2]), snd)
              on conflict (a, b) do update set sound = excluded.sound
              returning id into pair;
              -- an existing pair they already judged: try another sound
              if exists (select 1 from private.hot_votes v where v.user_id = uid and v.pair_id = pair) then
                pair := null;
                continue;
              end if;
              exit;
            end if;
          end loop;
        end if;
        if pair is not null then
          insert into private.interludes (user_id, kind, hot_pair, flip) values (uid, 'hot', pair, random() < 0.5)
          returning * into it;
          exit;
        end if;
      end if;
    end loop;

    if it.user_id is null then  -- nothing to offer: try again after another gap
      update public.profiles p set interlude_next_seq = seq + private.interlude_gap() where p.id = uid;
      return;
    end if;
  end if;

  if it.kind = 'rai' then
    return query select it.kind, r.audio_path, null::text from private.rai_items r where r.id = it.rai_item;
  else
    return query
      select it.kind, case when it.flip then c2.audio_path else c1.audio_path end,
                      case when it.flip then c1.audio_path else c2.audio_path end
      from private.hot_pairs h join public.clips c1 on c1.id = h.a join public.clips c2 on c2.id = h.b
      where h.id = it.hot_pair;
  end if;
end $$;
