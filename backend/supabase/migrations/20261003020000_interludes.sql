-- Interludes: a short break from labelling every 8-17 labels (uniform), one of two games picked at random:
--
--   Real or AI?       One clip: a real recording, or T2A's clone of the same voice saying another line? The answer
--                     comes after the guess, with how many listeners the clip fooled. Real voices are speakers who
--                     recorded for speech-synthesis research (scripts/rai_items.py); every voice appears both real
--                     and cloned, so recognising a voice gives nothing away. No creator's voice is ever cloned.
--   Which is hotter?  Two clips whose labels we are confident about (fused P >= hot_min_p) and that share their main
--                     vocal sound. Each pair is judged by several people; the votes are preference data for the
--                     generator. No clip shows who made it, and no ranking is published.
--
-- Items, pairs and answers live in `private`. The browser gets the open interlude's kind and audio paths (opaque
-- object names, like every clip) -- never the answer before it guesses, never anyone else's answers -- and storage
-- serves exactly those objects. Answers go to the caller's one open interlude, so no item id ever leaves the server.

insert into private.settings (key, value) values
  ('interlude_gap_min', 8), ('interlude_gap_max', 17),  -- labels between interludes
  ('hot_min_p', 0.9),                                   -- fused P a clip's main vocal sound needs for it to be paired
  ('hot_pair_votes', 5),                                -- judgements a pair collects before new pairs take over
  ('rai_min_answers', 5)                                -- other answers needed before "fooled x%" is shown
on conflict (key) do nothing;

alter table public.profiles add column interlude_next_seq integer;  -- label_seq at which the next interlude is due

create table private.rai_items (
  id uuid primary key default gen_random_uuid(),
  audio_path text not null unique,                      -- object in the private 'clips' bucket, named like any clip
  is_ai boolean not null,
  voice text not null,                                  -- speaker: each voice has both real and AI items
  model text,                                           -- generator of an AI item
  source text,                                          -- dataset utterance id
  active boolean not null default true,
  created_at timestamptz not null default now()
);

create table private.rai_answers (
  user_id uuid not null references public.profiles (id) on update cascade on delete cascade,
  item_id uuid not null references private.rai_items (id) on delete cascade,
  guess_ai boolean not null,
  correct boolean not null,
  listen_ms integer,
  created_at timestamptz not null default now(),
  primary key (user_id, item_id)
);

create table private.hot_pairs (
  id uuid primary key default gen_random_uuid(),
  a uuid not null references public.clips (id) on delete cascade,
  b uuid not null references public.clips (id) on delete cascade,
  sound text not null,
  votes integer not null default 0,
  a_wins integer not null default 0,
  created_at timestamptz not null default now(),
  check (a < b),
  unique (a, b)
);

create table private.hot_votes (
  user_id uuid not null references public.profiles (id) on update cascade on delete cascade,
  pair_id uuid not null references private.hot_pairs (id) on delete cascade,
  winner uuid not null,
  listen_a_ms integer,
  listen_b_ms integer,
  created_at timestamptz not null default now(),
  primary key (user_id, pair_id)
);

-- the one open interlude per person (it comes back on reload until answered or skipped)
create table private.interludes (
  user_id uuid primary key references public.profiles (id) on update cascade on delete cascade,
  kind text not null check (kind in ('rai', 'hot')),
  rai_item uuid references private.rai_items (id) on delete cascade,
  hot_pair uuid references private.hot_pairs (id) on delete cascade,
  flip boolean not null default false,                  -- hot: the pair's b clip is shown first
  served_at timestamptz not null default now(),
  check ((kind = 'rai') = (rai_item is not null) and (kind = 'hot') = (hot_pair is not null))
);

revoke all on private.rai_items, private.rai_answers, private.hot_pairs, private.hot_votes, private.interludes
  from public, anon, authenticated;

create function private.interlude_gap()
returns integer
language sql volatile security definer set search_path = '' as $$
  select (lo + floor(random() * (hi - lo + 1)))::integer
  from (select private.setting('interlude_gap_min', 8) as lo, greatest(private.setting('interlude_gap_min', 8),
                                                                       private.setting('interlude_gap_max', 17)) as hi) s;
$$;

-- a clip's main vocal sound, from its fused probabilities
create function private.hot_sound(p_info jsonb)
returns text
language sql immutable set search_path = '' as $$
  select e.k from jsonb_each_text(p_info -> 'fused') e (k, v)
  where e.k in ('moaning', 'whispering', 'breathing', 'kissing', 'oral sounds')
  order by e.v::numeric desc
  limit 1;
$$;

-- storage: the open assignment's clip, as before, plus the open interlude's audio
create or replace function private.can_hear(p_object text)
returns boolean
language sql stable security definer set search_path = '' as $$
  select exists (
           select 1 from public.assignments a join public.clips c on c.id = a.clip_id
           where a.user_id = auth.uid() and a.state = 'open' and c.audio_path = p_object)
      or exists (
           select 1 from private.interludes i join private.rai_items r on r.id = i.rai_item
           where i.user_id = auth.uid() and r.audio_path = p_object)
      or exists (
           select 1 from private.interludes i join private.hot_pairs h on h.id = i.hot_pair
           join public.clips c on c.id in (h.a, h.b)
           where i.user_id = auth.uid() and c.audio_path = p_object);
$$;

-- The caller's interlude, if one is due: kind 'rai' (audio_a) or 'hot' (audio_a, audio_b, in the order to show).
-- Nothing when none is due. The first call for a person only schedules their first interlude.
create function public.next_interlude()
returns table (kind text, audio_a text, audio_b text)
language plpgsql security definer set search_path = '' as $$
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
begin
  if uid is null then raise exception 'sign in first' using errcode = '28000'; end if;
  select p.label_seq, p.interlude_next_seq into seq, nxt
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
            and not exists (select 1 from private.hot_votes v where v.user_id = uid and v.pair_id = h.id)
          order by h.votes desc, random()
          limit 1;
        end if;
        if pair is null then
          foreach snd in array (select array_agg(x order by random())
                                from unnest(array['moaning', 'whispering', 'breathing', 'kissing', 'oral sounds']) x) loop
            select array_agg(y.clip_id) into two from (
              select s.clip_id from private.clip_sources s join public.clips c on c.id = s.clip_id
              where c.active and s.kind = 'confident'
                and (s.info -> 'fused' ->> snd)::numeric >= private.setting('hot_min_p', 0.9)
                and private.hot_sound(s.info) = snd
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

-- Answer the open Real-or-AI interlude. Returns whether the guess was right, the truth, how many other listeners
-- the clip fooled (once rai_min_answers others answered), and the caller's running score.
create function public.answer_rai(p_guess_ai boolean, p_listen_ms integer)
returns table (correct boolean, was_ai boolean, fooled_pct integer, others integer, my_correct integer, my_total integer)
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
  it private.interludes;
  r private.rai_items;
  ok boolean;
begin
  if uid is null then raise exception 'sign in first' using errcode = '28000'; end if;
  if p_guess_ai is null then raise exception 'pick real or AI' using errcode = '22023'; end if;
  select * into it from private.interludes i where i.user_id = uid and i.kind = 'rai' for update;
  if not found then raise exception 'nothing to answer' using errcode = 'P0001'; end if;
  if now() - it.served_at < interval '2 seconds' or coalesce(p_listen_ms, 0) < 1500 then
    raise exception 'listen to the clip first' using errcode = '22023';
  end if;
  select * into r from private.rai_items x where x.id = it.rai_item;
  ok := p_guess_ai = r.is_ai;
  insert into private.rai_answers (user_id, item_id, guess_ai, correct, listen_ms)
  values (uid, r.id, p_guess_ai, ok, p_listen_ms)
  on conflict do nothing;
  delete from private.interludes i where i.user_id = uid;
  update public.profiles p set interlude_next_seq = p.label_seq + private.interlude_gap() where p.id = uid;
  return query
    with o as (select x.correct as c from private.rai_answers x where x.item_id = r.id and x.user_id <> uid),
         m as (select x.correct as c from private.rai_answers x where x.user_id = uid)
    select ok, r.is_ai,
           case when (select count(*) from o) >= private.setting('rai_min_answers', 5)
                then round(100.0 * (select count(*) from o where not o.c) / (select count(*) from o))::integer end,
           (select count(*) from o)::integer,
           (select count(*) from m where m.c)::integer,
           (select count(*) from m)::integer;
end $$;

-- Answer the open Which-is-hotter interlude with 'a' or 'b' (as shown). Returns the share of earlier judges of this
-- pair who picked the same clip (once 3 have judged it) and how many judged it before.
create function public.answer_hot(p_pick text, p_listen_a_ms integer, p_listen_b_ms integer)
returns table (agree_pct integer, judges integer)
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
  it private.interludes;
  h private.hot_pairs;
  win uuid;
  same integer;
begin
  if uid is null then raise exception 'sign in first' using errcode = '28000'; end if;
  if p_pick is null or p_pick not in ('a', 'b') then raise exception 'pick a or b' using errcode = '22023'; end if;
  select * into it from private.interludes i where i.user_id = uid and i.kind = 'hot' for update;
  if not found then raise exception 'nothing to answer' using errcode = 'P0001'; end if;
  if now() - it.served_at < interval '3 seconds' or coalesce(p_listen_a_ms, 0) < 1000 or coalesce(p_listen_b_ms, 0) < 1000 then
    raise exception 'listen to both clips first' using errcode = '22023';
  end if;
  select * into h from private.hot_pairs x where x.id = it.hot_pair for update;
  win := case when (p_pick = 'a') = not it.flip then h.a else h.b end;   -- shown 'a' is the pair's a unless flipped
  same := case when win = h.a then h.a_wins else h.votes - h.a_wins end;
  insert into private.hot_votes (user_id, pair_id, winner, listen_a_ms, listen_b_ms)
  values (uid, h.id, win, p_listen_a_ms, p_listen_b_ms)
  on conflict do nothing;
  if found then
    update private.hot_pairs x set votes = x.votes + 1, a_wins = x.a_wins + (win = h.a)::integer where x.id = h.id;
  end if;
  delete from private.interludes i where i.user_id = uid;
  update public.profiles p set interlude_next_seq = p.label_seq + private.interlude_gap() where p.id = uid;
  return query select case when h.votes >= 3 then round(100.0 * same / h.votes)::integer end, h.votes;
end $$;

-- Pass on the open interlude: the next one comes after another gap.
create function public.skip_interlude()
returns void
language plpgsql security definer set search_path = '' as $$
begin
  if auth.uid() is null then raise exception 'sign in first' using errcode = '28000'; end if;
  delete from private.interludes i where i.user_id = auth.uid();
  update public.profiles p set interlude_next_seq = p.label_seq + private.interlude_gap() where p.id = auth.uid();
end $$;

-- [{audio_path, is_ai, voice, model, source}] -> items added (rai_items.py + sync/push_rai.py)
create function public.admin_add_rai_items(p_items jsonb)
returns integer
language plpgsql security definer set search_path = '' as $$
declare n integer;
begin
  perform private.require_service_role();
  insert into private.rai_items (audio_path, is_ai, voice, model, source)
  select x ->> 'audio_path', (x ->> 'is_ai')::boolean, x ->> 'voice', x ->> 'model', x ->> 'source'
  from jsonb_array_elements(p_items) x
  on conflict (audio_path) do nothing;
  get diagnostics n = row_count;
  return n;
end $$;

-- Every Real-or-AI answer (the generator's "how human does it sound" score) ...
create function public.admin_export_rai()
returns table (source text, voice text, is_ai boolean, model text, labeller text, guess_ai boolean, correct boolean,
               listen_ms integer, trust real, created_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select r.source, r.voice, r.is_ai, r.model, coalesce(p.fusion_name, 'crowd:' || left(p.id::text, 8)), x.guess_ai, x.correct,
         x.listen_ms, p.trust, x.created_at
  from private.rai_answers x join private.rai_items r on r.id = x.item_id join public.profiles p on p.id = x.user_id
  order by x.created_at;
$$;

-- ... and every Which-is-hotter vote (pairwise preferences between clips, by their pool uids)
create function public.admin_export_hot()
returns table (sound text, a_uid text, b_uid text, winner_uid text, labeller text, listen_a_ms integer, listen_b_ms integer,
               trust real, created_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select h.sound, sa.source_uid, sb.source_uid, case when v.winner = h.a then sa.source_uid else sb.source_uid end,
         coalesce(p.fusion_name, 'crowd:' || left(p.id::text, 8)), v.listen_a_ms, v.listen_b_ms, p.trust, v.created_at
  from private.hot_votes v join private.hot_pairs h on h.id = v.pair_id
  join private.clip_sources sa on sa.clip_id = h.a join private.clip_sources sb on sb.clip_id = h.b
  join public.profiles p on p.id = v.user_id
  order by v.created_at;
$$;

revoke all on function private.interlude_gap(), private.hot_sound(jsonb) from public;
revoke all on function public.next_interlude(), public.answer_rai(boolean, integer), public.answer_hot(text, integer, integer),
  public.skip_interlude(), public.admin_add_rai_items(jsonb), public.admin_export_rai(), public.admin_export_hot() from public;
grant execute on function public.next_interlude(), public.answer_rai(boolean, integer), public.answer_hot(text, integer, integer),
  public.skip_interlude() to authenticated;
grant execute on function public.admin_add_rai_items(jsonb), public.admin_export_rai(), public.admin_export_hot() to service_role;
