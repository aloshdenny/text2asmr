-- Pairing for Which-is-hotter read every confident clip's fused JSON to find its main vocal sound (~1.3 s on 8k
-- clips, while the labeller waits). Each clip now keeps that sound and its P in stored columns, indexed for the
-- confident clips that can be paired.

alter table private.clip_sources
  add column hot_sound text generated always as (private.hot_sound(info)) stored,
  add column hot_p real generated always as (((info -> 'fused') ->> private.hot_sound(info))::real) stored;
create index clip_sources_hot on private.clip_sources (hot_sound, hot_p) where kind = 'confident';

create or replace function public.next_interlude()
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
              where c.active and s.kind = 'confident' and s.hot_sound = snd
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
