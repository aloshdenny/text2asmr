-- Exports skip blocked clips. A blocked clip (its title puts minors, school / teen settings, age play or incest in a
-- sexual scene) must never be trained on, but the label export and the "Which is hotter?" export still carried answers
-- given before the clip was blocked. Both leave them out now.
--
-- The label export also drops answers that say nothing: a bare "something else" with no text is taken out of an
-- answer (an answer left with no labels is skipped), and an answer marked "can't tell" that also names sounds is
-- skipped as contradictory. The rows themselves stay in public.labels.

create or replace function public.admin_export_labels(p_since timestamptz default '-infinity')
returns table (label_id bigint, source_uid text, kind text, labeller text, username text, labels text[], other_text text,
               unsure boolean, listen_ms integer, attempt smallint, trust real, quality text, spans jsonb,
               created_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select l.id, s.source_uid, s.kind, coalesce(p.fusion_name, 'crowd:' || left(p.id::text, 8)), p.username, x.labels,
         l.other_text, l.unsure, l.listen_ms, l.attempt, p.trust, coalesce(q.status, 'too early'), l.spans, l.created_at
  from public.labels l
  join public.clips c on c.id = l.clip_id
  join private.clip_sources s on s.clip_id = l.clip_id
  join public.profiles p on p.id = l.user_id
  left join private.labeller_quality q on q.user_id = l.user_id
  cross join lateral (
    select case when coalesce(btrim(l.other_text), '') = '' then array_remove(l.labels, 'something else')
                else l.labels end as labels
  ) x
  where l.created_at >= p_since
    and not c.blocked
    and not (l.unsure and cardinality(l.labels) > 0)
    and (l.unsure or cardinality(x.labels) > 0)
  order by l.id;
$$;

create or replace function public.admin_export_hot()
returns table (sound text, a_uid text, b_uid text, winner_uid text, labeller text, listen_a_ms integer, listen_b_ms integer,
               trust real, created_at timestamptz)
language sql stable security definer set search_path = '' as $$
  select private.require_service_role();
  select h.sound, sa.source_uid, sb.source_uid, case when v.winner = h.a then sa.source_uid else sb.source_uid end,
         coalesce(p.fusion_name, 'crowd:' || left(p.id::text, 8)), v.listen_a_ms, v.listen_b_ms, p.trust, v.created_at
  from private.hot_votes v join private.hot_pairs h on h.id = v.pair_id
  join public.clips ca on ca.id = h.a join public.clips cb on cb.id = h.b
  join private.clip_sources sa on sa.clip_id = h.a join private.clip_sources sb on sb.clip_id = h.b
  join public.profiles p on p.id = v.user_id
  where not ca.blocked and not cb.blocked
  order by v.created_at;
$$;
