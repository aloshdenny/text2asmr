-- How much of the corpus is labelled, for the two rings on the home page: by people (site + Ear Check kits, counted
-- live) and by the CLAP-ASMR model. Corpus size and the model's coverage are set by the pipeline, not computed here.

create table private.settings (
  key text primary key,
  value numeric not null,
  updated_at timestamptz not null default now()
);
alter table private.settings enable row level security;

insert into private.settings (key, value) values ('corpus_items', 0), ('ai_labelled_items', 0);

create function public.dataset_progress()
returns table (corpus_items bigint, human_labelled_items bigint, ai_labelled_items bigint)
language sql stable security definer set search_path = '' as $$
  select (select value::bigint from private.settings where key = 'corpus_items'),
         (select count(*) from (
            select s.source_uid from public.labels l join private.clip_sources s on s.clip_id = l.clip_id where not l.unsure
            union
            select i.source_uid from public.imported_labels i) u),
         (select value::bigint from private.settings where key = 'ai_labelled_items');
$$;

create function public.admin_set_setting(p_key text, p_value numeric)
returns void
language plpgsql security definer set search_path = '' as $$
begin
  perform private.require_service_role();
  insert into private.settings (key, value) values (p_key, p_value)
  on conflict (key) do update set value = excluded.value, updated_at = now();
end $$;
