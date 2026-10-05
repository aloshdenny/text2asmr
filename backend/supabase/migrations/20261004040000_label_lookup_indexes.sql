-- Labels were only indexed by person, so every "does this clip have a label?" check -- the rings' recount over 21k
-- core clips (10.6 s, past the API's statement timeout) and the consensus update on each submitted label -- scanned
-- the whole table. Index them by clip, and imported (kit / seed) labels by source uid.

create index if not exists labels_by_clip on public.labels (clip_id) where not unsure;
create index if not exists labels_by_clip_attempt on public.labels (clip_id, attempt);
create index if not exists imported_by_source on public.imported_labels (source_uid);
