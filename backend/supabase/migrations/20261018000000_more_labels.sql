-- More labels: clapping, and an NSFW group.
--
-- The NSFW sounds were picked from what creators actually tag: the titles of the 71,353 named recordings in
-- aoxo/t2a-mommy and aoxo/t2a-daddy, counting sound events only (acts and genres are left out). Tag counts:
-- vibrator / toy 1,418, whimpering 1,229, wet sounds 1,138, spanking 1,109, growling 700, slapping / "clapping" 698,
-- gagging 339, squirting 248, bed creaking 137; grunting is rarely tagged but runs through the male corpus.
-- Kissing, oral sounds, moaning and breathing were on the menu already and stay in Voice.
--
-- A label can now show a display name that differs from its key: the key is what labels store and what the models
-- learn ("sex slapping" means something to a text encoder), the title is what the chip says ("clapping😉").
--
-- added_at records when each label joined the menu, so an exported answer knows which labels it was offered: an answer
-- given before squirting existed abstains on squirting instead of counting as "no squirting".

alter table public.label_options drop constraint label_options_grp_check;
alter table public.label_options add constraint label_options_grp_check check (grp in ('Voice', 'Triggers', 'NSFW', 'Other'));
alter table public.label_options add column title text;
alter table public.label_options add column added_at timestamptz not null default now();
update public.label_options set added_at = '2026-10-02 00:00:00+00';

insert into public.label_options (key, grp, hint, sort, title) values
  ('clapping', 'Triggers', 'hands clapping: single claps or applause', 21, null),
  ('sex slapping', 'NSFW', 'bodies slapping together during sex: fast, rhythmic skin smacks', 22, 'clapping😉'),
  ('wet sounds', 'NSFW', 'wet squelching from fingering, stroking or sex (not mouth sounds)', 23, null),
  ('squirting', 'NSFW', 'a sudden gush or spray of fluid, often with a splash', 24, null),
  ('spanking', 'NSFW', 'a hand or paddle smacking skin: single sharp slaps', 25, null),
  ('vibrator', 'NSFW', 'a sex toy''s motor: steady buzzing or humming', 26, null),
  ('gagging', 'NSFW', 'throat gags and sputters, usually during oral', 27, null),
  ('whimpering', 'NSFW', 'soft, high, broken whines', 28, null),
  ('growling', 'NSFW', 'low, rough rumbling from the throat', 29, null),
  ('grunting', 'NSFW', 'short, low grunts of effort or pleasure', 30, null),
  ('bed creaking', 'NSFW', 'a bed or other furniture creaking or squeaking in rhythm', 31, null);
