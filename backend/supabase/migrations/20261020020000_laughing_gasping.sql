-- Two labels from what labellers typed under "something else", and wider hints so the rest finds an existing label.
--
-- Of the 37 free-text answers so far (28 on the site, 9 in the Ear Check kits), 12 were laughing / giggling / laughter
-- from 5 different people: a third of all free text, so it gets a label. Gasping was typed once and asked for by
-- labellers (short, cut-off moans have no label of their own). The rest fit existing labels once their hints say so:
-- fingers dragged on a wall and "writing something" (scratching), hook-and-loop and slime (sticky), rain (liquid),
-- slurping and sucking (oral sounds). Spanking, clapping and sex sounds were typed before those labels existed.
-- Both new labels join the thin classes; clips whose titles promise them (title_tags) get the new-label boost.

insert into public.label_options (key, grp, hint, sort) values
  ('laughing', 'Voice', 'laughs, giggles or chuckles', 32),
  ('gasping', 'Voice', 'a sharp catch of breath, often voiced: quick gasps and short, cut-off moans', 33);

update public.label_options set hint = 'nails or fingers dragged over a surface, or a pen writing: continuous rough scraping'
  where key = 'scratching';
update public.label_options set hint = 'tacky surfaces pulling apart: tape, slime, velcro, sticky fingers' where key = 'sticky';
update public.label_options set hint = 'water or gel: pouring, dripping, sloshing, bubbles, rain' where key = 'liquid';
update public.label_options set hint = 'licking, sucking, slurping, lip smacks, wet mouth sounds (not kisses)' where key = 'oral sounds';

insert into private.thin_classes (label) values ('laughing'), ('gasping') on conflict do nothing;
insert into private.new_labels (label) values ('laughing'), ('gasping') on conflict do nothing;
