-- Calibrate the weak-labeller thresholds on the first real run (2026-10-18). On live data no one's answers came out
-- independent of the audio: follow ranged 0.12-0.34 for everyone with 15+ referenced answers. References here are
-- noisy (people often disagree about these sounds, and most clips have only two other listeners), so even a careful
-- listener cannot reach 1, and the 0.25 watch line put four genuine labellers, the most active among them, on watch.
-- Watch now means "close to chance" (follow under 0.1); weak still needs the evidence to place it under 0.1.

insert into private.settings (key, value) values ('quality_watch_follow', 0.1), ('quality_weak_follow', 0.1)
on conflict (key) do update set value = excluded.value;

select private.refresh_quality();
