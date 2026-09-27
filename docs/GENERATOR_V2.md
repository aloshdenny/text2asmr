# Generator v2: context windows sized to the data

v1 trained on `t2a-audios-v1` only and its trigger LoRA produced noise (labels were zero-shot CLAP tags).
v2 trains on the transcribed corpora (`t2a-mommy`, `t2a-daddy`), labels from Qwen3-Omni + CLAP v7, and has
one rule above all: **never ask the model to produce something longer, denser or of a kind it has not seen
thousands of times.** That is what "does not hallucinate" means here, and every window below is derived from
measured data, not chosen.

## 1. What the data is (sampled 800 alignment files, 2026-09-27)

| | mommy | daddy |
|---|---|---|
| recordings | 64,373 | 55,835 |
| median recording | 13.3 min | 12.3 min |
| speech fraction | 49.8% | 51.0% |
| phrase length p50 / p90 / p95 / p99 (s) | 3.1 / 9.8 / 14.3 / 19.8 | 2.8 / 8.6 / 12.3 / 19.7 |
| words per phrase p50 / p90 / p95 / p99 | 8 / 24 / 34 / 54 | 8 / 23 / 32 / 53 |
| speaking rate (words/s, median) | 2.50 | 2.69 |
| non-speech gaps >= 1.5 s, per recording | ~75 | ~75 |

Totals: ~29k hours of recordings, ~14k hours of speech (DeepASMR: 670 h). Phrases are split at pauses
> 0.7 s (`segment.py`). Labels: whispering 6.6M, normal speech 1.5M, breathing 1.15M, oral sounds ~1.2M,
moaning 0.47M clips. ASMR speech is **slow and gappy**: half of every recording is not speech.

## 2. Speech model (text -> S3 speech tokens, Chatterbox T3 lineage)

| window | size | why |
|---|---|---|
| **output per generation** | **<= 20 s** (500 S3 tokens @ 25 Hz) | p99 phrase is 19.8 s. Training targets are consecutive phrases *packed with their real pauses* up to 20 s, sampled so 2-20 s is covered roughly uniformly. Autoregressive TTS fails first by length: past the lengths it trained on, attention alignment slips and it repeats, skips or babbles. 20 s keeps every inference inside a length the model has seen hundreds of thousands of times. |
| **text input** | **<= 128 tokens (~55 words + tags)** | 20 s x 2.6 words/s = 52 words; p95 phrase is 34 words. The inference planner splits a script at sentence boundaries so each chunk's *predicted* duration (words / 2.6 + tagged pauses + tagged events) is <= 18 s, leaving headroom. |
| **continuation prompt** | **previous 4 s of generated speech tokens (100)** + a 6-10 s speaker reference | Long-form ASMR is minutes long. v2 does not generate minutes in one pass; it chains 20 s windows, each conditioned on the tail of the last. Training includes this: prompt = the 4 s before the target window *in the same recording*. The model learns continuation from real continuations, which is what prevents drift across chunks (§5 measures it). |

**Non-speech inside the speech stream is explicit, never implicit.** The target text carries inline tags for
what happens in the window's gaps, taken from the Qwen3/v7 labels of those gaps:

    Hey... [pause 1.8s] you look tired. [breathing] Come here, [oral sounds] let me take care of you.

The model is never left to decide on its own that a 2-second silence or a breath should happen. Every pause
and vocal event it produces was asked for, and every one it was trained on was in its text. That is the
single biggest lever against content hallucination in a corpus that is 50% non-speech.

Vocal events (breathing, oral sounds, moaning) live here because they come from the same speaker in the
same room. v7 verifies them at 0.75-0.85.

## 3. Physical trigger model (caption -> audio, flow matching on latents)

| window | size | why |
|---|---|---|
| output | **8 s** per generation, crossfaded for longer beds | verified trigger windows are 4 s (what v7 scores); single-trigger videos give long continuous takes, so 8 s windows are plentiful. Longer beds come from chaining, as with speech. |
| text input | **<= 32 tokens**: class + intensity (+ a few modifiers) | captions are short and closed-vocabulary; a free-text prompt would let users request things no data backs. |

**Allowed classes = the ones v7 can verify at >= 0.5 on held-out data** (currently tapping, scratching,
crinkling, paper rustling). The rest (microphone touching, sticky, fabric, cutting, and the six classes the
single-trigger downloads are filling) are *not* accepted as tags until v7.1 verifies them. A tag we cannot
measure is a tag we cannot claim to honour.

## 4. Balance

- **Creators:** ~6,600. No creator contributes more than 0.5% of training hours, or a handful of voices
  dominate what "ASMR voice" means.
- **Gender:** the mommy/daddy split is ~54/46 by files; sample to 50/50 by hours.
- **Delivery:** whispering outnumbers normal speech 4:1. Keep it (it *is* the genre), but guarantee >= 20% normal
  / soft-spoken so the model can do both on request.
- **First run:** a stratified ~2,000 h subset (creator-capped, gender-balanced), not all 14k h. It is enough
  to validate the windows and the tag scheme cheaply; scale only after §5 passes.

## 5. Acceptance tests (what "does not hallucinate" is measured as)

| failure | metric | gate |
|---|---|---|
| wrong / skipped / invented words | Whisper large-v3 WER of output vs input text, per 20 s window | at or below the WER of real held-out recordings |
| invented or missing vocal events | v7 on each tagged gap: requested class detected? untagged gaps: any event detected? | precision and recall per tag, reported per class |
| wrong trigger | v7 P(requested class) on generated trigger windows | >= v7's accuracy on real held-out windows of that class |
| long-form drift | WER, speaker similarity and v7 scores across 6 chained windows (2 min) | no monotone degradation from window 1 to 6 |

All tests on held-out *creators*, never held-out clips of seen creators.
