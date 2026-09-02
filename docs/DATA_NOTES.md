# Data notes

Findings from auditing the two source repositories, and the decisions that
follow from them. Recorded because several contradict the T2A paper, and
future-us will otherwise re-derive them the hard way.

## Corpora

| | `aoxo/text2asmr-uncensored` | `aoxo/audios` |
|---|---|---|
| Role | **eval / baseline only** | **primary training corpus** |
| Audio | 22.05 kHz mono WAV | 48 kHz, stereo (some mono) AAC |
| Duration | ~37 h (2208 × 60 s) | ~1072 h total; ~375 h transcribed |
| Text | clip-level transcript | word-level timestamps + silence spans |
| Files | `wavs.zip` 5.8 GB | 4901 `.m4a`, 1690 `.json` |
| Gating | gated (auto) | public, apache-2.0 |

`aoxo/audios` is the pre-downgrade source audio for the same project. All 1690
JSON files have a matching m4a; no JSON is orphaned.

## Findings against the paper

### 1. The trigger tags were never released

The paper's section IV.A, steps 6-8, describes bracket-tag preprocessing:
trigger tags (`[brushing]`, `[tapping]`, `[crinkling]`, `[rustling]`) and
intensity prefixes (`[mild]`, `[soft]`, `[vigorous]`, `[loud]`).

**Zero bracket tags exist in any released artifact.** Verified across
`metadata.csv` (2208 rows), `metadata_original.csv` (84 rows), all 2208 files
in `transcripts.zip`, and `MyTTS.ipynb`. The FiLM and BERT conditioning code
*is* present in the notebook, so the architecture is real; the tag ontology
that was supposed to condition it is not.

Consequence: T2A as published trained on plain ASR transcripts of speech, and
the non-speech-element claim has no supervision behind it.

**Decision:** reconstruct the tags. See "Trigger recovery" below.

### 2. The "normalised" transcript column is a byte-identical copy

`metadata.csv` is `id|transcript|normalised`, LJSpeech-style, but `col2 == col3`
for all 2208 rows. There is no cleaned variant to fall back on.

### 3. Transcripts are corrupted by ASR looping

- 24.7% of clips contain a token repeated >=4x consecutively (longest run: 28)
- median type-token ratio 0.558

Some repetition is genuine -- several source videos are name-reading formats
where the artist really does repeat each name. But the long runs are
characteristic ASR looping on whispered speech, which is close to worst-case
for acoustic models.

**Decision:** do not use `text2asmr-uncensored` text for training. The
`aoxo/audios` word-level alignments are cleaner and carry timing.

### 4. The corpus is broader than the paper states

The paper says 12 ASMRtists. `metadata.csv` ids carry 89 distinct video
prefixes. Speaker count is unverified -- prefixes are per-video, not
per-artist -- but the corpus is wider than documented.

### 5. Binaural was discarded in preprocessing

The abstract claims "high-fidelity binaural ASMR audio", but the released
audio is 22.05 kHz mono (the paper's own step 2 confirms the conversion).
Binaural imaging and the 11-22 kHz band cannot be recovered by upsampling, and
ASMR depends on both.

`aoxo/audios` retains 48 kHz stereo, which is why it became the primary corpus.

## Trigger recovery

Since no trigger annotations exist, they are derived from `aoxo/audios`:

1. **Segment by alignment.** Word spans are speech; `silence` spans are
   candidate non-speech. No VAD needed -- the boundaries are given.
2. **Separate true silence from trigger audio.** A `silence` span is labelled
   by the ASR only because no *words* were found. In ASMR the gaps are
   typically full of brushing, tapping and crinkling. Split on loudness: spans
   near the noise floor are real silence, the rest are trigger candidates.
3. **Tag zero-shot with CLAP** against the ontology in `t2a/data/ontology.py`,
   rejecting any span whose best trigger score fails to beat the negative
   probes by `MARGIN`.
4. **Assign intensity** from loudness, per `intensity_from_loudness`.

This reconstructs the annotation the paper describes, at higher quality than
the paper's own pipeline, because the alignment is word-level.

Caveat worth stating in the model card: intensity-from-loudness is a proxy.
It tracks vigour well for contact triggers (harder tapping is louder) and less
well for others. It is the only cue available from audio alone.

## Local constraints

Prep runs on an M4 / 16 GB / ~66 GB free. The full `aoxo/audios` corpus is
58.4 GiB, so only the 20.5 GiB paired subset is fetched, and segments are
written out as the source files are decoded rather than after.
