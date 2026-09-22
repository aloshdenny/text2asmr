# text2asmr

Text → ASMR audio generation, built as three layers:

1. **Data**: two large, public, creator-attributed ASMR corpora (`aoxo/audios2`, female voice; `aoxo/audios3`, male voice), word-aligned transcripts, and a non-speech "trigger" label layer on every gap between words.
2. **Ontology model (CLAP)**: a fine-tuned audio–text embedding model that can hear what the transcript cannot say (whispering vs. normal speech, kissing, mouth sounds, breathing, moaning, tapping, …). It is the supervision signal for the generator and the tool that lets us label millions of clips cheaply.
3. **Generator (T2A)**: text (with bracket-tag triggers) → speech + trigger audio → mixed ASMR clip.

Status (2026-09-22): the data and ontology layers are the active work. Transcription of both corpora is at 80–86% and finishing tonight; Qwen3-Omni labeling of the gaps runs nightly; CLAP v6 (trained on the fully-labeled, balanced corpus) is the next run. The generator v1 (Chatterbox + Stable Audio Open LoRAs, plus a from-scratch native model) exists as a baseline and will be retrained once the ontology is trustworthy.

---

## 1. Why an ontology model comes first

The original T2A paper conditions a generator on bracket tags like `[whispering]`, `[tapping]`, `[soft]`. Auditing the released dataset (`docs/DATA_NOTES.md`) showed that **no such tags were ever released**: the model was trained on plain ASR transcripts, the binaural 48 kHz audio was downsampled to 22 kHz mono, and a quarter of the transcripts were corrupted by ASR looping on whispered speech.

So the tags have to be reconstructed from the audio itself, at scale (the two corpora are ~118k files / >1,000 GB). A model that scores short audio windows against a fixed ontology is the only way to do that, and the same model later scores the generator's output. Everything in §2–§4 is about getting that model right.

### Ontology (current)

Vocal core (what the corpus actually contains): `whispering`, `normal speech`, `breathing`, `mouth sounds`, `moaning`, `kissing`, `silence`.
Physical tail (rare on soundgasm, sourced from YouTube "no-talking" chapters): `tapping`, `scratching`, `crinkling`, `brushing`, `liquid`, `page turning`, `other sound`.

Rules that shaped it: a label spans **≤3–4 s** (events, not scenes); `reject`/ambiguous windows are background negatives, never a class; classes that cannot be told apart from audio alone are dropped rather than left as noise.

---

## 2. Labelers: from frontier API to open model (the "superalignment" loop)

The core problem is that we cannot hand-label millions of 3-second windows, and the cheap labelers turned out to be wrong in ways that are invisible until you train on them. We used a weak-to-strong, teacher-validates-student loop, iterated three times:

| Round | Labeler | Cost / speed | What we learned |
|---|---|---|---|
| 0 | Zero-shot CLAP (`laion/clap-htsat-unfused`) with text probes per trigger, margin over negative probes | free | Agrees with later ground truth only ~4–8%. Fine for "is there sound here", useless for *which* sound. |
| 1 | Gemini 3.6 Flash, with 2 transcript words of context each side, `thinkingBudget=1` | ~$1/10k clips, 2.1M labels | **Not learnable from audio.** A linear probe on frozen CLAP embeddings scored at majority baseline; tapping-vs-page-turning 51%. Every CLAP fine-tune on these collapsed (all audio embeddings cos≈0.9). The labeler was reading the transcript, not listening. |
| 2 | Gemini 3.1 Pro, **audio only**, uncapped thinking (`scripts/pilot_gemini_pro.py`) | ~$4/1k clips, 8.6k labels | Learnable: probes 85–94% on class pairs. Revealed the corpus is *vocal* ASMR (physical triggers ≈0%). This is our ground-truth set. GCP billing then failed, so it could not be scaled. |
| 3 | Open audio LLMs benchmarked against the Pro set (`scripts/bench_audio_llms.py`): Qwen3-Omni-30B-A3B, Qwen2.5-Omni-7B, Audio-Flamingo-Next/3, Voxtral, Qwen2-Audio | free after GPU | **Qwen3-Omni-30B-A3B-Instruct** wins: 57% exact agreement with Pro and, more importantly, its labels are as learnable as Pro's (probe accuracy within 1–2 pts). Served with vLLM on one A100-80GB at ~65 clips/s (~$8 per 1M clips). |

The principle we kept applying: **validate a labeler by whether its labels are learnable from audio, not by whether they look plausible.** Learnability = train a linear probe on frozen CLAP embeddings; if it can't beat the majority class, the labels encode something not in the audio (transcript context, priors) and will poison training. The Pro set is the fixed yardstick; every new labeler and every new CLAP version is scored against it and against held-out creators.

The loop is then closed with the student: CLAP v3+ pseudo-labels the unlabeled pool with a calibrated head (`scripts/pseudo_label_clap.py`, keep only ≥85%-precision predictions), those are added back, and the teacher (Pro/Qwen3) audits samples of the student's confident predictions. Round 3's Qwen3-Omni now labels every gap in both corpora (3.15M clips so far).

---

## 3. CLAP fine-tuning: version history

Base model everywhere: `laion/clap-htsat-unfused` (HTS-AT audio tower + RoBERTa text tower, 512-d joint space). Audio path is our own GPU log-mel (validated to 0.03 dB against the HF processor; slaney filters, 1001×64 frames of 10 s @ 48 kHz), so we can precompute fp16 mel shards once and train from memmaps.

| Version | Data | Objective / architecture | Result |
|---|---|---|---|
| **v1** `finetune_clap.py` | 17k Gemini-flash-labeled gap clips, live-refreshing manifest | Standard CLAP contrastive (audio↔caption), LoRA-free full fine-tune | 39% agreement with Gemini vs 7.5% for stock CLAP — but Gemini-flash labels were themselves untrustworthy (§2). Retired. |
| **v2** `train_clap_v2.py` | Class-tempered subset of the flash labels (temperature sampling to flatten the whispering-heavy distribution) | **Multi-positive contrastive**: all same-class clips in the batch are positives (SupCon over the audio–text similarity matrix) | Collapsed (audio cos 0.9). A memorisation test (shuffle labels → loss still drops) proved the pipeline was fine and the labels were the problem. |
| **v3** `train_clap_v3.py` | 8.6k Gemini-Pro audio-only labels; 3 targets (kissing / mouth sounds / breathing) | Multi-positive contrastive **+ background negatives** (reject/whispering windows are only ever negatives) **+ joint 4-way linear head** (`head.pt`, targets + background) for calibrated abstention; SpecAugment-style gain / freq / time masking | Held-out: 4-way acc 0.90, background rejection 0.93, kissing recall 0.92, mouth 0.66, breathing 0.64–0.72. Self-training over a 296k pool added only 3.5k confident positives — mouth/breathing were data-starved (223 / 86 train clips). |
| **v4** `clapv4.sh`, `prep_yt_chapters.py` | v3 data + YouTube ASMR videos with creator-written chapters (146 videos, `aoxo/asmr-yt-chapters`), windowed and Silero-VAD speech-gated | Chapter title → weak label for every 10 s window in the chapter | Adds the physical-trigger classes the soundgasm corpus lacks, but chapter labels are noisy at window level (a "tapping" chapter has silence, talking, and tapping). Motivated v5. |
| **v5** `train_clap_v5.py` | Same as v4 | **Multiple-instance learning**: a YouTube chapter is a *bag* of windows; the top-k mean of window scores must match the chapter label. Pro clips are single-window bags; whisper/talk chapters and Pro rejects are background bags (mean-pooled). Contrastive loss kept at 0.5 weight. | Chapter-level accuracy on held-out videos + window-level on Pro held-out; MIL stops the model being punished for the quiet windows inside a chapter. |
| **v6** (next) | Full Qwen3-Omni labels on audios2 + audios3 + ~600 GB expansion (3.15M+ clips, ~6k creators), + Pro set as held-out | Same v3/v5 objectives, plus the imbalance fixes in §4 | Target: match Pro accuracy per class on held-out creators, per gender. |

Checkpoints: `aoxo/clap-htsat-unfused-asmr-v2` (`v3/{teacher,student_agree,student_all}`, `v4/`, `v5/`, each with `head.pt`). The processor must remain `laion/clap-htsat-unfused`; the *fused* CLAP variant crashes in training (BatchNorm) and is never used.

---

## 4. Data imbalance: what it is and what we do about it

Label distribution across both corpora after Qwen3-Omni labeling (3.15M clips):

| label | share | note |
|---|---|---|
| whispering | 62% | 4× over any sensible target |
| normal speech | 16% | male side is more speech-heavy (23%) |
| breathing | 10% | |
| mouth sounds | 6% | |
| moaning | 3.2% | corpus-defining, was "reject" under the old prompt |
| kissing | 1.5% | the starved class |
| silence / physical triggers | <1% | physical triggers come from YouTube chapters instead |

Two independent fixes, because oversampling alone does not add diversity — it just makes the model see the same kissing clips more often.

### 4.1 Acquire more of the rare classes (`docs/DATA_BALANCE_PIPELINE.md`)

A closed loop: `inventory → per-label deficit → discover new creators → acquire → transcribe → cut gaps → label → inventory`.
- Discovery hits the soundgasm search API with tag queries per deficit label (kissing, moaning, mouth sounds, breathing, plus generic sweeps), **excludes every creator already in the corpora** (3,746), routes each post by its voice tag (`F4*` → audios2, `M4*` → audios3) so gender dimorphism is preserved, and caps files per creator so the expansion adds breadth (new voices) rather than depth.
- Two rounds so far: 5,100 new creators, ~594 GB streamed straight to the Hub from a 1-vCPU droplet (download → commit in batches of 12 → delete).
- Measured, not assumed: tag-targeting lifts kissing/moaning yield only ~1.2–1.5× over a random creator, so the original "bring every class to half of whispering" target is not reachable from this source; the realistic target is ≥100k clips per vocal class with whispering capped, and physical triggers from YouTube.

### 4.2 Train so the long tail is learned, not just seen

Applied in v6 (`docs/DATA_BALANCE_PIPELINE.md` §3):
- **Sampling over (label × creator)**, not label alone: a batch draws classes with a tempered distribution, then draws a *creator* uniformly within the class, then a clip. No creator may exceed 3% of a class. This is what turns "oversampling" into diversity.
- **Whispering cap** (300k, creator-stratified subsample) and **effective-number class weights** (Cui et al.) / focal loss on the head.
- **Decoupled training (cRT)**: the encoder is trained with instance-balanced sampling (representation quality), then the classification head is re-fit with class-balanced sampling.
- **Channel augmentation** for the rare classes: room impulse responses, EQ, codec round-trips, gain, same-class mixup, so each rare clip yields many acoustically distinct views.
- **Hard negatives**: whispering↔breathing and mouth sounds↔kissing pairs are mined into the same batch.
- **Diversity is measured**, not hoped for: per-label creator entropy and embedding-cluster counts before/after balancing, with creator-split evaluation per gender so a class that is "solved" on one creator's mic is not reported as solved.

---

## 5. The generator (T2A)

Bracket-tag script → audio. The grammar (`text2asmr/compose/grammar.py`) parses the paper's notation (`[whispering] hello there [tapping] [soft] ...`) into an ordered list of speech and trigger segments.

| Component | Model | Training | Status |
|---|---|---|---|
| Speech | **Chatterbox** (Llama-backed T3 text→S3 speech tokens, S3Gen vocoder, voice encoder) | `scripts/train_speech.py`: fine-tune only T3 with a causal-LM loss over speech tokens, conditioned on text + a reference clip from the same recording, so it learns whispered delivery, breathiness and pacing rather than one speaker; S3Gen and the voice encoder frozen. Checkpoint `aoxo/text2asmr-chatterbox`. | v1 adapter works; A/B vs base in `inference/`. |
| Triggers | **Stable Audio Open 1.0** LoRA | `scripts/train_triggers.py`: LoRA via stable-audio-tools on ~4.4 s trigger clips captioned from the ontology (`"ASMR tapping, close-mic binaural recording, no speech"`). | v1 frozen: samples were noise/artifacts. Cause traced to the label layer (v1 tags were the untrustworthy zero-shot CLAP tags) — which is exactly why the ontology work in §2–§4 happens before generator v2. |
| Native | **BERT → FiLM U-Net → DDPM** (`text2asmr/native/`), the paper's own architecture from scratch | `scripts/train_native.py`, built with checkpoint/resume from the start. | Trained as a baseline to compare against fine-tuned pretrained backbones on the same corpus. |
| Renderer | `scripts/compose_asmr.py` | Walks the parsed script, generates each segment with the model that owns it (two conda envs, handing off through per-segment `.npy`), resamples to one rate and crossfades. | Working end-to-end for v1 models. |

Generator v2 plan: retrain Chatterbox T3 on the fully transcribed audios2/audios3 (whisper-vs-speech labels become a conditioning tag), retrain the trigger model on CLAP-v6-verified clips, add CLAP-v6 as a reward/consistency scorer (does the generated window score as the tag it was asked for?), and treat the long-context problem (smooth minutes-long output) with prefill/completion windows over the token sequence.

---

## 6. Data pipeline and infrastructure

- **Corpora**: `aoxo/audios2` (63k files) and `aoxo/audios3` (55k files), 48 kHz AAC, one `<creator>/` prefix per uploader, word-level alignment JSON beside each file. All repos are public.
- **Transcription** (`scripts/transcribe_audios2.py`): faster-whisper large-v3 fp16, batched inference pipeline, Silero-VAD **speech gate** (skip files with <5% speech or <20 s of speech — no GPU time on near-silent files), Hub-resumable (`<file>.json` present ⇒ done), sharded across machines by `crc32(path) % n`, commits every ≤30 min.
- **Gap cutting** (`scripts/candidates_from_transcripts.py`): every silence span between words becomes a candidate clip (1 s pre-roll, ≤8 s), clips ≥ noise floor only.
- **Labeling** (`scripts/label_audios2_qwen3.py`): bounded sliding-window prep (int16 decode, per-clip 16 kHz wav) → vLLM `Qwen3-Omni-30B-A3B-Instruct` with an audio-only prompt → incremental ledgers in `aoxo/clap-ft-data` (`v2/*_qwen3omni_labels.jsonl`).
- **Compute**: Modal (`modal_t2a.py`) — a single scheduled `dispatcher` respawns 7 L40S Whisper shards + 1 A100 label batch daily; every worker self-limits to 23 h (Modal containers die at 24 h) and resumes from the Hub, so backups are implicit. Network-heavy acquisition runs on a DigitalOcean droplet; the Mac only orchestrates. Earlier phases ran on RunPod (`scripts/launch_*_pod.py`, `runpod_*.py`).
- **Cost discipline**: nothing goes to a GPU until a linear probe shows the labels are learnable; first eval is watched before a run is allowed to continue; idle pods are killed by watchdogs.

Key artifacts on the Hub:

| repo | contents |
|---|---|
| `aoxo/audios2`, `aoxo/audios3` | raw audio + alignments (female / male) |
| `aoxo/clap-ft-data` | Pro pilot labels, Qwen3-Omni ledgers, gate index, acquisition snapshot |
| `aoxo/asmr-yt-chapters` | 146 chaptered YouTube ASMR videos for physical triggers |
| `aoxo/clap-htsat-unfused-asmr-v2` | CLAP v3/v4/v5 checkpoints + heads |
| `aoxo/text2asmr-chatterbox`, `aoxo/text2asmr-stable-audio` | generator v1 adapters |

---

## 7. Repository map

```
docs/DATA_NOTES.md              audit of the released T2A data; why tags are reconstructed
docs/DATA_BALANCE_PIPELINE.md   the closed-loop balance/expansion pipeline and v6 training recipe
text2asmr/data/ontology.py      trigger ontology + zero-shot CLAP probes
text2asmr/compose/grammar.py    bracket-tag script parser
text2asmr/native/               BERT→FiLM U-Net→DDPM from-scratch generator
text2asmr/models/chatterbox_ft.py  Chatterbox T3 fine-tune wrapper
scripts/transcribe_audios2.py   gated, sharded, Hub-resumable Whisper
scripts/candidates_from_transcripts.py, label_audios2_qwen3.py   gap cutting + Qwen3-Omni labeling
scripts/pilot_gemini_pro.py, bench_audio_llms.py                  ground-truth pilot + labeler benchmark
scripts/train_clap_v2.py / v3.py / v5.py, pseudo_label_clap.py    CLAP versions + self-training
scripts/prep_yt_chapters.py, yt_vad_probe.py                      YouTube chapter bags
scripts/discover_creators.py, acquire_stream.py                   deficit-targeted expansion
scripts/train_speech.py, train_triggers.py, train_native.py, compose_asmr.py   generator
modal_t2a.py                    Modal deployment (transcription shards + label batch + dispatcher)
```
