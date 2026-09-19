# Data-balance pipeline: Qwen3-Omni labels + targeted 1 TB expansion

Goal: a CLAP model that is good on `audios2`/`audios3`, trained on labels that are (a) learnable from audio and
(b) balanced across the vocal ontology, from (c) many creators. Everything below is measured, not assumed.

## 0. Where we are (audios2, Qwen3-Omni-30B via vLLM, 1.89M gap clips, 443 creators)

| label | clips | share | hours | creators | top-creator share |
|---|---|---|---|---|---|
| whispering | 1,226,627 | 65.0% | 1,482 | 439 | 3.5% |
| normal speech | 211,926 | 11.2% | 177 | 424 | 3.7% |
| breathing | 192,048 | 10.2% | 394 | 426 | 4.9% |
| mouth sounds | 136,542 | 7.2% | 308 | 405 | 4.5% |
| moaning | 65,452 | 3.5% | 122 | 392 | 4.3% |
| kissing | 36,782 | 1.9% | 69 | 397 | 6.0% |
| silence | 9,738 | 0.5% | 27 | 357 | 7.2% |
| crinkling / tapping / liquid / scratching / brushing | 7,613 | 0.4% | 19 | — | tapping 38% from one creator |

Labeler: `Qwen/Qwen3-Omni-30B-A3B-Instruct` (audio-only prompt, `scripts/label_audios2_qwen3.py`), validated against
Gemini-3.1-Pro audio-only labels (57% exact agreement; learnability probes within 1-2 pts of Pro). Throughput with vLLM on one
A100-80GB: ~65 clips/s (~$8 per 1M clips). Gemini/GCP is out (billing); Claude has no audio input.

## 1. Ontology and targets (vocal core + physical tail)

Targets are for the **combined** audios2 + audios3 + expansion set, per label, with a creator cap so no creator exceeds 3% of a label.

| label | target clips | target hours | why |
|---|---|---|---|
| whispering | cap at 300k (subsample) | — | already 4x over; subsample by creator |
| normal speech | 150k | — | hard negative, keep |
| breathing | 150k | 300 | near target |
| mouth sounds | 150k | 300 | +15k |
| **moaning** | 150k | 250 | **+85k** — corpus-defining class, was "reject" before |
| **kissing** | 150k | 250 | **+113k** — the starved class |
| silence | 30k | — | negative |
| **each physical trigger** (tapping, crinkling, scratching, brushing, liquid, page turning) | 25k | 30 | soundgasm barely has them: source from YouTube "no talking" chapters (`docs`: `yt_scope_notalking.py`, DO box) |

Everything ≤4 s at training time (longer events are split); vocal classes are cut from ASR gaps as today.

## 2. Pipeline (closed loop, re-run until targets are met)

```
inventory -> deficit -> discover new creators (tag-targeted, exclusion list) -> acquire (stream to HF)
   -> transcribe (speech-gated Whisper, sharded) -> cut gaps -> Qwen3-Omni label (vLLM) -> inventory ...
   -> when targets met: build balanced manifest -> train CLAP v6 -> eval on Pro held-out + held-out creators
```

### 2.1 Inventory (`scripts/label_inventory.py`, Mac, seconds)
Input: `v2/audios2_qwen3omni_labels.jsonl` + `v2/audios2_clap_gate_index.jsonl` (and the audios3 equivalents).
Output: per-label clips / hours / creators / top-creator share (the table above), and `deficit.json` = target - current, per label,
after applying the 3%-per-creator cap. This is the only input to discovery, so acquisition stops automatically when a label is full.

### 2.2 Discovery (`scripts/discover_creators.py`, DO box, network only)
- Source: `ilovesoundgasm.com` search API (`/api/search?q=<tag>&cursor=`), same as the audios3 discovery.
- Query sets per deficit label (tags/keywords soundgasm creators actually use):
  - kissing: `kissing`, `kisses`, `mwah`, `smooches`, `kiss sounds`, `cheek kisses`
  - moaning: `moaning`, `moans`, `whimpers`, `heavy moaning`, `orgasm`
  - mouth sounds: `mouth sounds`, `wet sounds`, `licking`, `ear licking`, `lip smacking`, `sloppy`
  - breathing: `heavy breathing`, `breathing`, `panting`
  - (physical labels are *not* sourced from soundgasm; see YouTube route)
- **Exclusion**: creators already in `aoxo/audios2` or `aoxo/audios3` (creator = first path component). Build once from the Hub file
  lists into `label_tool/v2/existing_creators.txt`; the discovery script refuses any uploader in it.
- Ranking: for each new creator, score = (#posts hitting deficit tags) x (deficit weight of those labels); keep creators with >= 8
  matching posts; **cap 40 files / 4 GB per creator** so the expansion adds breadth (new creators) not depth.
- Gender balance: deficits are computed **per repo** (female/audios2 and male/audios3 separately), so kissing/moaning get filled on both sides; the voice tag decides the destination repo (see 2.3).
- Output: `expansion_plan.jsonl` (creator, post URLs, predicted label contributions, GB). Stop adding creators when the projected
  contribution covers every deficit or the byte budget is reached.

### 2.3 Acquire (DO box for downloads; `scripts/acquire_stream.py`)
- Stream one file at a time: download -> push to **`aoxo/audios2` (female voice) or `aoxo/audios3` (male voice)** -> delete
  local. The repo is the gender axis and stays that way: route by the post's voice tag (`F4M`/`F4F`/`F4A`/`F4TF` -> audios2;
  `M4F`/`M4M`/`M4A` -> audios3; multi-voice tags like `FF4M`/`MF4F` -> the leading voice letter; posts with no voice tag are
  checked against the creator's other posts, and skipped if still ambiguous). Provenance lives in `expansion_manifest.jsonl`
  (creator, file, tags, source query, batch), not in the repo name; new creators land under their own `<creator>/` prefix like
  every existing one, so sharding, transcription resume (`<file>.json` present => done) and labeling work unchanged. The droplet has ~15 GB disk and 1 vCPU, so it must never hold more than a few files; at ~8 MB/s a 1 TB expansion is ~35 h
  of wall-clock. If that is too slow, a RunPod CPU pod (~$0.10/h, 200 GB disk) does it in a few hours.
- Byte budget: ~1 TB total across audios2+audios3 expansion, allocated in proportion to deficits (kissing/moaning first).
- Record `expansion_manifest.jsonl` (creator, file, size, tags, source query) so every file traces back to the deficit it serves.

### 2.4 Transcribe (tinkerspace + RunPod shards, `scripts/transcribe_audios2.py --repo aoxo/audios4`)
- Speech-ratio gate already skips near-silent files (<5% speech or <20 s) before Whisper runs.
- Sharding: `--num-shards N --shard-index i` by file hash; each shard commits <= every 25 min (Hub-resumable).
- Rate: ~2.3 files/min on tinkerspace, ~3.6 on a 4090. Plan 4 shards for a 60k-file expansion (~4 days, ~$0.74/h per pod).

### 2.5 Cut + label (A100-80GB, `scripts/label_audios2_qwen3.py`)
- Candidates = Whisper gaps (+-1 s, <= 8 s), one decode per source, bounded worker window (fixed the OOM), 16 kHz wav per clip,
  CLAP-v3 scores recorded (second opinion, not a gate).
- vLLM serve `Qwen3-Omni-30B-A3B-Instruct` (cu129 wheel on 12.8 drivers; `ninja` required for FlashInfer JIT), labeler in
  `--follow` mode with `--delete-wav`, ledger snapshot-uploaded every 30 min.
- Prompt lists `moaning` and `normal speech` explicitly (the old prompt folded both into "reject", which is why v1-v3 never had them).

### 2.6 Build the balanced training set (`scripts/build_clap_v2_subset.py` successor)
- Union of Pro labels (8.6k, highest quality) + Qwen3 labels (audios2 + audios3, including the expansion files).
- Per label: cap at target, **3% per-creator cap**, alpha=0.5 tempering for anything still uneven, `normal speech` + `silence` as
  explicit negatives, whispering capped at 300k.
- Splits by **creator** (not clip, not file): 20% of creators held out. Also keep the Pro held-out set as the fixed yardstick.

### 2.7 Train + evaluate CLAP v6 (`scripts/train_clap_v3.py` / `v5`, RunPod)
- Joint contrastive + classification head (bg = silence/speech), augmentation on, early stop on held-out-creator macro recall.
- Report per-label recall on (a) Pro held-out, (b) held-out creators, (c) audios3-only held-out creators (male voices).
- Ship when kissing/moaning/mouth/breathing each exceed ~85% recall at >= 85% precision with abstention.

## 3. Budget and timeline (rough)

| step | resource | time | cost |
|---|---|---|---|
| audios3 transcription (in progress) | tinkerspace + 1x4090 | ~5 days | ~$90 |
| audios3 gap labeling | 1x A100 | ~10 h | ~$16 |
| discovery + 1 TB acquisition | DO box (or CPU pod) | 1-2 days | ~$0-5 |
| expansion transcription (~60k files) | 4 GPU shards | ~4 days | ~$280 |
| expansion labeling | 1x A100 | ~15 h | ~$25 |
| CLAP v6 training + eval | 1x A6000/L40S | ~3 h | ~$3 |

The expensive line is transcription. Two ways to cut it: transcribe only the expansion files whose tags match a deficit label
(discovery already filters for that), and raise the speech gate (skip files with <15% speech) since we only need the gaps.

## 4. Guardrails
- Never acquire from a creator already in audios2/audios3 (exclusion list is the first check in discovery).
- Never let one creator exceed 3% of any label in the training set.
- Every acquisition batch re-runs the inventory before the next batch; no label is over-collected.
- Commit ledgers every <= 30 min; every stage is resumable from the Hub.
- Pods are terminated between stages; the placement rule stands (network on DO, compute on RunPod, Mac orchestrates).
