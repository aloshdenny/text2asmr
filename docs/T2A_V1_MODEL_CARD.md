---
base_model: ResembleAI/chatterbox
library_name: peft
license: mit
language:
- en
tags:
- text-to-speech
- asmr
- lora
- chatterbox
pipeline_tag: text-to-speech
---

# text2asmr T3 v2 (T2A v1 speech adapter)

A LoRA adapter (r=64, 45M trainable parameters) on the T3 text-to-speech-token model of
[Chatterbox](https://huggingface.co/ResembleAI/chatterbox), trained to deliver ASMR-style speech: slow pacing,
whispered and soft-spoken delivery, and explicit pauses and vocal events requested inline in the text.

## What it does differently from base Chatterbox

- **Inline control tags.** Training text carries every pause and vocal event that happens in the audio:
  `[pause 1.5s]`, `[breathing]`, `[oral sounds]`, `[moaning]`. The model is never left to invent a pause or a
  breath; it produces the ones the script asks for.
- **Continuation.** Each training window is conditioned on the ~4 s of the same recording that preceded it
  (Chatterbox's speech-prompt slot). Long scripts are generated as chained windows of at most 20 s, each prompted
  with the tail of the one before, which keeps delivery consistent over minutes.
- **Window sizes come from the data.** Windows are 2-20 s (the corpus' phrase p99 is 19.8 s), text up to 330
  tokens. See `docs/GENERATOR_V2.md` in the project repository for the derivation.

## Training data

- 881,646 windows, 1,931 hours, from ~6,000 creators of long-form spoken ASMR audio (`aoxo/t2a-mommy`,
  `aoxo/t2a-daddy`), transcribed with Whisper large-v3 and event-labelled with Qwen3-Omni.
- Creator-capped (no creator above 10 h) and balanced across the two corpora.
- Windows whose transcript loops (Whisper repeating a filler word 4+ times, typically a mis-transcribed moan)
  were dropped, so the model is not taught to loop.

## Evaluation

- Held-out-creator next-token loss (74 creators never seen in training, 20,676 windows): **4.7613**, falling
  monotonically across the run (4.839 at step 1k).
- Listening evaluation and WER / event-fidelity tests are in progress; this card will be updated with them.

## Use

```python
from chatterbox.tts import ChatterboxTTS
from peft import PeftModel
from huggingface_hub import snapshot_download

model = ChatterboxTTS.from_pretrained("cuda")
path = snapshot_download("aoxo/text2asmr-t3-v2", allow_patterns=["adapter/*"]) + "/adapter"
model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, path).merge_and_unload()
```

For long scripts with chained continuation windows, use `scripts/generate_t2a_v1.py` from the project repository.

## Responsible use

- All output of the reference pipeline is watermarked with Resemble AI's Perth implicit watermark. Keep it on.
- The voice comes from the reference clip you supply. Only use voices you have the right to use, and never to
  impersonate a real person.
- Training audio is adult-oriented long-form ASMR; the model can produce intimate delivery and vocal sounds.

## Limitations

- English only.
- `[oral sounds]` and `[moaning]` are rarer in training than `[pause]` (about 1 event tag per 30 pauses), so their
  placement is less reliable than pauses.
- Vocal event tags are only as good as the Qwen3-Omni labels behind them; the CLAP v7 classifier that will verify
  them has not yet met its 90%-per-class release bar.
