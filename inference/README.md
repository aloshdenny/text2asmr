# Local inference: T2A speech adapter

Conda env `t2a-infer` (python 3.11, torch+MPS for the M4).

    /opt/homebrew/Caskroom/miniconda/base/bin/conda activate t2a-infer
    python scripts/infer_speech.py \
        --ref inference/refs/113_000356120.flac \
        --prompts inference/prompts/dummy_prompts.txt \
        --out inference/out

`--base-only` runs stock Chatterbox with no adapter, for an A/B comparison.

## Setup notes

- `resemble-perth`'s watermarker silently becomes `None` on import failure
  (wrapped in a try/except in `perth/__init__.py`). The cause here was
  `pkg_resources` missing because setuptools >=81 dropped it; pin
  `setuptools<81` to get it back.
- Reference clips in `inference/refs/` are drawn from the training corpus
  (`aoxo/text2asmr-segments`, speech shard 0) purely for voice/delivery
  conditioning -- their content is unrelated to what gets generated.
