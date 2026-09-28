#!/usr/bin/env python3
"""T2A sample check: Whisper small.en transcript vs the script (tags stripped). Deletions = skipped words
(content hallucination), insertions = extra words (e.g. tags read aloud). Run over ~/t2a_samples/{v1,base}."""
import glob, os, re
import jiwer
from faster_whisper import WhisperModel
m = WhisperModel("small.en", device="cpu", compute_type="int8")
norm = lambda s: re.sub(r"[^a-z' ]", " ", re.sub(r"\[[^\]]*\]", " ", s.lower())).split()
rows = []
for d in ("v1", "base"):
    for f in sorted(glob.glob(os.path.expanduser(f"~/t2a_samples/{d}/*.wav"))):
        script = os.path.basename(f).split("__")[0]
        ref = " ".join(norm(open(os.path.expanduser(f"~/t2a_samples/scripts/{script}.txt")).read()))
        hyp = " ".join(norm(" ".join(s.text for s in m.transcribe(f, language="en", vad_filter=False)[0])))
        o = jiwer.process_words(ref, hyp)
        rows.append((d, os.path.basename(f), o.wer, o.deletions, o.insertions, o.substitutions, len(ref.split())))
for r in rows: print(f"{r[0]:4} {r[1]:32} WER {r[2]:5.0%}  deleted {r[3]:3d}  inserted {r[4]:3d}  substituted {r[5]:3d}  of {r[6]} words")
