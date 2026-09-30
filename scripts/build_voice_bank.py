#!/usr/bin/env python3
"""Reference-voice bank for T2A: tagged, balanced reference clips from creators the model never trained on.

The reference clip decides more than timbre: its *delivery* carries into generation (a formal narration read
makes a fast "interviewer" voice). So references are chosen on purpose:
  * candidates: 8-12 s untagged speech windows from held-out creators (train_t3_v2's creator split)
  * Gemini tags each: gender, accent, delivery (whisper / soft / narration / normal), pitch
  * the default set keeps soft/whispered delivery only, balanced across gender x accent

  python build_voice_bank.py --n 40 --out ~/t2a_samples/voice_bank
"""
from __future__ import annotations
import argparse, base64, json, os, random, re, subprocess, urllib.request, zlib
from collections import defaultdict
from pathlib import Path

PROMPT = ("This is a voice sample for text-to-speech voice cloning. Describe the speaker. Answer with JSON only: "
          '{"gender": "female|male|other", "accent": "British|American|Australian|Irish|Scottish|Indian|other", '
          '"delivery": "whisper|soft|normal|narration|energetic", "pitch": "low|mid|high", "clean": true|false} '
          '-- "clean" means a single speaker, no music, no background noise, no sound effects.')


def held_out(creator: str) -> bool:                      # the creator split train_t3_v2 used (--eval-pct 2)
    return zlib.crc32((creator or "").encode()) % 10000 < 200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="google/gemini-3.1-pro-preview")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    a.out.mkdir(parents=True, exist_ok=True)
    man = hf_hub_download("aoxo/t2a-speech-v2", "manifests/speech_windows_v2.jsonl", repo_type="dataset", local_dir=str(a.out / ".dl"))
    by_creator = defaultdict(list)
    for l in open(man, encoding="utf-8"):
        r = json.loads(l)
        if held_out(r["creator"]) and "[" not in r["text"] and 8 <= r["dur"] <= 12 and len(r["text"].split()) >= 12:
            by_creator[(r["repo"], r["creator"])].append(r)
    rng = random.Random(11); keys = sorted(by_creator); rng.shuffle(keys)
    # alternate corpora so both voices are represented, one clip per creator
    mommy = [k for k in keys if k[0].endswith("mommy")]; daddy = [k for k in keys if k[0].endswith("daddy")]
    picks = [k for pair in zip(mommy, daddy) for k in pair][: a.n]
    key = os.environ["OPENROUTER_API_KEY_GS"]; bank = []
    for repo, creator in picks:
        r = rng.choice(by_creator[(repo, creator)])
        wav = a.out / f"{re.sub(r'[^A-Za-z0-9_-]', '_', creator)}.wav"
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{r['source']}"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(r["start"]), "-t", str(r["dur"]), "-i", url, "-ac", "1", "-ar", "24000", str(wav)],
                       check=True, timeout=300)
        body = {"model": a.model, "temperature": 0, "max_tokens": 2000, "usage": {"include": True},
                "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                             {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav.read_bytes()).decode(), "format": "wav"}}]}]}
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            txt = json.loads(urllib.request.urlopen(req, timeout=240).read())["choices"][0]["message"]["content"]
            tags = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        except Exception as e:
            tags = {"error": f"{type(e).__name__}"}
        row = {"file": wav.name, "creator": creator, "repo": repo, "source": r["source"], "start": r["start"], "dur": r["dur"],
               "text": r["text"][:120], **tags}
        bank.append(row); print(json.dumps(row), flush=True)
    (a.out / "bank.jsonl").write_text("".join(json.dumps(b) + "\n" for b in bank))
    # default set: clean, soft/whisper delivery, one per gender x accent where available
    default = {}
    for b in bank:
        if b.get("clean") and b.get("delivery") in ("whisper", "soft"):
            default.setdefault(f"{b.get('gender')}_{b.get('accent')}", b)
    (a.out / "default_set.json").write_text(json.dumps(default, indent=1))
    print("default set:", {k: v["file"] for k, v in default.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
