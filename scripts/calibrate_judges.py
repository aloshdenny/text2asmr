#!/usr/bin/env python3
"""Which OpenRouter audio models to trust, per class, before a bulk labelling run.

Every candidate hears the same clips with the same 20-label prompt the bulk run uses (label_agreement.ask):
  * external  human-labelled trigger clips from FSD50K / ESC-50 (cut_external48.py's clips.json, one class each):
              per-class recall (did it name the class?) and which wrong triggers it names instead
  * people    clips people labelled on ASMR Board and the Ear Check kits: per-class precision / recall / F1 against
              the people's majority (a class is "present" when at least half of a clip's people ticked it)
The report ranks judges per class; fuse_labels.py later weights each judge by its measured reliability anyway, so the
point here is to leave out judges that add only noise (MiMo flash heard breathing in trigger audio) and to see which
classes no machine can hear (those go to people first).

Resumable (one jsonl per model under --work); a dollar cap across the whole run.

  python calibrate_judges.py --external D:/t2a/gate_ext48/clips.json --humans D:/t2a/fused/crowd_all.jsonl \\
      D:/t2a/fused/humans/*.jsonl --clips D:/t2a/pool D:/t2a/pool_yt ... --work D:/t2a/judgecal --cap 20
"""
from __future__ import annotations
import argparse, glob, json, os, random, re, subprocess, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_agreement import LABELS, ask

TRIGGERS = ["tapping", "scratching", "crinkling", "brushing", "liquid", "spraying", "microphone touching", "sticky",
            "fabric rustling", "paper rustling", "cutting"]
DEFAULT_MODELS = ["google/gemini-3.1-pro-preview", "google/gemini-3.8-flash", "openai/gpt-audio-mini",
                  "qwen/qwen3.8-omni-flash", "xiaomi/mimo-v2.6-pro", "thinkingmachines/inkling"]


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)
def safe(uid: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def long_path(p: str) -> str:
    p = os.path.abspath(p)
    return "\\\\?\\" + p if os.name == "nt" and not p.startswith("\\\\?\\") else p


def mp3_of(path: str) -> bytes:
    """The bulk run's clip format (6 s, mono, 24 kHz, 48 kbps mp3); pool clips already are, wavs are converted."""
    if path.endswith(".mp3"):
        with open(long_path(path), "rb") as fh: return fh.read()
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-t", "6", "-ac", "1", "-ar", "24000", "-b:a", "48k",
                        "-f", "mp3", "-"], capture_output=True, timeout=120)
    if r.returncode or not r.stdout: raise RuntimeError(r.stderr.decode()[-200:])
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", type=Path, required=True)
    ap.add_argument("--humans", type=Path, nargs="+", required=True)
    ap.add_argument("--clips", type=Path, nargs="+", required=True, help="pool dirs: clips under pool_*/clips")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--max-human", type=int, default=240, help="people's clips to use (trigger clips first)")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--cap", type=float, default=20.0, help="USD across all models")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    key = os.environ["OPENROUTER_API_KEY_GS"]

    ext = [{"uid": r["uid"], "set": "external", "path": r["wav"], "truth": {r["label"]}} for r in json.load(open(a.external, encoding="utf-8"))]
    ext = [r for r in ext if r["truth"] & set(LABELS)]   # page turning is scored as paper rustling below
    for r in json.load(open(a.external, encoding="utf-8")):
        if r["label"] == "page turning": ext.append({"uid": r["uid"], "set": "external", "path": r["wav"], "truth": {"paper rustling"}})
    files = {}
    for d in a.clips:
        for f in glob.glob(str(d / "pool_*" / "clips" / "*.mp3")): files.setdefault(Path(f).stem, f)
    votes: dict[str, dict[str, set]] = defaultdict(dict)
    for f in a.humans:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r.get("uid") and not r.get("skipped"): votes[r["uid"]][r.get("who") or f.stem] = set(r["labels"]) & set(LABELS)
    people = []
    for uid, vs in votes.items():
        if safe(uid) not in files: continue
        share = {c: sum(c in v for v in vs.values()) / len(vs) for c in LABELS}
        people.append({"uid": uid, "set": "people", "path": files[safe(uid)], "truth": {c for c, s in share.items() if s >= 0.5},
                       "n_people": len(vs)})
    rng = random.Random(0); rng.shuffle(people)
    people.sort(key=lambda r: -len(r["truth"] & set(TRIGGERS)))   # clips with a trigger first: those are the hard classes
    people = people[: a.max_human]
    items = ext + people
    log(f"{len(ext)} external clips, {len(people)} people's clips (of {len(votes)} labelled), models: {a.models}")

    audio, lock, spent = {}, threading.Lock(), [0.0]
    for f in a.work.glob("*.jsonl"):
        for l in open(f, encoding="utf-8"): spent[0] += json.loads(l).get("cost", 0.0)
    for it in items:
        try: audio[(it["set"], it["uid"])] = mp3_of(it["path"])
        except Exception as e: log(f"  skip {it['uid']}: {type(e).__name__}")
    items = [it for it in items if (it["set"], it["uid"]) in audio]

    for model in a.models.split(","):
        out = a.work / f"{safe(model)}.jsonl"
        done = {(r["set"], r["uid"]) for r in map(json.loads, open(out, encoding="utf-8"))} if out.exists() else set()
        todo = [it for it in items if (it["set"], it["uid"]) not in done]
        log(f"{model}: {len(todo)} clips to ask (${spent[0]:.2f} spent so far)")

        def one(it):
            if spent[0] >= a.cap: return None
            labels, raw, cost = ask(model, audio[(it["set"], it["uid"])], key)
            with lock:
                spent[0] += cost
                with open(out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"set": it["set"], "uid": it["uid"], "labels": labels, "error": raw.startswith("ERROR"),
                                         "raw": raw[-200:], "cost": cost}) + "\n")
            return labels

        with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, todo))
        if spent[0] >= a.cap: log(f"cap ${a.cap} reached"); break

    # ---- report
    truth = {(it["set"], it["uid"]): it["truth"] for it in items}
    lines, summary = [], {}
    for model in a.models.split(","):
        out = a.work / f"{safe(model)}.jsonl"
        if not out.exists(): continue
        rs = [r for r in map(json.loads, open(out, encoding="utf-8")) if (r["set"], r["uid"]) in truth]
        ok = [r for r in rs if not r["error"]]
        cost = sum(r["cost"] for r in rs) / max(1, len(rs))
        ext_rec, wrong = {}, Counter()
        for c in sorted({c for it in ext for c in it["truth"]}):
            rows = [r for r in ok if r["set"] == "external" and c in truth[("external", r["uid"])]]
            if rows: ext_rec[c] = (sum(c in r["labels"] for r in rows) / len(rows), len(rows))
            for r in rows: wrong.update(x for x in r["labels"] if x in TRIGGERS and x != c)
        f1 = {}
        for c in LABELS:
            tp = sum(c in r["labels"] and c in truth[("people", r["uid"])] for r in ok if r["set"] == "people")
            fp = sum(c in r["labels"] and c not in truth[("people", r["uid"])] for r in ok if r["set"] == "people")
            fn = sum(c not in r["labels"] and c in truth[("people", r["uid"])] for r in ok if r["set"] == "people")
            if tp + fn: f1[c] = (2 * tp / (2 * tp + fp + fn), tp + fn)
        trig = [f1[c][0] for c in TRIGGERS if c in f1]
        summary[model] = {"clips": len(rs), "errors": len(rs) - len(ok), "usd_per_clip": round(cost, 5),
                          "labels_per_clip": round(sum(len(r["labels"]) for r in ok) / max(1, len(ok)), 2),
                          "external_recall": {c: round(v, 2) for c, (v, _) in ext_rec.items()},
                          "people_f1": {c: round(v, 2) for c, (v, _) in f1.items()},
                          "trigger_f1_mean": round(sum(trig) / max(1, len(trig)), 3)}
        lines.append(f"\n== {model}: {len(rs)} clips, {len(rs) - len(ok)} errors, ${cost:.4f}/clip, "
                     f"{summary[model]['labels_per_clip']} labels/clip, trigger F1 vs people {summary[model]['trigger_f1_mean']}")
        lines.append("  external recall: " + ", ".join(f"{c} {v:.0%} (n={n})" for c, (v, n) in ext_rec.items()))
        lines.append("  says instead:    " + ", ".join(f"{c} {n}" for c, n in wrong.most_common(5)))
        lines.append("  F1 vs people:    " + ", ".join(f"{c} {v:.2f} (n={n})" for c, (v, n) in sorted(f1.items(), key=lambda x: -x[1][1])))
    (a.work / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (a.work / "report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines))
    log(f"CALIBRATE_DONE ${spent[0]:.2f} spent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
