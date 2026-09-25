#!/usr/bin/env python3
"""Forced three-way judging of kissing / moaning / mouth sounds by several independent models.

Purpose is triage, not truth: the human still decides every label that lands in the eval set.  What this
buys is attention -- where independent judges agree the clip is unambiguous, and where they disagree is
exactly where a person's ears are worth spending.

Judges:
  * qwen3-omni  -- free, already computed: for these strata its open-set label *is* its answer.  Not
                   independent of our training labels (it produced them), so it never breaks a tie alone.
  * voxtral-mini-3b (Mistral), af3 (NVIDIA) -- different families, run locally on the free GPU.

Gemini is absent on purpose: both of the account's projects are access-denied, verified by API call.

  python3 ensemble_3way.py --stage cut,judge,report --models voxtral-mini-3b --manifest humaneval/...jsonl
"""
from __future__ import annotations
import argparse, json, os, queue, subprocess, sys, threading, time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

THREE = ["kissing", "moaning", "mouth sounds"]
SR = 16_000


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def stage_cut(a):
    """Fetch each source once and cut its clips -- per-clip range requests cost ~3 s each."""
    from huggingface_hub import hf_hub_download
    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    rows = [r for r in rows if r.get("stratum") in THREE]
    log(f"{len(rows)} clips in the three strata")
    wav_dir = a.work / "wav"; wav_dir.mkdir(parents=True, exist_ok=True)
    by_src = defaultdict(list)
    for r in rows: by_src[(r["repo"], r["source"])].append(r)
    out, done = [], 0
    idx_path = a.work / "clips.json"
    for (repo, src), rs in by_src.items():
        local = None
        try:
            local = hf_hub_download(repo, src, repo_type="dataset", cache_dir=str(a.work / "cache"))
            for r in rs:
                w = wav_dir / (r["uid"].replace("/", "__") + ".wav")
                if not w.exists():
                    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{r.get('cut_start', r.get('start', 0)):.3f}",
                           "-t", f"{r.get('cut_duration', 8.0):.3f}", "-i", str(local),
                           "-ar", str(SR), "-ac", "1", str(w)]
                    if subprocess.run(cmd, capture_output=True, timeout=180).returncode != 0: continue
                out.append({"uid": r["uid"], "wav": str(w), "stratum": r["stratum"],
                            "source": r["source"], "repo": r["repo"]})
        except Exception as e:
            log(f"  source failed {src[:50]}: {type(e).__name__}")
        finally:
            if local:
                try: os.remove(os.path.realpath(local))
                except Exception: pass
        done += 1
        if done % 25 == 0: log(f"  {done}/{len(by_src)} sources, {len(out)} clips cut")
    idx_path.write_text(json.dumps(out))
    log(f"cut {len(out)} clips -> {idx_path}")


def stage_judge(a):
    """Reuse the benchmark's model runners, with the open-set prompt swapped for a forced choice."""
    import bench_audio_llms as bench
    bench.CHOICES = THREE
    bench.PROMPT = ("This is a short clip from an ASMR recording. It contains exactly one of these three sounds. "
                    "Answer with one of these three labels only, nothing else: kissing, moaning, mouth sounds.")
    clips = json.loads((a.work / "clips.json").read_text())
    log(f"{len(clips)} clips to judge")
    for name in [m.strip() for m in a.models.split(",") if m.strip()]:
        out_path = a.work / f"judge_{name}.jsonl"
        have = set()
        if out_path.exists():
            for l in out_path.open():
                try: have.add(json.loads(l)["uid"])
                except Exception: pass
        todo = [c for c in clips if c["uid"] not in have]
        if not todo: log(f"{name}: already complete"); continue
        runner = bench.MODELS.get(name)
        if not runner: log(f"{name}: no such runner in bench (have {sorted(bench.MODELS)})"); continue
        log(f"{name}: judging {len(todo)} clips")
        res = runner(a, todo)
        with out_path.open("a") as fh:
            for r in res: fh.write(json.dumps(r) + "\n")
        got = Counter(r.get("label") for r in res)
        log(f"{name}: wrote {len(res)} -> {dict(got)}")


def kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa: agreement above what two judges would reach by guessing with their own biases."""
    if not pairs: return float("nan")
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    ax, ay = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((ax[c] / n) * (ay[c] / n) for c in set(ax) | set(ay))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def stage_report(a):
    clips = {c["uid"]: c for c in json.loads((a.work / "clips.json").read_text())}
    judges: dict[str, dict[str, str]] = {"qwen3-omni": {u: c["stratum"] for u, c in clips.items()}}
    for p in sorted(a.work.glob("judge_*.jsonl")):
        name = p.stem.replace("judge_", "")
        d = {}
        for l in p.open():
            try: r = json.loads(l)
            except Exception: continue
            if r.get("label"): d[r["uid"]] = r["label"]
        if d: judges[name] = d
    log(f"judges: {[f'{k}({len(v)})' for k, v in judges.items()]}")
    if len(judges) < 2:
        log("need at least two judges before a report means anything"); return

    common = set.intersection(*(set(v) for v in judges.values()))
    log(f"{len(common)} clips judged by every judge")
    for x, y in combinations(sorted(judges), 2):
        pairs = [(judges[x][u], judges[y][u]) for u in common]
        agree = sum(1 for p, q in pairs if p == q) / max(len(pairs), 1)
        log(f"  {x:18} vs {y:18} agree {agree:.1%}  kappa {kappa(pairs):.3f}")
    for cls in THREE:
        sub = [u for u in common if judges["qwen3-omni"][u] == cls]
        if not sub: continue
        for x, y in combinations(sorted(judges), 2):
            pairs = [(judges[x][u], judges[y][u]) for u in sub]
            agree = sum(1 for p, q in pairs if p == q) / len(pairs)
            log(f"  [{cls:13}] {x:18} vs {y:18} agree {agree:.1%} (n={len(sub)})")

    # unanimous clips need no ears; the rest are the human queue, ordered by how split the judges are
    unanimous, contested = [], []
    for u in sorted(common):
        votes = Counter(judges[k][u] for k in judges)
        row = dict(clips[u], votes=dict(votes))
        (unanimous if len(votes) == 1 else contested).append(row)
    q = a.work / "human_queue.jsonl"
    q.write_text("".join(json.dumps(r) + "\n" for r in contested))
    (a.work / "provisional_unanimous.jsonl").write_text("".join(json.dumps(r) + "\n" for r in unanimous))
    mins = len(contested) * 11 / 60
    log(f"unanimous {len(unanimous)} ({len(unanimous)/max(len(common),1):.0%})  contested {len(contested)} -> {q}")
    log(f"human listening needed: ~{mins:.0f} min at 11 s/clip (was ~{len(common)*11/60:.0f} min unfiltered)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/home/tinkerspace/t2a/ens"))
    ap.add_argument("--manifest", type=Path, default=None, help="human eval manifest (needs stratum + geometry)")
    ap.add_argument("--stage", default="cut,judge,report")
    ap.add_argument("--models", default="voxtral-mini-3b")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    for st in a.stage.split(","):
        st = st.strip()
        if st == "cut":
            if not a.manifest: raise SystemExit("--manifest is required for the cut stage")
            stage_cut(a)
        elif st == "judge": stage_judge(a)
        elif st == "report": stage_report(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
