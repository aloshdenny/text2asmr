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
    for r in rows:
        # the sampler strips geometry to keep the model's guess away from the annotator; the uid still
        # carries the gap start in milliseconds, and the cut convention is 1 s of pre-roll, 8 s long
        if "cut_start" not in r and ".m4a_" in r["uid"]:
            start = int(r["uid"].rsplit(".m4a_", 1)[1]) / 1000.0
            r["start"] = start; r["cut_start"] = max(0.0, start - 1.0); r["cut_duration"] = 8.0
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
    try:
        import bench_audio_llms as bench
        bench.CHOICES = THREE
    except Exception as e:                      # GPU runners are optional; OpenRouter judges need no torch
        log(f"bench runners unavailable ({type(e).__name__}); OpenRouter judges still work")
        class bench: MODELS = {}; CHOICES = THREE; PROMPT = ""
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
        if name.startswith("or:"):
            model = name[3:]
            log(f"{name}: judging {len(todo)} clips via OpenRouter (cap ${a.max_cost})")
            res = run_openrouter(a, todo, model, name)
        else:
            runner = bench.MODELS.get(name)
            if not runner: log(f"{name}: no such runner in bench (have {sorted(bench.MODELS)})"); continue
            log(f"{name}: judging {len(todo)} clips")
            res = runner(a, todo)
        with out_path.open("a") as fh:
            for r in res: fh.write(json.dumps(r) + "\n")
        got = Counter(r.get("label") for r in res)
        log(f"{name}: wrote {len(res)} -> {dict(got)}")


def run_openrouter(a, clips, model: str, name: str):
    """Judge via OpenRouter. Network-only, so it runs on the droplet while the GPU box is down, and it is
    the only judge here from a family that had no hand in our training labels."""
    import base64, urllib.request
    key = os.environ["OPENROUTER_API_KEY_GS"]
    prompt = ("This is a short clip from an ASMR recording. It contains exactly one of these three sounds. "
              "Answer with one of these three labels only, nothing else: kissing, moaning, mouth sounds.")
    spent = [0.0]; lock = threading.Lock(); out = []

    def parse(t: str):
        t = (t or "").lower()
        for c in ("mouth sounds", "mouth_sounds", "kissing", "moaning"):
            if c in t: return "mouth sounds" if c.startswith("mouth") else c
        return None

    def one(c):
        if spent[0] >= a.max_cost: return {"uid": c["uid"], "raw": None, "label": None, "error": "budget"}
        try:
            wav = base64.b64encode(open(c["wav"], "rb").read()).decode()
            body = {"model": model, "temperature": 0, "max_tokens": 2000,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": wav, "format": "wav"}}]}]}
            if a.reasoning: body["reasoning"] = {"effort": a.reasoning}
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                         "HTTP-Referer": "https://github.com/aloshdenny/text2asmr"})
            d = json.loads(urllib.request.urlopen(req, timeout=300).read())
            if "error" in d: return {"uid": c["uid"], "raw": None, "label": None, "error": str(d["error"])[:120]}
            u = d.get("usage") or {}
            txt = (d["choices"][0]["message"].get("content") or "")
            with lock:
                # if OpenRouter has not attributed a cost yet, price it from tokens so the cap still holds
                cost = float(u.get("cost") or 0.0) or (u.get("prompt_tokens", 0) * 2e-6 + u.get("completion_tokens", 0) * 12e-6)
                spent[0] += cost
            return {"uid": c["uid"], "raw": txt.strip()[:200], "label": parse(txt),
                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}
        except Exception as e:
            err = getattr(e, "read", lambda: b"")()
            return {"uid": c["uid"], "raw": None, "label": None, "error": f"{type(e).__name__} {str(e)[:60]} {err[:120]}"}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for i, r in enumerate(ex.map(one, clips)):
            out.append(r)
            if i % 25 == 0: log(f"  {name} {i}/{len(clips)} spent=${spent[0]:.3f} last={r.get('label')} err={r.get('error','')[:40]}")
    log(f"{name}: done, ${spent[0]:.3f} spent")
    return out


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
    ap.add_argument("--models", default="voxtral-mini-3b",
                    help="bench runner names, and/or 'or:<openrouter model>' which needs no GPU")
    ap.add_argument("--max-cost", type=float, default=3.0, help="hard USD cap per OpenRouter judge")
    ap.add_argument("--reasoning", default="", help="OpenRouter reasoning effort: low/medium/high, blank for none")
    ap.add_argument("--concurrency", type=int, default=8)
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
