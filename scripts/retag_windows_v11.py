#!/usr/bin/env python3
"""T2A v1.1 data: re-tag the v1 training windows with fused (Qwen x Gemini, human-anchored) labels.

v1 learned its inline vocal tags from Qwen3-Omni gap labels, and fusion measured those at breathing 45%,
oral sounds 56% precise -- 15% of "breathing" is actually moaning, which is exactly the formal-speech-slides-
into-moans the user heard. The audio tokens are unchanged, so this only rewrites text:

  * each window's gaps are re-derived from its word alignment (same phrase rules as the builder) and matched
    one-to-one, in order, to the tags in its text (a count mismatch leaves the window as it was, and is counted)
  * a gap with a confident fused verdict (p >= --conf) for a vocal class gets that tag -- including a plain
    [pause] that Gemini confirmed was a breath
  * a gap confidently found to hold speech (whispering / normal speech) drops the window: its transcript is
    missing words
  * unconfirmed [breathing] / [oral sounds] become [pause Ns] (their Qwen precision is below the plurality line)
  * [moaning] stays (96% precise); pauses stay

Reads v1 shards from the Hub (aoxo/t2a-speech-v2/t3v2), writes t3v11 shards, pushes each, deletes local copies
of what it no longer needs.

  python retag_windows_v11.py --fused D:\\t2a\\fusion\\fused_vocal_all.jsonl --out D:\\t2a\\t3v11 --push aoxo/t2a-speech-v2
"""
from __future__ import annotations
import argparse, bisect, json, os, re, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text2asmr.data.segment import split_alignment

TAG = re.compile(r"\[([a-z ]+?)(?: [\d.]+s)?\]")
VOCAL = {"breathing", "oral sounds", "moaning"}
SPEECH = {"whispering", "normal speech"}
PACE_S = 0.25
_pl = threading.Lock(); _next = [0.0]


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def pace():
    with _pl:
        now = time.time(); w = _next[0] - now; _next[0] = max(now, _next[0]) + PACE_S
    if w > 0: time.sleep(w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--push", default="aoxo/t2a-speech-v2")
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--emit-only", action="store_true",
                    help="write judge_gaps.jsonl (the unconfirmed vocal-tag gaps) and stop; judge them, re-fuse, rerun")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); a.out.mkdir(parents=True, exist_ok=True); scratch = a.out / "dl"

    # fused verdicts per recording: sorted gap-clip starts (ms) -> (class, confidence)
    fused: dict[str, list] = defaultdict(list)
    for l in open(a.fused, encoding="utf-8"):
        r = json.loads(l)
        src, ms = r["uid"].rsplit("_", 1)
        fused[src].append((int(ms), r["fused"], float(r["conf"])))
    for v in fused.values(): v.sort()
    log(f"fused verdicts for {len(fused)} recordings")

    man = [json.loads(l) for l in open(hf_hub_download(a.push, "manifests/speech_windows_v2.jsonl", repo_type="dataset",
                                                        local_dir=str(scratch)), encoding="utf-8")]
    win = {r["uid"]: r for r in man}
    by_rec = defaultdict(list)
    for r in man: by_rec[(r["repo"], r["source"])].append(r["uid"])
    log(f"{len(man)} windows in {len(by_rec)} recordings")

    # ---- per recording: re-derive gaps, decide each tag ----
    new_text, dropped, stats = {}, set(), Counter()

    def decide(src, g0, g1, old, orig):
        vs = fused.get(src, [])
        i = bisect.bisect_left(vs, (int(g0 * 1000) - 50,))
        best = None
        while i < len(vs) and vs[i][0] < g1 * 1000:
            if best is None or vs[i][2] > best[2]: best = vs[i]
            i += 1
        pause = f"[pause {max(0.5, round((g1 - g0) * 2) / 2):.1f}s]"
        if best and best[2] >= a.conf:
            if best[1] in VOCAL: return f"[{best[1]}]", f"confirmed {best[1]}" + ("" if old == best[1] else f" (was {old})")
            if best[1] in SPEECH: return None, "speech in gap"
        if old in ("breathing", "oral sounds"): return pause, f"unconfirmed {old} -> pause"
        return orig, f"kept {old}"                      # pauses and [moaning] (96% precise) stay exactly as written

    gap_cache = a.out / "gaps.jsonl"                    # window uid -> gaps; saves re-fetching 18k alignments
    cached = {}
    if gap_cache.exists():
        for l in open(gap_cache, encoding="utf-8"):
            g = json.loads(l); cached[g["uid"]] = [tuple(x) for x in g["gaps"]]
        log(f"gap cache: {len(cached)} windows")
    gap_rows = []

    def do(item):
        (repo, src), uids = item
        if uids and all(u in cached for u in uids):
            return src, apply(repo, src, {u: cached[u] for u in uids})
        for i in range(5):
            try:
                pace(); p = hf_hub_download(repo, src + ".json", repo_type="dataset", local_dir=str(scratch)); break
            except Exception:
                time.sleep(20 * 2 ** i)
        else:
            return src, None
        align = json.load(open(p, encoding="utf-8"))
        try: os.remove(p)
        except OSError: pass
        phrases = [s for s in split_alignment(align, src) if s.kind == "speech"]
        gm = {}
        for u in uids:
            w = win[u]
            ph = [s for s in phrases if s.start >= w["start"] - 0.01 and s.end <= w["end"] + 0.01]
            gm[u] = [(round(x.end, 3), round(y.start, 3)) for x, y in zip(ph, ph[1:]) if y.start - x.end >= 0.7]
            gap_rows.append({"uid": u, "gaps": gm[u]})
        return src, apply(repo, src, gm)

    def apply(repo, src, gm):
        out, need = {}, []
        for u, gaps in gm.items():
            w = win[u]
            tags = list(TAG.finditer(w["text"]))
            if len(gaps) != len(tags):
                out[u] = ("mismatch", w["text"], []); continue
            pieces, last, notes, drop = [], 0, [], False
            for m, (g0, g1) in zip(tags, gaps):
                t, note = decide(src, g0, g1, m.group(1), m.group(0))
                notes.append(note)
                if m.group(1) in VOCAL and not note.startswith("confirmed"):
                    need.append({"uid": f"{src}_{int(g0 * 1000):09d}", "qwen": m.group(1), "repo": repo,
                                 "start": round(g0, 3), "end": round(g1, 3)})
                if t is None: drop = True; break
                pieces += [w["text"][last:m.start()], t]; last = m.end()
            if drop: out[u] = ("drop", None, notes); continue
            pieces.append(w["text"][last:])
            out[u] = ("ok", "".join(pieces), notes)
        return src, (out, need)

    t0 = time.time(); done = 0; to_judge = []
    with ThreadPoolExecutor(a.workers) as ex:
        for src, res in ex.map(do, by_rec.items()):
            done += 1
            if res is None: stats["alignment unavailable"] += 1; continue
            out, need = res; to_judge += need
            for u, (kind, text, notes) in out.items():
                stats[kind] += 1
                for n in notes: stats[n] += 1
                if kind == "drop": dropped.add(u)
                elif kind == "ok": new_text[u] = text
            if done % 500 == 0:
                log(f"  {done}/{len(by_rec)} recordings ({(time.time() - t0) / 60:.0f} min)")
    if gap_rows:
        with open(gap_cache, "a", encoding="utf-8") as fh: fh.write("".join(json.dumps(g) + "\n" for g in gap_rows))
    log("tag decisions: " + ", ".join(f"{k}={v}" for k, v in stats.most_common()))
    seen_u = set(); uniq = [g for g in to_judge if not (g["uid"] in seen_u or seen_u.add(g["uid"]))]
    (a.out / "judge_gaps.jsonl").write_text("".join(json.dumps(g) + "\n" for g in uniq), encoding="utf-8")
    log(f"{len(uniq)} unconfirmed vocal-tag gaps -> {a.out / 'judge_gaps.jsonl'}")
    if a.emit_only: log("EMIT_DONE"); return 0

    # ---- rewrite shards: stream v1 shards from the Hub, write v1.1, push, delete ----
    shards = sorted(f for f in api.list_repo_files(a.push, repo_type="dataset") if re.fullmatch(r"t3v2/shard_\d+\.pt", f))
    kept = changed = 0
    for f in shards:
        p = hf_hub_download(a.push, f, repo_type="dataset", local_dir=str(scratch))
        rows = torch.load(p, weights_only=False); os.remove(p)
        out = []
        for r in rows:
            if r["uid"] in dropped: continue
            t = new_text.get(r["uid"])
            if t is not None and t != r["text"]: r["text"] = t; changed += 1
            out.append(r)
        kept += len(out)
        dst = a.out / Path(f).name
        torch.save(out, dst)
    # one commit per 40 shards: a commit per shard hit the Hub's hourly commit limit
    from huggingface_hub import CommitOperationAdd
    files = sorted(a.out.glob("shard_*.pt"))
    for b in range(0, len(files), 40):
        ops = [CommitOperationAdd(f"t3v11/{p.name}", str(p)) for p in files[b:b + 40]]
        for i in range(6):
            try:
                api.create_commit(a.push, repo_type="dataset", operations=ops, commit_message=f"t3 v1.1 retag shards {b}-{b + len(ops) - 1}"); break
            except Exception as e:
                log(f"  push retry {i}: {type(e).__name__} {str(e)[:80]}"); time.sleep(60 * (i + 1))
    log(f"RETAG_DONE windows kept {kept}, text changed {changed}, dropped {len(dropped)} -> {a.out} and {a.push}/t3v11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
