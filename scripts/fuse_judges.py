#!/usr/bin/env python3
"""Fuse noisy annotators into calibrated soft labels (Dawid-Skene EM, anchored by human-labelled calibration).

Why not "Gemini = positive": on human-labelled clips Gemini is right 88% on brushing but 30% on scratching and
0% on page turning. Treating its answer as truth teaches its mistakes. Instead every annotator gets a confusion
matrix P(says j | truly i):
  * judges scored on the human-labelled calibration set (class_gate.py --stage cutext) are *anchored* there:
    their measured counts are a strong Dirichlet prior, updated only lightly by EM
  * annotators nobody can calibrate directly -- the YouTube chapter title -- are *learned* by EM from how they
    co-vary with the anchored judges
The output is a posterior over the true class for every item: a soft label with an honest confidence, the
estimated precision of every annotator/class cell, and hard negatives (the annotator said c, the evidence says
it is almost certainly not c).

  python fuse_judges.py --calib gate_ext --items gclean/gemini_judged.jsonl --out fused_yt.jsonl
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OTHER = "other"


def load_calibration(d: Path, judge_files: dict[str, str]) -> tuple[list[str], dict[str, np.ndarray], list[str]]:
    """Human truth x judge answer counts from a class_gate cutext directory."""
    truth = {c["uid"]: c["label"] for c in json.load(open(d / "clips.json"))}
    classes = sorted(set(truth.values()))
    answers = sorted({OTHER} | set(classes))
    counts = {}
    for name, fn in judge_files.items():
        m = np.zeros((len(classes) + 1, len(answers)))          # rows: true classes + other (unobserved here)
        for l in open(d / fn):
            r = json.loads(l)
            if r.get("pred") is None or r["uid"] not in truth: continue
            a = r["pred"] if r["pred"] in answers else OTHER
            m[classes.index(truth[r["uid"]]), answers.index(a)] += 1
        counts[name] = m
    return classes, counts, answers


def load_calibration_jsonl(path: Path) -> tuple[list[str], dict[str, np.ndarray], list[str]]:
    """Calibration rows written by gemini_vocal_judge.py --mode calib: {truth, pred}."""
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("pred") is not None]
    classes = sorted({r["truth"] for r in rows if r["truth"] != OTHER})
    answers = sorted({OTHER} | set(classes) | {r["pred"] for r in rows})
    m = np.zeros((len(classes) + 1, len(answers)))
    for r in rows:
        i = classes.index(r["truth"]) if r["truth"] in classes else len(classes)
        m[i, answers.index(r["pred"])] += 1
    return classes, {"gemini": m}, answers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=Path, default=None, help="class_gate cutext work dir (human-labelled)")
    ap.add_argument("--calib-jsonl", type=Path, default=None, help="gemini_vocal_judge calib.jsonl {truth, pred}")
    ap.add_argument("--label-field", default="label", help="field holding the weak annotator's label (chapter / qwen)")
    ap.add_argument("--extra-classes", default="", help="true classes with no human calibration (e.g. moaning)")
    ap.add_argument("--gemini-file", default="judge_google_gemini-3.1-pro-preview.jsonl")
    ap.add_argument("--items", type=Path, required=True, help="gemini_judged.jsonl: label = chapter, pred = gemini")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--anchor", type=float, default=20.0, help="calibration counts weight vs EM evidence")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--confident", type=float, default=0.8)
    ap.add_argument("--negative", type=float, default=0.1)
    a = ap.parse_args()

    if a.calib_jsonl: classes, calib, answers = load_calibration_jsonl(a.calib_jsonl)
    else: classes, calib, answers = load_calibration(a.calib, {"gemini": a.gemini_file})
    # classes nobody has human labels for still exist as truths: flat (unanchored) rows, learned by EM
    for c in [x for x in a.extra_classes.split(",") if x and x not in classes]:
        classes.append(c)
        for k in calib: calib[k] = np.insert(calib[k], len(classes) - 1, 0.0, axis=0)
        if c not in answers:
            j = len(answers); answers.append(c)
            for k in calib: calib[k] = np.insert(calib[k], j, 0.0, axis=1)
    K = classes + [OTHER]                                      # true-class space
    A = {c: i for i, c in enumerate(answers)}
    items = []
    for l in open(a.items):
        r = json.loads(l)
        if r.get("pred") is None: continue
        g = r["pred"] if r["pred"] in A else OTHER
        items.append((r["uid"], r[a.label_field], g, r))
    chap_vals = sorted({c for _, c, _, _ in items})
    C = {c: i for i, c in enumerate(chap_vals)}
    n = len(items); nk = len(K)
    g_obs = np.array([A[g] for _, _, g, _ in items]); c_obs = np.array([C[c] for _, c, _, _ in items])
    print(f"{n} items; chapter labels {chap_vals}; true-class space {K}")

    # anchored prior for gemini: measured calibration counts, scaled; the unobserved 'other' row starts flat
    g_prior = calib["gemini"].copy()
    g_prior[-1, :] = 1.0
    g_prior[g_prior.sum(1) == 0] = 1.0                          # unanchored truths start flat
    g_prior = g_prior / g_prior.sum(1, keepdims=True) * a.anchor + 0.5
    g_conf = g_prior / g_prior.sum(1, keepdims=True)
    # chapter annotator starts believing itself 60%, the rest spread evenly
    ch_conf = np.full((nk, len(chap_vals)), 0.4 / max(1, len(chap_vals) - 1))
    for c, j in C.items():
        if c in K: ch_conf[K.index(c), :] = 0.4 / max(1, len(chap_vals) - 1); ch_conf[K.index(c), j] = 0.6
    ch_conf /= ch_conf.sum(1, keepdims=True)
    prior = np.full(nk, 1.0 / nk)

    for it in range(a.iters):
        post = prior[None, :] * g_conf[:, g_obs].T * ch_conf[:, c_obs].T      # (n, nk)
        post /= post.sum(1, keepdims=True)
        prior = post.mean(0)
        g_cnt = np.zeros_like(g_conf)
        for k in range(nk): g_cnt[k] = np.bincount(g_obs, weights=post[:, k], minlength=len(answers))
        g_conf = (g_cnt + g_prior) / (g_cnt + g_prior).sum(1, keepdims=True)  # evidence + anchor
        c_cnt = np.zeros_like(ch_conf)
        for k in range(nk): c_cnt[k] = np.bincount(c_obs, weights=post[:, k], minlength=len(chap_vals))
        ch_conf = (c_cnt + 0.5) / (c_cnt + 0.5).sum(1, keepdims=True)

    # estimated precision of each chapter label: P(true = c | chapter says c)
    print(f"\n{a.label_field} label precision (EM estimate) and what it really contains:")
    for c, j in C.items():
        m = c_obs == j
        dist = post[m].mean(0)
        top = ", ".join(f"{K[k]} {dist[k]:.0%}" for k in np.argsort(-dist)[:3])
        print(f"  '{c}' (n={m.sum()}): {top}")
    stats, rows = Counter(), []
    for i, (uid, chap, g, r) in enumerate(items):
        p = post[i]; k = int(p.argmax())
        row = {"uid": uid, "chapter": chap, "gemini": g, "fused": K[k], "conf": round(float(p[k]), 4),
               "p": {K[j]: round(float(p[j]), 4) for j in range(nk) if p[j] > 0.01}}
        if p[k] >= a.confident and K[k] != OTHER: stats[f"confident {K[k]}"] += 1
        if chap in K and p[K.index(chap)] <= a.negative:
            row["hard_negative_for"] = chap; stats[f"hard negative for {chap}"] += 1
        rows.append(row)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("\n" + "\n".join(f"  {k:32} {v}" for k, v in sorted(stats.items())))
    print(f"\nwrote {len(rows)} fused rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
