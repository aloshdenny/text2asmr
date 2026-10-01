#!/usr/bin/env python3
"""Large-scale weak supervision, step 2: fuse many noisy labellers into one probabilistic label per clip and class.

Per class, every labeller has a sensitivity (says X when X is there) and a specificity (stays quiet when it is
not), and the class has a prior. They start from the human-labelled anchor clips (add-one smoothed) and are
refined by EM over the whole pool (Dawid-Skene, one binary model per class), with anchors held at the human
answer and their counts weighted by --anchor. A labeller only votes on the classes it covers; elsewhere it
abstains. Output: per clip and class, P(present).

Measured on adi's 150 clips (5-fold CV, rates from the other folds): fused F1 0.47 / precision 54% vs Gemini
alone 0.41 / 35% and majority vote 0.38. --cv repeats that check on the current anchors.

  python fuse_labels.py --pool D:\\t2a\\pool --anchors user_101.jsonl adi_150.jsonl --out D:\\t2a\\pool\\fused.jsonl
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = ["breathing", "kissing", "oral sounds", "moaning", "whispering", "normal speech", "tapping", "scratching",
          "crinkling", "brushing", "liquid", "microphone touching", "sticky", "fabric rustling", "paper rustling",
          "cutting", "background music", "silence / room tone", "something else"]
V7 = ["breathing", "crinkling", "cutting", "fabric rustling", "microphone touching", "moaning", "oral sounds",
      "paper rustling", "scratching", "sticky", "tapping"]
COVERS = {"pipeline": set(V7), "clap": {"breathing", "moaning", "oral sounds"}, "energy": {"silence / room tone"},
          "ast": {"breathing", "oral sounds", "moaning", "whispering", "normal speech", "tapping", "scratching", "crinkling",
                  "liquid", "fabric rustling", "paper rustling", "cutting", "background music", "silence / room tone",
                  "microphone touching"}}


def load_votes(pool: Path, extra: list[Path]) -> dict[str, dict[str, set]]:
    """uid -> labeller -> set of labels it said (missing labeller = abstain on every class)."""
    votes: dict = defaultdict(dict)
    for f in list(pool.glob("*.jsonl")) + extra:
        name = f.stem
        if name in ("pool", "fused") or name.startswith("fused"): continue
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if str(r.get("raw", "")).startswith("ERROR"): continue
            votes[r.get("clip") or r["uid"]][name.replace("-3.1-pro", "").replace("-v2.6-flash", "")] = set(r["labels"])
    pf = pool / "pool.jsonl"
    if pf.exists():
        for l in open(pf, encoding="utf-8"):
            r = json.loads(l)
            votes[r["uid"]]["pipeline"] = set() if r["label"] == "__bg__" else {r["label"]}
    return votes


def says(votes, uid, l, lab):
    v = votes[uid].get(l)
    if v is None or (l in COVERS and lab not in COVERS[l]): return None
    return lab in v


def fit(votes, uids, truth, labellers, anchor=5.0, iters=30):
    """Per class: prior + (sensitivity, specificity) per labeller; anchors fixed, EM over the rest."""
    model = {}
    for lab in LABELS:
        post = {u: (1.0 if lab in truth[u] else 0.0) for u in uids if u in truth}
        unl = [u for u in uids if u not in truth]
        for u in unl:                                                 # start: fraction of covering labellers saying yes
            vs = [says(votes, u, l, lab) for l in labellers]; vs = [v for v in vs if v is not None]
            post[u] = (sum(vs) + 0.5) / (len(vs) + 1) if vs else 0.1
        for _ in range(iters):
            w = {u: (anchor if u in truth else 1.0) for u in post}
            prior = (sum(w[u] * post[u] for u in post) + 1) / (sum(w.values()) + 2)
            rates = {}
            for l in labellers:
                tp = fn = tn = fp = 1e-9
                for u in post:
                    v = says(votes, u, l, lab)
                    if v is None: continue
                    p = post[u]
                    if v: tp += w[u] * p; fp += w[u] * (1 - p)
                    else: fn += w[u] * p; tn += w[u] * (1 - p)
                rates[l] = ((tp + 1) / (tp + fn + 2), (tn + 1) / (tn + fp + 2))
            for u in unl: post[u] = posterior(votes, u, lab, prior, rates)
        model[lab] = (prior, rates)
    return model


def posterior(votes, u, lab, prior, rates):
    lo = np.log(prior / (1 - prior))
    for l, (se, sp) in rates.items():
        v = says(votes, u, l, lab)
        if v is None: continue
        lo += np.log(se / (1 - sp)) if v else np.log((1 - se) / sp)
    return float(1 / (1 + np.exp(-np.clip(lo, -30, 30))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True, help="label_pool.py out dir (labeller jsonls + pool.jsonl)")
    ap.add_argument("--votes", type=Path, nargs="*", default=[], help="more labeller files (anchors' votes, ast, energy, ...)")
    ap.add_argument("--anchors", type=Path, nargs="+", required=True, help="human label files (read_label_html format, with uid)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--anchor", type=float, default=5.0)
    ap.add_argument("--cv", action="store_true", help="5-fold CV on the anchors: fused vs each labeller")
    a = ap.parse_args()
    truth = defaultdict(set)
    for f in a.anchors:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r.get("uid") and not r.get("skipped"): truth[r["uid"]] |= set(r["labels"]) & set(LABELS)
    votes = load_votes(a.pool, a.votes)
    labellers = sorted({l for v in votes.values() for l in v})
    uids = sorted(votes)
    anchored = [u for u in truth if u in votes]
    print(f"{len(uids)} clips, {len(anchored)} with human labels; labellers: {labellers}")

    if a.cv:
        rng = random.Random(0); A = anchored[:]; rng.shuffle(A); P = {}
        for k in range(5):
            test = set(A[k::5]); tr_truth = {u: truth[u] for u in A if u not in test}
            m = fit(votes, uids, tr_truth, labellers, a.anchor)
            for u in test: P[u] = {lab for lab in LABELS if posterior(votes, u, lab, *m[lab]) > 0.5}
        def score(pred):
            i = sum(len(pred[u] & truth[u]) for u in A); n_p = sum(len(pred[u]) for u in A); n_h = sum(len(truth[u]) for u in A)
            pr, rc = i / max(n_p, 1), i / max(n_h, 1); return pr, rc, 2 * pr * rc / max(pr + rc, 1e-9)
        print(f"{'labeller':12} prec recall  F1   (vs {len(A)} human-labelled clips)")
        print(f"{'FUSED':12} {score(P)[0]:4.0%} {score(P)[1]:6.0%} {score(P)[2]:5.2f}")
        for l in labellers:
            pl = {u: (votes[u].get(l) or set()) for u in A}
            if sum(1 for u in A if l in votes[u]) < 20: continue
            print(f"{l:12} {score(pl)[0]:4.0%} {score(pl)[1]:6.0%} {score(pl)[2]:5.2f}")

    m = fit(votes, uids, dict(truth), labellers, a.anchor)
    with open(a.out, "w", encoding="utf-8") as fh:
        for u in uids:
            p = {lab: round(posterior(votes, u, lab, *m[lab]), 3) if u not in truth else float(lab in truth[u]) for lab in LABELS}
            fh.write(json.dumps({"uid": u, "probs": p, "labels": [k for k, v in p.items() if v > 0.5], "human": u in truth}) + "\n")
    rates = {lab: {"prior": round(m[lab][0], 3), **{l: {"sens": round(se, 2), "spec": round(sp, 2)} for l, (se, sp) in m[lab][1].items()}}
             for lab in LABELS}
    a.out.with_suffix(".rates.json").write_text(json.dumps(rates, indent=1))
    print(f"wrote {a.out} and {a.out.with_suffix('.rates.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
