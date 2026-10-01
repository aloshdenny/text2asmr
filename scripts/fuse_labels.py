#!/usr/bin/env python3
"""Large-scale weak supervision, step 2: fuse many noisy labellers into one probabilistic label per clip and class.

Per class, every labeller has a sensitivity (says X when X is there) and a specificity (stays quiet when it is
not), and the class has a prior; EM over the whole pool fits them (Dawid-Skene, one binary model per class).
People are labellers too, not ground truth: two careful listeners agree on only ~1/3 of physical-sound labels
(Jaccard 0.33 on 30 clips) and adi's silence/music labels were wrong (Alita agreed on 1/11 and 0/9). Each person
starts from a strong reliability prior (--human-prior pseudo-counts at sensitivity 0.8 / specificity 0.95) and EM
learns where they are actually reliable. A labeller only votes on the classes it covers; elsewhere it abstains.
Output: per clip and class, P(present), plus every labeller's learned rates.

Measured on adi's 150 clips (5-fold CV, rates from the other folds): fused F1 0.47 / precision 54% vs Gemini
alone 0.41 / 35% and majority vote 0.38. --cv repeats that check on the current anchors.

  python fuse_labels.py --pool D:\\t2a\\pool --humans user_101.jsonl adi_150.jsonl alita_48.jsonl --out fused.jsonl [--cv-against adi]
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
CHAPTER = {"tapping", "brushing", "scratching", "liquid", "microphone touching", "crinkling", "sticky", "fabric rustling",
           "paper rustling", "cutting", "oral sounds"}
COVERS = {"pipeline": set(V7), "chapter": CHAPTER, "clap": {"breathing", "moaning", "oral sounds"}, "energy": {"silence / room tone"},
          "ast": {"breathing", "oral sounds", "moaning", "whispering", "normal speech", "tapping", "scratching", "crinkling",
                  "liquid", "fabric rustling", "paper rustling", "cutting", "background music", "silence / room tone",
                  "microphone touching"}}


def load_votes(pool: Path, extra: list[Path]) -> dict[str, dict[str, set]]:
    """uid -> labeller -> set of labels it said (missing labeller = abstain on every class)."""
    votes: dict = defaultdict(dict)
    for f in [x for x in extra if x.stem == "pool"]:                  # anchor sets' own pipeline labels
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); lab = r.get("label")
            if lab in (None, "uniform"): continue                     # drawn at random: the pipeline said nothing
            votes[r["uid"]]["pipeline"] = {lab} if lab in V7 else set()
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


def arrays(votes, uids, labellers):
    """Per class and labeller: covered (it voted on this clip and covers the class) and said-yes, as bool arrays."""
    A = {}
    for lab in LABELS:
        A[lab] = {}
        for l in labellers:
            cov = np.zeros(len(uids), bool); yes = np.zeros(len(uids), bool)
            if l in COVERS and lab not in COVERS[l]: continue
            for i, u in enumerate(uids):
                v = votes[u].get(l)
                if v is not None: cov[i] = True; yes[i] = lab in v
            if cov.any(): A[lab][l] = (cov, yes)
    return A


def fit(A, humans, has_human, human_prior=10.0, unlabelled_weight=1.0, iters=30):
    """Per class: prior + (sensitivity, specificity) per labeller, by EM. People get Beta pseudo-counts that say
    "reliable" (sens 0.8, spec 0.95, strength --human-prior); machines start flat. Clips no person labelled count
    `unlabelled_weight` toward the rates: Gemini and MiMo make correlated mistakes, and with full weight their
    agreement on 8k unlabelled clips swamps the people (precision fell 54% -> 32%)."""
    w = np.where(has_human, 1.0, unlabelled_weight)
    model = {}
    for lab in LABELS:
        L = A[lab]; n = len(w)
        hy = sum((L[h][1] & L[h][0]).astype(float) for h in humans if h in L) if any(h in L for h in humans) else np.zeros(n)
        hc = sum(L[h][0].astype(float) for h in humans if h in L) if any(h in L for h in humans) else np.zeros(n)
        my = sum((v[1] & v[0]).astype(float) for k, v in L.items() if k not in humans) if L else np.zeros(n)
        mc = sum(v[0].astype(float) for k, v in L.items() if k not in humans) if L else np.zeros(n)
        post = np.where(hc > 0, (hy + 0.5) / (hc + 1), np.where(mc > 0, (my + 0.5) / (mc + 1), 0.1))
        for _ in range(iters):
            prior = (np.sum(w * post) + 1) / (np.sum(w) + 2)
            rates = {}
            for l, (cov, yes) in L.items():
                c = w * cov
                tp = np.sum(c * post * yes); fp = np.sum(c * (1 - post) * yes)
                fn = np.sum(c * post * ~yes); tn = np.sum(c * (1 - post) * ~yes)
                if l in humans:
                    k = human_prior; tp += 0.8 * k; fn += 0.2 * k; tn += 0.95 * k; fp += 0.05 * k
                rates[l] = ((tp + 1) / (tp + fn + 2), (tn + 1) / (tn + fp + 2))
            post = posterior(L, prior, rates)
        model[lab] = (prior, rates)
    return model


def posterior(L, prior, rates):
    lo = np.full(len(next(iter(L.values()))[0]) if L else 0, np.log(prior / (1 - prior)))
    for l, (se, sp) in rates.items():
        cov, yes = L[l]
        lo += cov * np.where(yes, np.log(se / (1 - sp)), np.log((1 - se) / sp))
    return 1 / (1 + np.exp(-np.clip(lo, -30, 30)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True, help="label_pool.py out dir (labeller jsonls + pool.jsonl)")
    ap.add_argument("--votes", type=Path, nargs="*", default=[], help="more labeller files (anchors' votes, ast, energy, ...)")
    ap.add_argument("--humans", type=Path, nargs="+", required=True, help="people's label files (read_label_html format, with uid)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--human-prior", type=float, default=10.0)
    ap.add_argument("--cv-against", default="", help="5-fold: hide this person's labels on a fold, score the fusion against them")
    ap.add_argument("--unlabelled-weight", type=float, default=0.0, help="weight of clips no person labelled, in the rates")
    a = ap.parse_args()
    votes = load_votes(a.pool, a.votes)
    humans = set()
    for f in a.humans:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("uid") or r.get("skipped"): continue
            who = r.get("who") or f.stem; humans.add(who)
            votes[r["uid"]][who] = set(r["labels"]) & set(LABELS)
    labellers = sorted({l for v in votes.values() for l in v}); uids = sorted(votes)
    print(f"{len(uids)} clips; people: {sorted(humans)} ({sum(any(h in votes[u] for h in humans) for u in uids)} clips); "
          f"machines: {[l for l in labellers if l not in humans]}")

    if a.cv_against:
        who = a.cv_against; A = [u for u in uids if who in votes[u]]; random.Random(0).shuffle(A)
        truth = {u: votes[u][who] for u in A}; P = {}; ix = {u: i for i, u in enumerate(uids)}
        for k in range(5):
            test = set(A[k::5]); saved = {u: votes[u].pop(who) for u in test}
            AR = arrays(votes, uids, labellers); hh = np.array([any(h in votes[u] for h in humans) for u in uids])
            m = fit(AR, humans, hh, a.human_prior, a.unlabelled_weight)
            post = {lab: posterior(AR[lab], m[lab][0], m[lab][1]) for lab in LABELS}
            for u in test: P[u] = {lab for lab in LABELS if post[lab][ix[u]] > 0.5}
            for u, v in saved.items(): votes[u][who] = v
        def score(pred):
            i = sum(len(pred[u] & truth[u]) for u in A); n_p = sum(len(pred[u]) for u in A); n_h = sum(len(truth[u]) for u in A)
            pr, rc = i / max(n_p, 1), i / max(n_h, 1); return pr, rc, 2 * pr * rc / max(pr + rc, 1e-9)
        print(f"{'labeller':12} prec recall  F1   (vs {who}'s labels on {len(A)} clips, held out)")
        print(f"{'FUSED':12} {score(P)[0]:4.0%} {score(P)[1]:6.0%} {score(P)[2]:5.2f}")
        for l in labellers:
            if l == who or sum(1 for u in A if l in votes[u]) < 20: continue
            pr, rc, f1 = score({u: (votes[u].get(l) or set()) for u in A}); print(f"{l:12} {pr:4.0%} {rc:6.0%} {f1:5.2f}")

    AR = arrays(votes, uids, labellers); hh = np.array([any(h in votes[u] for h in humans) for u in uids])
    m = fit(AR, humans, hh, a.human_prior, a.unlabelled_weight)
    post = {lab: posterior(AR[lab], m[lab][0], m[lab][1]) for lab in LABELS}
    with open(a.out, "w", encoding="utf-8") as fh:
        for i, u in enumerate(uids):
            p = {lab: round(float(post[lab][i]), 3) for lab in LABELS}
            fh.write(json.dumps({"uid": u, "probs": p, "labels": [k for k, v in p.items() if v > 0.5],
                                 "people": sorted(h for h in humans if h in votes[u])}) + "\n")
    rates = {lab: {"prior": round(m[lab][0], 3), **{l: {"sens": round(se, 2), "spec": round(sp, 2)} for l, (se, sp) in m[lab][1].items()}}
             for lab in LABELS}
    a.out.with_suffix(".rates.json").write_text(json.dumps(rates, indent=1))
    print(f"wrote {a.out} and {a.out.with_suffix('.rates.json')}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
