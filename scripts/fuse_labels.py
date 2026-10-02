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

LABELS_V1 = ["breathing", "kissing", "oral sounds", "moaning", "whispering", "normal speech", "tapping", "scratching",
             "crinkling", "brushing", "liquid", "microphone touching", "sticky", "fabric rustling", "paper rustling",
             "cutting", "background music", "silence / room tone", "something else"]
#: "spraying" joined 2026-10-02 (the user heard a spray bottle in a "liquid" chapter clip). Every vote remembers the
#: menu it was offered; votes cast before that (no "menu" field) abstain on it rather than counting as "no spray".
LABELS = LABELS_V1[:11] + ["spraying"] + LABELS_V1[11:]
MENU: dict = {}                                       # (uid, labeller) -> labels it was offered
V7 = ["breathing", "crinkling", "cutting", "fabric rustling", "microphone touching", "moaning", "oral sounds",
      "paper rustling", "scratching", "sticky", "tapping"]
CHAPTER = {"tapping", "brushing", "scratching", "liquid", "spraying", "microphone touching", "crinkling", "sticky",
           "fabric rustling", "paper rustling", "cutting", "oral sounds"}
COVERS = {"pipeline": set(V7), "chapter": CHAPTER, "clap": {"breathing", "moaning", "oral sounds"}, "energy": {"silence / room tone"},
          "ast": {"breathing", "oral sounds", "moaning", "whispering", "normal speech", "tapping", "scratching", "crinkling",
                  "liquid", "spraying", "fabric rustling", "paper rustling", "cutting", "background music",
                  "silence / room tone", "microphone touching"}}


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
            u, l = r.get("clip") or r["uid"], name.replace("-3.1-pro", "").replace("-v2.6-flash", "")
            votes[u][l] = set(r["labels"])
            if r.get("menu"): MENU[(u, l)] = set(r["menu"])
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
                if v is not None and lab in MENU.get((u, l), LABELS_V1): cov[i] = True; yes[i] = lab in v
            if cov.any(): A[lab][l] = (cov, yes)
    return A


def fit(A, humans, has_human, human_prior=10.0, unlabelled_weight=1.0, iters=30, shrink=0.0, pooled=None, crowd_prior=3.0):
    """Per class: prior + (sensitivity, specificity) per labeller, by EM. People get Beta pseudo-counts that say
    "reliable" (sens 0.8, spec 0.95, strength --human-prior); machines start flat. Clips no person labelled count
    `unlabelled_weight` toward the rates: Gemini and MiMo make correlated mistakes, and with full weight their
    agreement on 8k unlabelled clips swamps the people (precision fell 54% -> 32%).
    shrink > 0: a machine's rates on a class are pulled toward its rates over all classes (`pooled`), with that
    many pseudo-counts, so a class with a handful of human-labelled positives cannot learn a wild sensitivity."""
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
                    # Ear Check site listeners ("crowd:<name>") are strangers: a weak prior, so trust comes from the
                    # control clips and agreement, not from being a person
                    k = crowd_prior if l.startswith("crowd:") else human_prior; tp += 0.8 * k; fn += 0.2 * k; tn += 0.95 * k; fp += 0.05 * k
                elif shrink and pooled and l in pooled:
                    se0, sp0 = pooled[l]; tp += se0 * shrink; fn += (1 - se0) * shrink; tn += sp0 * shrink; fp += (1 - sp0) * shrink
                rates[l] = ((tp + 1) / (tp + fn + 2), (tn + 1) / (tn + fp + 2))
            post = posterior(L, prior, rates, n)
        model[lab] = (prior, rates)
    return model


def pooled_rates(model, humans):
    """Each machine's sensitivity / specificity over all classes, weighted by how many positives / negatives a
    class has (its prior)."""
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for prior, rates in model.values():
        for l, (se, sp) in rates.items():
            if l in humans: continue
            a = acc[l]; a[0] += prior * se; a[1] += prior; a[2] += (1 - prior) * sp; a[3] += 1 - prior
    return {l: (a[0] / a[1], a[2] / a[3]) for l, a in acc.items() if a[1] > 0 and a[3] > 0}


def fit_shrunk(A, humans, has_human, human_prior, unlabelled_weight, shrink, crowd_prior=3.0):
    m = fit(A, humans, has_human, human_prior, unlabelled_weight, crowd_prior=crowd_prior)
    if not shrink: return m
    return fit(A, humans, has_human, human_prior, unlabelled_weight, shrink=shrink, pooled=pooled_rates(m, humans), crowd_prior=crowd_prior)


def posterior(L, prior, rates, n):
    """A class nobody was asked about (e.g. a vote set from before it joined the menu) keeps its prior."""
    lo = np.full(n, np.log(prior / (1 - prior)))
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
    ap.add_argument("--crowd-prior", type=float, default=3.0, help="reliability pseudo-counts for crowd:<name> listeners (site labels)")
    ap.add_argument("--shrink", type=float, default=20.0, help="pseudo-counts pulling a machine's per-class rates to its overall rates (CV: precision vs Ryyan 38% -> 44%, F1 flat)")
    a = ap.parse_args()
    votes = load_votes(a.pool, a.votes)
    humans = set()
    for f in a.humans:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("uid") or r.get("skipped"): continue
            who = r.get("who") or f.stem; humans.add(who)
            votes[r["uid"]][who] = set(r["labels"]) & set(LABELS)
            if r.get("menu"): MENU[(r["uid"], who)] = set(r["menu"])
    labellers = sorted({l for v in votes.values() for l in v}); uids = sorted(votes)
    print(f"{len(uids)} clips; people: {sorted(humans)} ({sum(any(h in votes[u] for h in humans) for u in uids)} clips); "
          f"machines: {[l for l in labellers if l not in humans]}")

    if a.cv_against:
        who = a.cv_against; A = [u for u in uids if who in votes[u]]; random.Random(0).shuffle(A)
        truth = {u: votes[u][who] for u in A}; P = {}; ix = {u: i for i, u in enumerate(uids)}
        for k in range(5):
            test = set(A[k::5]); saved = {u: votes[u].pop(who) for u in test}
            AR = arrays(votes, uids, labellers); hh = np.array([any(h in votes[u] for h in humans) for u in uids])
            m = fit_shrunk(AR, humans, hh, a.human_prior, a.unlabelled_weight, a.shrink, a.crowd_prior)
            post = {lab: posterior(AR[lab], m[lab][0], m[lab][1], len(uids)) for lab in LABELS}
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
    m = fit_shrunk(AR, humans, hh, a.human_prior, a.unlabelled_weight, a.shrink, a.crowd_prior)
    post = {lab: posterior(AR[lab], m[lab][0], m[lab][1], len(uids)) for lab in LABELS}
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
