#!/usr/bin/env python3
"""Large-scale weak supervision, step 4b: a CLAP linear probe trained on the fused labels, applied to every training row.

The fusion (fuse_labels.py) gives P(class) for the ~11k clips some machine or person labelled. A probe on frozen CLAP
v7 audio embeddings, trained on those soft labels, carries them to the 254k index rows nobody labelled (the
label-model -> end-model step of weak supervision). Checked before use: out-of-fold (folds split by recording),
the probe is scored against each person's labels the same way fuse_labels.py --cv-against scores the fusion.

  python probe_fused.py --emb D:\\t2a\\emb --fused fused.jsonl --humans humans/*.jsonl --out D:\\t2a\\probe
Out: <out>/rows.jsonl (uid, src, index label, probe probs >= 0.05) for every index row, <out>/report.json.
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse_labels import LABELS


def rec_of(uid: str) -> str:
    """Recording id: folds must not split one recording (its windows sound alike)."""
    return uid.rsplit(":", 1)[0] if uid.startswith("yt:") else re.sub(r"_\d+$", "", uid)


def load_emb(d: Path, name: str):
    rows = [json.loads(l) for l in open(d / f"{name}.jsonl", encoding="utf-8")]
    E = np.fromfile(d / f"{name}.f16", dtype=np.float16).reshape(len(rows), -1).astype(np.float32)
    return rows, E


def train(X, Y, W, epochs=300, wd=1e-4, dev="cpu"):
    """Per-class logistic regression with soft targets (BCE), full batch."""
    X = torch.tensor(X, device=dev); Y = torch.tensor(Y, device=dev); W = torch.tensor(W, device=dev)
    lin = torch.nn.Linear(X.shape[1], Y.shape[1]).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=0)
    for _ in range(epochs):
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(lin(X * 10), Y, reduction="none").mean(1) * W).mean()
        loss = loss + wd * (lin.weight ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return lin.cpu().eval()


def predict(lin, X, bs=65536):
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(lin(torch.tensor(X[i:i + bs]) * 10)).numpy() for i in range(0, len(X), bs)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--humans", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--mine", type=int, default=0, help="per rare class, this many unlabelled YouTube windows to label next")
    ap.add_argument("--mine-per-rec", type=int, default=3)
    ap.add_argument("--mine-classes", nargs="*", default=["microphone touching", "sticky", "fabric rustling", "paper rustling",
                                                          "cutting", "brushing", "liquid", "spraying", "scratching"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    F = {json.loads(l)["uid"]: json.loads(l)["probs"] for l in open(a.fused, encoding="utf-8")}
    sets = {name: load_emb(a.emb, name) for name in ("clips", "indomain", "yt") if (a.emb / f"{name}.jsonl").exists()}
    where = {}                         # a labelled clip's own audio first (what the labellers heard), else its mel row
    for name, (rows, _) in sets.items():
        for i, r in enumerate(rows):
            if r["uid"] in F: where.setdefault(r["uid"], (name, i))
    uids = sorted(where); CE = np.stack([sets[n][1][i] for n, i in (where[u] for u in uids)])
    print(f"{len(uids)} fused clips with embeddings: " + ", ".join(f"{k} {v}" for k, v in Counter(n for n, _ in where.values()).items()))
    Y = np.array([[F[u][lab] for lab in LABELS] for u in uids], np.float32)
    H = defaultdict(dict)
    for f in a.humans:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r.get("uid") and not r.get("skipped"): H[r["uid"]][r.get("who") or f.stem] = set(r["labels"]) & set(LABELS)
    # people's own clips are the check, so their fused labels (which include them) never train the fold they are scored in
    recs = sorted({rec_of(u) for u in uids}); rng = np.random.default_rng(0); rng.shuffle(recs)
    fold_of_rec = {r: i % a.folds for i, r in enumerate(recs)}; fold = np.array([fold_of_rec[rec_of(u)] for u in uids])
    oof = np.zeros_like(Y)
    for k in range(a.folds):
        tr = fold != k
        lin = train(CE[tr], Y[tr], np.ones(tr.sum(), np.float32), wd=a.wd)
        oof[~tr] = predict(lin, CE[~tr])
    ix = {u: i for i, u in enumerate(uids)}
    report = {"clips": len(uids), "vs_people": {}, "per_class": {}}
    for who in sorted({w for h in H.values() for w in h}):
        A = [u for u, h in H.items() if who in h and u in ix]
        if len(A) < 20: continue
        def score(pred):
            i = sum(len(pred[u] & H[u][who]) for u in A); n_p = sum(len(pred[u]) for u in A); n_h = sum(len(H[u][who]) for u in A)
            p, r = i / max(n_p, 1), i / max(n_h, 1); return round(p, 3), round(r, 3), round(2 * p * r / max(p + r, 1e-9), 3)
        probe = {u: {lab for j, lab in enumerate(LABELS) if oof[ix[u], j] > 0.5} for u in A}
        fused = {u: {lab for lab in LABELS if F[u][lab] > 0.5} for u in A}
        report["vs_people"][who] = {"n": len(A), "probe_oof": score(probe), "fused_incl_person": score(fused)}
        # the fair comparison for the fusion is fuse_labels.py --cv-against (person held out); this one is its ceiling
        print(f"vs {who:6} ({len(A)} clips)  probe P/R/F1 {score(probe)}   fused with {who}'s own votes {score(fused)}")
    from sklearn.metrics import average_precision_score
    print(f"{'class':20} {'fused>0.5':>9} {'AP(probe vs fused>0.5)':>22}")
    for j, lab in enumerate(LABELS):
        t = Y[:, j] > 0.5
        ap_ = float(average_precision_score(t, oof[:, j])) if 0 < t.sum() < len(t) else float("nan")
        report["per_class"][lab] = {"fused_pos": int(t.sum()), "ap_oof": round(ap_, 3)}
        print(f"{lab:20} {int(t.sum()):9} {ap_:22.3f}")
    lin = train(CE, Y, np.ones(len(Y), np.float32), wd=a.wd)
    torch.save(lin.state_dict(), a.out / "probe.pt")
    mine = defaultdict(list)
    with open(a.out / "rows.jsonl", "w", encoding="utf-8") as fh:
        for name in ("indomain", "yt"):
            if name not in sets: continue
            rows, E = sets[name]; P = predict(lin, E)
            for r, p in zip(rows, P):
                fh.write(json.dumps({"uid": r["uid"], "set": name, "label": r.get("label"), "split": r.get("split"),
                                     "p": {lab: round(float(v), 3) for lab, v in zip(LABELS, p) if v >= 0.05}}) + "\n")
            if name == "yt":
                for j, lab in enumerate(LABELS):
                    if lab in a.mine_classes: mine[lab] = [(float(P[i, j]), rows[i]) for i in np.argsort(-P[:, j])[:20000] if rows[i]["uid"] not in F]
            print(f"{name}: {len(rows)} rows scored")
    # active weak supervision: the unlabelled windows the probe ranks highest for a rare class, a few per recording,
    # are where new labels find positives (random chapter windows: 4-20% positive)
    with open(a.out / "mine.jsonl", "w", encoding="utf-8") as fh:
        taken = set()
        for lab, ranked in mine.items():
            if not ranked or not a.mine: continue
            per_rec, n = Counter(), 0
            for pv, r in ranked:
                rec = rec_of(r["uid"])
                if r["uid"] in taken or per_rec[rec] >= a.mine_per_rec: continue
                per_rec[rec] += 1; taken.add(r["uid"]); n += 1
                fh.write(json.dumps({"uid": r["uid"], "target": lab, "probe_p": round(pv, 3), "chapter_label": r.get("label")}) + "\n")
                if n >= a.mine: break
            print(f"mine {lab:20} {n} windows, probe P {ranked[0][0]:.2f} .. {pv:.2f}")
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    print("PROBE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
