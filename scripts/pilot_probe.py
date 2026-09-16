#!/usr/bin/env python3
"""Agreement + linear-probe learnability of pilot Gemini-Pro labels vs the old flash labels (same clips)."""
import json, random, subprocess, sys, numpy as np, torch
from collections import Counter, defaultdict
from pathlib import Path
from transformers import ClapModel, ClapFeatureExtractor
SR = 48000; out = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/pilot")
rows = [json.loads(l) for l in open(out / "pilot_labels.jsonl")]
rows = {r["uid"]: r for r in rows if r.get("new_label") and not r.get("error")}.values(); rows = list(rows)
print(f"labeled clips: {len(rows)}")
agree = sum(r["new_label"] == r["label"] for r in rows); print(f"old(flash+context) vs new(pro, audio-only) exact agreement: {agree}/{len(rows)} = {100*agree/len(rows):.1f}%")
per = defaultdict(lambda: [0, 0]); conf = defaultdict(Counter)
for r in rows:
    per[r["label"]][1] += 1; conf[r["label"]][r["new_label"]] += 1
    if r["new_label"] == r["label"]: per[r["label"]][0] += 1
print(f"\n{'old label':18s} {'n':>4s} {'agree':>6s}  new-label distribution (top 4)")
for c, (a, n) in sorted(per.items(), key=lambda x: -x[1][1]):
    print(f"{c:18s} {n:4d} {100*a/n:5.1f}%  {conf[c].most_common(4)}")
print("\nnew label totals:", Counter(r["new_label"] for r in rows).most_common(30))

fe = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused"); m = ClapModel.from_pretrained("laion/clap-htsat-unfused").cuda().eval()
def wav(r):
    fp = out / "flac" / (r["uid"].replace("/", "__") + ".flac")
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", str(fp), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(b, dtype=np.float32)[: SR * 10]
feats = {}
with torch.no_grad():
    for i in range(0, len(rows), 64):
        ch = rows[i:i+64]; x = np.stack([fe(wav(r), sampling_rate=SR, return_tensors="np")["input_features"][0].reshape(1001, 64) for r in ch])
        e = m.audio_model(input_features=torch.from_numpy(x).unsqueeze(1).cuda(), is_longer=torch.zeros(len(ch),1,dtype=torch.bool,device="cuda")).pooler_output.float().cpu()
        for r, v in zip(ch, e): feats[r["uid"]] = v
rng = random.Random(0)
np.save(out / "feats.npy", {k: v.numpy() for k, v in feats.items()}, allow_pickle=True)
def probe(classes, key, name, n_rep=5, cap=None):
    accs = []
    for rep in range(n_rep):
        tr, ev = [], []
        for i, c in enumerate(classes):
            rs = [r for r in rows if r[key] == c]; rng.shuffle(rs); rs = rs[:cap] if cap else rs; cut = int(len(rs) * 0.75)
            tr += [(r, i) for r in rs[:cut]]; ev += [(r, i) for r in rs[cut:]]
        if len(ev) < 10: return
        Xtr = torch.stack([feats[r["uid"]] for r,_ in tr]).cuda(); Xev = torch.stack([feats[r["uid"]] for r,_ in ev]).cuda()
        ytr = torch.tensor([y for _,y in tr]).cuda(); yev = torch.tensor([y for _,y in ev]).cuda()
        mu, sd = Xtr.mean(0), Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xev=(Xev-mu)/sd
        W = torch.zeros(Xtr.shape[1], len(classes), device="cuda", requires_grad=True); b = torch.zeros(len(classes), device="cuda", requires_grad=True)
        opt = torch.optim.Adam([W,b], lr=1e-2)
        for _ in range(500):
            loss = torch.nn.functional.cross_entropy(Xtr@W+b, ytr) + 1e-3*(W**2).sum(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): accs.append(((Xev@W+b).argmax(1)==yev).float().mean().item())
    cnts = [min(cap, newc[c]) if cap else Counter(r[key] for r in rows)[c] for c in classes]; maj = max(cnts) / sum(cnts)
    print(f"{name:52s} [{key:9s}] acc={np.mean(accs):.3f}±{np.std(accs):.3f} majority={maj:.3f} chance={1/len(classes):.3f}")
phys = ["tapping","scratching","page turning","footsteps","brushing","liquid","crinkling","fabric rustling","cutting","clinking"]
newc = Counter(r["new_label"] for r in rows)
print()
for key in ("label", "new_label"):
    cls = [c for c in phys if (key == "label" or newc[c] >= 25)]
    probe(cls, key, f"{len(cls)}-way physical")
    for pair in (("tapping","page turning"), ("tapping","scratching"), ("whispering","tapping"), ("reject","tapping"), ("breathing heavy","kissing")):
        if key == "label" or all(newc[c] >= 20 for c in pair): probe(list(pair), key, " vs ".join(pair))

print("\n=== balanced probes on Pro (audio-only) labels ===")
vocal = ["reject", "whispering", "kissing", "mouth sounds"]
probe(vocal, "new_label", "4-way vocal (cap 100/class)", cap=100)
for pair in (("reject","whispering"), ("reject","kissing"), ("whispering","kissing"), ("whispering","mouth sounds"), ("kissing","mouth sounds")):
    probe(list(pair), "new_label", " vs ".join(pair) + " (cap 100)", cap=100)
probe(["reject","whispering","kissing"], "new_label", "3-way (cap 250)", cap=250)
print("\n=== same classes, OLD flash labels, balanced ===")
probe(vocal, "label", "4-way vocal (cap 100/class)", cap=100)
for pair in (("reject","whispering"), ("reject","kissing"), ("whispering","kissing")):
    probe(list(pair), "label", " vs ".join(pair) + " (cap 100)", cap=100)
