#!/usr/bin/env python3
"""Which audio model labels ASMR Board clips most like people do? A bake-off on clips the crowd has already answered.

Clips come from the site: every clip with two or more first answers (so people can be compared with each other),
topped up with single-answer clips from labellers whose quality status is "ok". Each model hears the same mp3 the
labellers heard and gets the site's label menu (titles and hints from label_options), with one of two prompts:

  menu       pick every label that applies (what the site asks people)
  checklist  answer yes/no for every label (forces the model to consider each class)

Scoring is against each person's answer: set F1 per (clip, person), averaged, plus per-class precision/recall
pooled over people. The same numbers for person-vs-person pairs are the ceiling: a model cannot be expected to
agree with people more than they agree with each other.

  python bakeoff_site.py fetch --out DIR --n 150
  python bakeoff_site.py run --out DIR --models google/gemini-3.8-flash,openai/gpt-audio-1.5 --prompts menu,checklist
  python bakeoff_site.py score --out DIR

Keys: SUPABASE_URL / SUPABASE_SECRET_KEY / SUPABASE_DB_URL (~/.t2a/asmrboard.env); GOOGLE_API_KEY, GEMINI_API_KEY,
GEMINI_API_KEY_2, OPENAI_API_KEY, OPENROUTER_API_KEY_GS (~/.t2a_env). Spend is capped per run with --cap (USD).
"""
from __future__ import annotations
import argparse, base64, itertools, json, os, random, re, subprocess, threading, time, urllib.error, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# USD per million tokens (input, output incl. thinking), for the cap; audio is billed as input tokens
PRICES = {"gemini-3.1-pro-preview": (2.0, 12.0), "gemini-3.8-flash": (0.75, 4.5), "gemini-3.7-flash": (0.75, 4.5),
          "gemini-omni-1.1-flash": (0.75, 4.5), "gemini-3.5-flash": (3.0, 12.0), "gpt-audio-1.5": (32.0, 64.0),
          "gpt-audio": (32.0, 64.0), "gpt-audio-mini": (0.6, 2.4)}
spent = 0.0
spent_lock = threading.Lock()


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def psql(sql: str) -> list[list[str]]:
    out = subprocess.run(["psql", os.environ["SUPABASE_DB_URL"], "-At", "-F", "\t", "-c", sql],
                         capture_output=True, text=True, check=True).stdout
    return [line.split("\t") for line in out.splitlines() if line]


# ---------- fetch: clips, human answers, menu, audio ----------

def fetch(a) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "audio").mkdir(exist_ok=True)
    menu = [{"key": k, "title": t or k, "hint": h, "grp": g} for k, t, h, g in
            psql("select key, coalesce(title, key), coalesce(hint, ''), grp from public.label_options order by sort")]
    (a.out / "menu.json").write_text(json.dumps(menu, indent=1))
    rows = psql("""select c.id, c.audio_path, p.username, array_to_json(l.labels), coalesce(q.status, 'too early'), c.votes
                   from public.labels l join public.clips c on c.id = l.clip_id join public.profiles p on p.id = l.user_id
                   left join private.labeller_quality q on q.user_id = l.user_id
                   where l.attempt = 1 and not l.unsure and not c.blocked and p.deleted_at is null
                     and coalesce(q.status, '') <> 'weak'""")
    by = defaultdict(lambda: {"answers": []})
    for cid, path, user, labs, status, votes in rows:
        by[cid].update(id=cid, audio=path)
        by[cid]["answers"].append({"user": user, "labels": json.loads(labs), "status": status})
    multi = [c for c in by.values() if len({x["user"] for x in c["answers"]}) >= 2]
    single = [c for c in by.values() if len(c["answers"]) == 1 and c["answers"][0]["status"] == "ok"]
    random.Random(0).shuffle(single)
    clips = multi + single[:max(0, a.n - len(multi))]
    log(f"{len(multi)} clips with 2+ people, {len(clips) - len(multi)} single-answer top-ups from 'ok' labellers")
    base = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SECRET_KEY"]

    def get(c):
        dst = a.out / "audio" / c["audio"]
        if dst.exists(): return True
        # a secret key goes in apikey only: as a Bearer token it is not a JWT and storage refuses it
        req = urllib.request.Request(f"{base}/storage/v1/object/clips/{c['audio']}", headers={"apikey": key})
        try:
            dst.write_bytes(urllib.request.urlopen(req, timeout=60).read())
            return True
        except urllib.error.HTTPError as e:
            log(f"  no audio for {c['id'][:8]} ({e.code} {e.read()[:80]!r})")
            return False
    with ThreadPoolExecutor(8) as ex: ok = list(ex.map(get, clips))
    clips = [c for c, good in zip(clips, ok) if good]
    (a.out / "clips.jsonl").write_text("".join(json.dumps(c) + "\n" for c in clips))
    log(f"wrote {len(clips)} clips to {a.out}")
    return 0


# ---------- prompts ----------

def prompt(kind: str, menu: list[dict]) -> str:
    lines = "\n".join(f"- {m['title']}: {m['hint']}" if m["hint"] else f"- {m['title']}" for m in menu)
    head = ("You are labelling a short clip (a few seconds) from an ASMR recording for a research dataset. Some clips "
            "are adult audio; label what you hear, as a sound, without judging it.\n\nLabels:\n" + lines + "\n\n")
    if kind == "menu":
        return head + ('Which of these sounds can you hear in the clip? Choose every label that applies, only from the '
                       'list (use the label names exactly). Answer with JSON only: {"labels": ["...", ...]}')
    return head + ("For every label in the list, decide whether that sound is clearly present in the clip. Answer with "
                   'JSON only, one entry per label, exactly as named: {"label name": true or false, ...}')


def parse(txt: str, kind: str, menu: list[dict]) -> list[str]:
    titles = {m["title"].lower(): m["key"] for m in menu} | {m["key"].lower(): m["key"] for m in menu}
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m: return []
    try: d = json.loads(m.group(0))
    except json.JSONDecodeError: return []
    picked = d.get("labels", []) if kind == "menu" else [k for k, v in d.items() if v is True or str(v).lower() == "true"]
    out = []
    for x in picked if isinstance(picked, list) else []:
        k = titles.get(str(x).strip().lower())
        if k and k not in out: out.append(k)
    return out


# ---------- providers ----------

GOOGLE_KEYS = [k for k in (os.environ.get("GOOGLE_API_KEY"), os.environ.get("GEMINI_API_KEY"), os.environ.get("GEMINI_API_KEY_2")) if k]
dead_keys: set[str] = set()


def cost(name: str, tin: int, tout: int) -> float:
    pin, pout = PRICES.get(name, (2.0, 12.0))
    return (tin * pin + tout * pout) / 1e6


def call_google(name: str, text: str, mp3: bytes) -> tuple[str, float]:
    body = {"contents": [{"role": "user", "parts": [{"text": text}, {"inline_data": {"mime_type": "audio/mp3", "data": base64.b64encode(mp3).decode()}}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 8000, "responseMimeType": "application/json"},
            "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in
                               ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]}
    last = None
    for key in [k for k in GOOGLE_KEYS if k not in dead_keys]:
        req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{name}:generateContent",
                                     data=json.dumps(body).encode(), headers={"x-goog-api-key": key, "Content-Type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=180))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code} {e.read()[:160]!r}"
            if e.code in (429, 403): dead_keys.add(key); continue     # this key is out of quota for today: try the next
            raise RuntimeError(last)
        u = d.get("usageMetadata", {})
        c = cost(name, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0))
        cand = (d.get("candidates") or [{}])[0]
        txt = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
        if not txt: txt = "REFUSED:" + str(cand.get("finishReason") or d.get("promptFeedback", {}).get("blockReason"))
        return txt, c
    raise RuntimeError(last or "no Google key with quota left")


def call_openai(name: str, text: str, mp3: bytes) -> tuple[str, float]:
    body = {"model": name, "modalities": ["text"], "temperature": 0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": text},
                                                      {"type": "input_audio", "input_audio": {"data": base64.b64encode(mp3).decode(), "format": "mp3"}}]}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=180))
    u = d.get("usage", {})
    return d["choices"][0]["message"].get("content") or "REFUSED:" + str(d["choices"][0].get("finish_reason")), \
        cost(name, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


def call_openrouter(name: str, text: str, mp3: bytes) -> tuple[str, float]:
    body = {"model": name, "temperature": 0, "usage": {"include": True},
            "messages": [{"role": "user", "content": [{"type": "text", "text": text},
                                                      {"type": "input_audio", "input_audio": {"data": base64.b64encode(mp3).decode(), "format": "mp3"}}]}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY_GS']}", "Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=180))
    if "error" in d: raise RuntimeError(str(d["error"])[:200])
    return d["choices"][0]["message"].get("content") or "", float((d.get("usage") or {}).get("cost") or 0)


def call(model: str, text: str, mp3: bytes) -> tuple[str, float]:
    provider, name = model.split("/", 1)
    if provider == "google": return call_google(name, text, mp3)
    if provider == "openai": return call_openai(name, text, mp3)
    return call_openrouter(model.split("/", 1)[1] if provider == "or" else model, text, mp3)


# ---------- run ----------

def run(a) -> int:
    global spent
    menu = json.loads((a.out / "menu.json").read_text())
    clips = [json.loads(x) for x in (a.out / "clips.jsonl").read_text().splitlines()][: a.limit or None]
    for model, kind in itertools.product(a.models.split(","), a.prompts.split(",")):
        path = a.out / "runs" / f"{model.replace('/', '_').replace(':', '-')}.{kind}.jsonl"
        path.parent.mkdir(exist_ok=True)
        done = {json.loads(x)["id"] for x in path.read_text().splitlines()} if path.exists() else set()
        todo = [c for c in clips if c["id"] not in done]
        text = prompt(kind, menu)
        log(f"{model} / {kind}: {len(todo)} to go")
        lock = threading.Lock()
        fails = 0

        def one(c):
            nonlocal fails
            global spent
            if spent >= a.cap: return
            try:
                txt, usd = call(model, text, (a.out / "audio" / c["audio"]).read_bytes())
            except Exception as e:                                   # noqa: BLE001 - one bad call must not stop the run
                with lock: fails += 1
                log(f"  {c['id'][:8]} failed: {str(e)[:160]}")
                return
            with spent_lock: spent += usd
            row = {"id": c["id"], "labels": parse(txt, kind, menu), "refused": txt.startswith("REFUSED:"), "raw": txt[:400], "usd": round(usd, 6)}
            with lock, path.open("a") as f: f.write(json.dumps(row) + "\n")
        with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, todo))
        log(f"  done ({fails} failed); spent so far ${spent:.2f} of ${a.cap:.2f}")
        if spent >= a.cap: log("cap reached, stopping"); break
    return 0


# ---------- score ----------

def f1(x: set, y: set) -> float:
    if not x and not y: return 1.0
    tp = len(x & y)
    return 2 * tp / (len(x) + len(y)) if (x or y) else 0.0


def score(a) -> int:
    menu = json.loads((a.out / "menu.json").read_text())
    grp = {m["key"]: m["grp"] for m in menu}
    clips = {c["id"]: c for c in (json.loads(x) for x in (a.out / "clips.jsonl").read_text().splitlines())}

    def stats(pairs):  # pairs: (predicted set, reference set)
        per = defaultdict(lambda: [0, 0, 0])                         # tp, predicted, reference
        fs = []
        for p, r in pairs:
            fs.append(f1(p, r))
            for k in p | r:
                per[k][0] += k in p and k in r; per[k][1] += k in p; per[k][2] += k in r
        return sum(fs) / max(len(fs), 1), per

    def grp_f1(per, g):
        tp = sum(v[0] for k, v in per.items() if grp.get(k) == g); pp = sum(v[1] for k, v in per.items() if grp.get(k) == g)
        rr = sum(v[2] for k, v in per.items() if grp.get(k) == g)
        return 2 * tp / (pp + rr) if pp + rr else float("nan")

    groups = ["Voice", "Triggers", "NSFW", "Other"]
    hh = [(set(x["labels"]), set(y["labels"])) for c in clips.values() for x, y in itertools.permutations(c["answers"], 2)]
    hf, hper = stats(hh)
    res = [("people vs each other", hf, hper, len(hh), 0, 0.0)]
    for path in sorted((a.out / "runs").glob("*.jsonl")):
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        pairs = [(set(r["labels"]), set(h["labels"])) for r in rows if r["id"] in clips and not r["refused"] for h in clips[r["id"]]["answers"]]
        mf, mper = stats(pairs)
        res.append((path.stem, mf, mper, len(rows), sum(r["refused"] for r in rows), sum(r["usd"] for r in rows)))
    lines = [f"{'labeller':52} {'F1':>5} " + " ".join(f"{g:>8}" for g in groups) + f" {'clips':>5} {'refused':>7} {'USD':>6}"]
    for name, mf, per, n, refused, usd in sorted(res, key=lambda x: -x[1]):
        lines.append(f"{name:52} {mf:5.2f} " + " ".join(f"{grp_f1(per, g):8.2f}" for g in groups) + f" {n:5d} {refused:7d} {usd:6.2f}")
    report = "\n".join(lines)
    print(report)
    (a.out / "report.txt").write_text(report + "\n")
    detail = {name: {k: {"tp": v[0], "pred": v[1], "ref": v[2]} for k, v in per.items()} for name, _, per, *_ in res}
    (a.out / "per_class.json").write_text(json.dumps(detail, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--out", type=Path, required=True); f.add_argument("--n", type=int, default=150)
    r = sub.add_parser("run"); r.add_argument("--out", type=Path, required=True); r.add_argument("--models", required=True)
    r.add_argument("--prompts", default="menu,checklist"); r.add_argument("--cap", type=float, default=10.0)
    r.add_argument("--workers", type=int, default=4); r.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("score"); s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    return {"fetch": fetch, "run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
