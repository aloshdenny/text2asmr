#!/usr/bin/env python3
"""Keep acquiring ASMR audio for the classes that are still under target, until the ontology is balanced.

One cycle: measure labeled-class counts -> pick the classes still short -> discover NEW creators whose posts
are tagged for those classes -> stream their audio into the corpus repo its voice belongs to -> repeat.
Network-only, sized for the 1 vCPU / ~15 GB DO droplet: it never holds more than a few files at a time and
every creator is committed to the Hub before the next one starts.

  python3 fountain.py --target 150000 --cycle-gb 40 --state /root/t2a/fountain

Acquisition alone does not balance anything: it buys *audio*.  That audio only becomes labeled clips after
transcription (Modal/tinkerspace) and Qwen3-Omni labeling, so the counts this loop reads lag the downloads
by a day or so.  That lag is fine -- it stops the loop from over-buying a class while its labels are in
flight only if you let a cycle finish; hence the long default cycle gap.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from collections import Counter
from pathlib import Path

MOMMY, DADDY = "aoxo/t2a-mommy", "aoxo/t2a-daddy"
def ledgers() -> list[tuple[str, str]]:
    """Every label ledger on the Hub, including the per-shard files parallel labeling runs write.
    Missing these undercounts the corpus and makes the loop keep buying a class that is already full."""
    from huggingface_hub import HfApi
    api = HfApi(); out = []
    for repo in (MOMMY, DADDY):
        out += [(repo, f) for f in api.list_repo_files(repo, repo_type="dataset")
                if f.startswith("labels/qwen3omni") and f.endswith(".jsonl")]
    return out
# the vocal core, which soundgasm can actually supply
TARGET_CLASSES = ["kissing", "moaning", "mouth sounds", "breathing"]
# the physical tail: present in the corpora only in traces (scratching ~370, brushing ~46 out of 3.15M
# clips), so it gets its own, much lower target and its own queries -- soundgasm is a vocal-ASMR site
PHYSICAL_CLASSES = ["tapping", "scratching", "crinkling", "brushing", "liquid"]
PHYSICAL_TARGET_FRAC = 0.1


def log(m): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def class_counts(cache: Path) -> Counter:
    """Labeled clips per class, per corpus and combined, straight from the ledgers on the Hub."""
    from huggingface_hub import hf_hub_download
    c = Counter()
    for repo, name in ledgers():
        try: p = hf_hub_download(repo, name, repo_type="dataset", cache_dir=str(cache), force_download=True)
        except Exception as e: log(f"  ledger {repo}:{name} unavailable ({type(e).__name__})"); continue
        side = "f" if repo == MOMMY else "m"
        for line in open(p):
            try: r = json.loads(line)
            except Exception: continue
            lab = r.get("label")
            if lab: c[lab] += 1; c[f"{lab}|{side}"] += 1
        try: os.remove(os.path.realpath(p))
        except OSError: pass
    return c


def refresh_exclusions(path: Path) -> int:
    """Every creator already in either corpus; discovery refuses all of them, so the loop only ever adds breadth."""
    from huggingface_hub import HfApi
    api = HfApi(); names = set()
    for repo in (MOMMY, DADDY):
        for f in api.list_repo_files(repo, repo_type="dataset"):
            if f.endswith(".m4a"): names.add(f.split("/")[0].lower())
    path.write_text("\n".join(sorted(names)) + "\n")
    return len(names)


def run(cmd, cwd, timeout=None) -> int:
    """A slow or failing step must not end the fountain: report it and let the next cycle try again."""
    log("  $ " + " ".join(str(c) for c in cmd))
    try:
        return subprocess.run(cmd, cwd=cwd, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        log(f"  step exceeded {timeout}s and was killed"); return -1
    except Exception as e:
        log(f"  step failed: {type(e).__name__} {str(e)[:160]}"); return -2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, default=Path("/root/t2a/fountain"))
    ap.add_argument("--repo-dir", type=Path, default=Path("/root/t2a"))
    ap.add_argument("--python", default="/root/t2a/venv/bin/python")
    ap.add_argument("--target", type=int, default=150000, help="labeled clips wanted per target class")
    ap.add_argument("--per-side", action="store_true", help="require the target per gender, not combined")
    ap.add_argument("--cycle-gb", type=float, default=40.0, help="GB to acquire per cycle")
    ap.add_argument("--cap-files", type=int, default=60, help="files per creator: breadth over depth")
    ap.add_argument("--pages", type=int, default=60, help="search pages per query; a cycle must finish, so keep it small")
    ap.add_argument("--min-seen", type=int, default=3)
    ap.add_argument("--sleep", type=int, default=3600, help="seconds between cycles")
    ap.add_argument("--max-cycles", type=int, default=0, help="0 = until balanced")
    a = ap.parse_args()
    a.state.mkdir(parents=True, exist_ok=True)
    cache = a.state / "cache"; cache.mkdir(exist_ok=True)
    hist = a.state / "fountain.jsonl"

    cycle = 0
    while not a.max_cycles or cycle < a.max_cycles:
        cycle += 1
        log(f"=== cycle {cycle}")
        counts = class_counts(cache)
        if a.per_side:
            short = [c for c in TARGET_CLASSES
                     if min(counts[f"{c}|f"], counts[f"{c}|m"]) < a.target // 2]
        else:
            short = [c for c in TARGET_CLASSES if counts[c] < a.target]
        phys_target = max(1, int(a.target * PHYSICAL_TARGET_FRAC))
        short_phys = [c for c in PHYSICAL_CLASSES if counts[c] < phys_target]
        log("  labeled: " + ", ".join(f"{c}={counts[c]} (f{counts[f'{c}|f']}/m{counts[f'{c}|m']})" for c in TARGET_CLASSES))
        log("  physical: " + ", ".join(f"{c}={counts[c]}" for c in PHYSICAL_CLASSES) + f" (target {phys_target})")
        short = short + short_phys
        if not short:
            log(f"  every class is at target ({a.target} vocal / {phys_target} physical); fountain done"); break
        log(f"  short: {short}")

        n_excl = refresh_exclusions(a.state / "existing_creators.txt")
        log(f"  exclusion list: {n_excl} creators already in the corpora")

        # Breadth queries stay in the mix (deficit tags only lift yield ~1.3x; new creators are what scales),
        # but only one breadth group per cycle -- running all four took >6 h and never reached acquisition.
        breadth = ["_generic_f", "_generic_m", "_round2_f", "_round2_m"]
        labels = ",".join(short + [breadth[(cycle - 1) % len(breadth)]])
        plan = a.state / f"plan.c{cycle}.jsonl"
        rc = run([a.python, str(a.repo_dir / "discover_creators.py"), "--exclude", str(a.state / "existing_creators.txt"),
                  "--pages", str(a.pages), "--min-seen", str(a.min_seen), "--cap-files", str(a.cap_files),
                  "--only-labels", labels, "--out", str(plan)], cwd=a.state, timeout=3 * 3600)
        n_plan = sum(1 for _ in open(plan)) if plan.exists() else 0
        log(f"  discovery rc={rc}: {n_plan} new creators planned")
        if not n_plan:
            log("  soundgasm is exhausted for these queries; sleeping longer before retrying")
            time.sleep(max(a.sleep, 6 * 3600)); continue

        rc = run([a.python, str(a.repo_dir / "acquire_stream.py"), "--plan", str(plan),
                  "--work", str(a.state / "acq"), "--max-gb", str(a.cycle_gb), "--cap-files", str(a.cap_files)],
                 cwd=a.state, timeout=20 * 3600)
        log(f"  acquire rc={rc}")

        # publish the acquisition ledger so the transcribe/label workers pick the new creators up
        try:
            from huggingface_hub import HfApi
            led = a.state / "acq" / "acquired.jsonl"
            if led.exists():
                HfApi().upload_file(path_or_fileobj=str(led), path_in_repo="v2/acquired_snapshot.jsonl",
                                    repo_id="aoxo/clap-ft-data", repo_type="dataset",
                                    commit_message=f"fountain cycle {cycle}: acquisition ledger")
                log(f"  uploaded acquisition ledger ({sum(1 for _ in open(led))} creators)")
        except Exception as e:
            log(f"  ledger upload failed: {type(e).__name__} {str(e)[:120]}")

        with hist.open("a") as f:
            f.write(json.dumps({"cycle": cycle, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "short": short,
                                "planned": n_plan, "counts": {c: counts[c] for c in TARGET_CLASSES}}) + "\n")
        time.sleep(a.sleep)
    log("FOUNTAIN_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
