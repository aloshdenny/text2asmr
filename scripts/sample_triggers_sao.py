#!/usr/bin/env python3
"""Non-speech ASMR samples from base Stable Audio Open 1.0 (zero-shot), each checked by AST: did it make the sound
it was asked for? The baseline a trigger fine-tune has to beat.

Runs in its own env (diffusers >= 0.30 for StableAudioPipeline; the speech env pins an older diffusers), on the
research server (samples are scp'd to the Mac afterwards):
  t2a_run.ps1 -Name trig -Py D:\\t2a\\venv-sao\\Scripts\\python.exe -Cmd "scripts\\sample_triggers_sao.py --out D:\\t2a\\samples\\triggers_sao"
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np

PROMPTS = {
    "tapping": "ASMR close-up fingernail tapping on a wooden box, crisp gentle taps, quiet room, no talking",
    "scratching": "ASMR slow scratching on a foam microphone cover, textured scratching sounds, no talking",
    "crinkling": "ASMR crinkling a plastic wrapper close to the microphone, soft crinkles, no talking",
    "brushing": "ASMR soft makeup brush strokes on a microphone, gentle brushing, no talking",
    "liquid": "ASMR slowly pouring water into a glass, gentle water and liquid sounds, no talking",
    "paper rustling": "ASMR turning the pages of an old book, soft paper rustling, quiet room, no talking",
    "fabric rustling": "ASMR soft fabric rustling, hands moving over a cotton blanket, close up, no talking",
    "mouth sounds": "ASMR soft mouth sounds, tongue clicks and lip smacks close to the microphone, no words",
}
NEG = "talking, speech, voice, singing, music, hum, buzz, distortion, low quality"
AST = {"tapping": ["Tap", "Knock"], "scratching": ["Scratch", "Scrape", "Rub"], "crinkling": ["Crumpling, crinkling"],
       "brushing": ["Rub", "Scrape", "Toothbrush"], "liquid": ["Water", "Liquid", "Pour", "Drip", "Splash, splatter", "Trickle, dribble"],
       "paper rustling": ["Rustle", "Crumpling, crinkling", "Tearing"], "fabric rustling": ["Rustle", "Zipper (clothing)"],
       "mouth sounds": ["Chewing, mastication", "Biting", "Gargling", "Lip smacking"]}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--only", default="", help="comma-separated class keys")
    a = ap.parse_args()
    import torch, soundfile as sf, librosa
    from diffusers import StableAudioPipeline
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    a.out.mkdir(parents=True, exist_ok=True)
    # fp16 on CUDA fits beside a long training job (~4 GB); MPS needs fp32 (fp16 gives NaNs there)
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0",
                                               torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
    # the scheduler's Brownian-tree noise (torchsde) recurses past Python's limit with this torch; its output is
    # normalised Brownian increments, i.e. N(0, 1) per step, so seeded Gaussian noise is the same distribution
    import diffusers.schedulers.scheduling_cosine_dpmsolver_multistep as cos

    class GaussianNoise:
        def __init__(self, x, sigma_min=None, sigma_max=None, seed=None):
            self.shape, self.dtype = x.shape, x.dtype
            self.g = torch.Generator("cpu").manual_seed(int(seed[0] if isinstance(seed, list) else (seed or 0)))

        def __call__(self, sigma, sigma_next):
            return torch.randn(self.shape, generator=self.g).to(self.dtype)
    cos.BrownianTreeNoiseSampler = GaussianNoise
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe = ASTFeatureExtractor.from_pretrained(name); ast = ASTForAudioClassification.from_pretrained(name).eval()
    n2i = {v: k for k, v in ast.config.id2label.items()}; i2n = ast.config.id2label
    rows = []
    for key, prompt in PROMPTS.items():
        if a.only and key not in a.only.split(","): continue
        for s in range(a.seeds):
            t0 = time.time()
            g = torch.Generator("cpu").manual_seed(1000 + s)
            audio = pipe(prompt, negative_prompt=NEG, num_inference_steps=a.steps, guidance_scale=a.cfg,
                         audio_end_in_s=a.seconds, num_waveforms_per_prompt=1, generator=g).audios[0]
            x = audio.float().cpu().numpy().T                                          # (samples, channels)
            sr = pipe.vae.sampling_rate
            x = x / max(1e-6, np.abs(x).max()) * 0.5
            f = a.out / f"{key.replace(' ', '_')}_{s}.wav"; sf.write(f, x, sr)
            mono16 = librosa.resample(x.mean(1), orig_sr=sr, target_sr=16000)
            with torch.no_grad(): p = torch.sigmoid(ast(**fe([mono16], sampling_rate=16000, return_tensors="pt")).logits)[0].numpy()
            want = max(p[n2i[n]] for n in AST[key] if n in n2i)
            top = [(i2n[int(i)], round(float(p[i]), 2)) for i in p.argsort()[::-1][:4]]
            rows.append({"file": f.name, "class": key, "prompt": prompt, "requested_p": round(float(want), 3), "ast_top": top,
                         "speech_p": round(float(max(p[n2i["Speech"]], p[n2i["Whispering"]])), 3)})
            log(f"{f.name}: requested-class p {want:.2f} | speech {rows[-1]['speech_p']:.2f} | AST top {top} ({time.time() - t0:.0f} s)")
    (a.out / "report.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
