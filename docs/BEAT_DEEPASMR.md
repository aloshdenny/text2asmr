# Beating DeepASMR (and everything else in the space)

DeepASMR (arXiv 2601.15596) is the strongest published ASMR generator: zero-shot ASMR speech in anyone's
voice from one short snippet of ordinary read speech, LLM content-style encoder plus a flow-matching
acoustic decoder, trained on DeepASMR-DB (670 h, English + Chinese, 35 speakers, 28 F / 7 M), evaluated with
objective metrics, human ASMR-MOS, LLM-based scoring and unvoiced-speech analysis.

We do not beat that by doing the same thing slightly better. We beat it on the axis their corpus cannot
express, while matching them on theirs.

## 1. Where they are ahead, honestly

| | DeepASMR | us, today |
|---|---|---|
| generator | shipped, SOTA naturalness and style fidelity | v1 frozen (trigger LoRA produced noise); nothing shippable |
| human evaluation | ASMR-MOS listening study, done | none yet |
| decoder | flow matching | DDPM U-Net (native) / latent diffusion (Stable Audio LoRA) |
| zero-shot voice | one read-speech snippet is enough | Chatterbox reference conditioning exists, never evaluated for this |

Their lead is real and it is mostly *execution*, not data.

## 2. Where we are ahead, and why it is structural

| | DeepASMR | us |
|---|---|---|
| speakers / creators | 35 | **2,810 with labels**, ~6,600 in corpus |
| labelled events | none — speech style only | **~6.1 M** non-speech clips across a 14-class ontology |
| non-speech triggers | not modelled | the entire point of the ontology layer |
| gender | 28 F / 7 M, acknowledged imbalance | separate corpora by voice, balance targeted per gender |
| language | EN + ZH | EN (their ZH is a real advantage we do not have) |

A 670 h speech corpus can teach *how ASMR speech sounds*. It cannot teach *when a kiss happens, how long it
lasts, and what follows it*, because those events were never annotated. Ours were. That is the wedge.

## 3. What we steal, deliberately

1. **Flow-matching acoustic decoder.** Our native model is a DDPM U-Net; flow matching is faster to sample
   and better behaved at the same parameter count. Replace the sampler in `text2asmr/native/diffusion.py`
   before spending another GPU-week on the old one.
2. **Zero-shot from ordinary read speech.** Their framing is better than ours: condition on a snippet of the
   speaker's *normal* voice, not on an ASMR sample. That makes the model useful to people who have never
   recorded ASMR, and it is a harder, more honest test of style transfer.
3. **ASMR-MOS.** Rate comfort and relaxation, not just naturalness. A clip can be perfectly natural and
   useless as ASMR, and MOS-naturalness will not notice.
4. **LLM-based scoring.** Cheap, repeatable, and runs today through OpenRouter with audio input (~$0.0025
   per clip on Gemini Pro). It is not a substitute for listeners; it is the fast inner loop between them.
5. **Unvoiced-speech analysis.** An objective proxy for whispered delivery — measurable without listeners
   and directly relevant to our whispering class.

## 4. What we add that they cannot

1. **Trigger-conditioned generation with time placement.** `[whispering] ... [kissing] ... [tapping]` where
   each tag lands at a requested time and lasts a requested duration. This needs event-level labels; nobody
   else in the space has them at our scale.
2. **CLAP-v6 as an automatic trigger-fidelity metric.** Ask for kissing at 3.0-6.0 s, generate, and score the
   generated window with the ontology model: did the requested event actually appear, at the right time,
   without leaking into neighbouring windows? This is an evaluation contribution in its own right, and it is
   only possible because the ontology model came first.
3. **Creator-split generalisation.** With 2,810 labelled creators we can hold out creators, not just
   utterances, and report zero-shot quality on voices never seen. With 35 speakers that measurement is noise.

## 5. The evaluation protocol we ship against

Every number reported per class and per gender, never averaged into one figure.

| axis | metric | how |
|---|---|---|
| trigger fidelity | CLAP-v6 P(requested class) in the requested window, and leakage outside it | automatic |
| delivery | unvoiced-speech ratio vs real ASMR reference distribution | automatic |
| speaker similarity | embedding cosine to the reference snippet | automatic |
| comfort | **ASMR-MOS**, human listeners | manual, small n |
| comfort, inner loop | LLM-judge rating over the same rubric | OpenRouter, ~$0.0025/clip |
| long-form | drift in the above over 60 s+, prefill/completion windows | automatic |

The last row is the user's original concern and nobody else measures it: ASMR is consumed in tens of
minutes, and every published result is on clips of a few seconds.

## 6. Order of work

1. **Label quality first.** The kissing / mouth-sounds boundary was undefined in our own prompt and three
   judges disagreed on 70-80% of mouth-sounds clips. Fixed 2026-09-26; re-label affected classes. Nothing
   downstream is worth building on labels that mean two different things.
2. **CLAP v7** — full ontology including the physical tail, so the metric in §4.2 covers every tag the
   generator can be asked for.
3. **Generator v2**: Chatterbox T3 retrained on whisper-tagged speech, flow-matching decoder for triggers,
   CLAP-v6 as reward/consistency scorer.
4. **Evaluation** per §5, published with the model card, including the classes where we are *worse*.

## 7. What would make us wrong

If trigger-conditioned generation turns out not to matter perceptually — if listeners rate a well-delivered
whisper with no trigger control above a clumsy one with perfect trigger placement — then their approach is
simply the right one and our ontology work buys nothing at generation time. That is testable in the first
ASMR-MOS study, and it should be tested early, before we spend a generator-scale training budget on it.

## 8. Ontology decision, 2026-09-26: kissing + mouth sounds -> oral sounds

Five judges from four families (Gemini 3.1 Pro, Gemini 3.8 Flash, Perceptron mk1.5, Xiaomi MiMo v2.5,
Qwen3-Omni) were given the same 398 held-out clips and the same forced three-way question.

| | kissing / mouth sounds / moaning | oral sounds / moaning |
|---|---|---|
| mean pairwise agreement | 66.3% | **82.8%** |
| worst pair | 58.3% | **76.4%** |
| unanimous clips | 41% | **65%** |

With three options, chance agreement is 33%. Several cross-family pairs on *mouth sounds alone* landed at
20-45% -- at or below chance. Merging lifts every single pair.

Two earlier attempts to fix this without merging both failed, which is what makes the merge defensible
rather than lazy:

* the class definitions in our own prompt overlapped ("lip/kiss sounds" vs "lip smacks"). Rewriting them as
  discrete gesture vs continuous texture moved Qwen3-vs-Gemini agreement on mouth sounds from 30.7% to
  48.0% and stalled there.
* of the clips still disputed after that fix, a quarter were called *moaning* by Gemini -- the audio is
  genuinely multi-label, and no single-choice question can represent it.

Decision: train and evaluate on `oral sounds`; keep `moaning` separate (91.5-100% agreement, the most solid
class we have). `raw_label` retains kissing / mouth sounds on every row, so the split can be restored from
human labels later without relabelling.

Consequence for the generator: `[kissing]` remains a legal bracket tag, but trigger fidelity is measured at
the `oral sounds` level until a human-labelled set shows the finer distinction is real. Claiming per-kiss
control we cannot measure would be the kind of unsupported claim the T2A paper already made once.
