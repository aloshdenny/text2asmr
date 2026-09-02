"""LoRA finetuning of Chatterbox's T3 on ASMR speech.

Chatterbox ships no finetuning entry point, but it does expose the pieces:
``T3.loss()`` computes the text and speech cross-entropies given conditioning,
text tokens and speech tokens. So the work here is assembling batches in the
shape ``T3.loss`` expects, and attaching LoRA to the Llama backbone.

What is trained and what is not:

  * **T3** (text -> S3 speech tokens) is where whispered ASMR delivery lives --
    pacing, breathiness and prosody are all properties of the token sequence.
    LoRA adapters go here.
  * **S3Gen** (tokens -> waveform) and the **voice encoder** stay frozen. They
    are speaker/vocoder machinery, not style, and freezing them keeps the job
    inside a 20 GB card that is shared with other work.

Conditioning uses a reference clip from the *same source recording* as the
target, so the model learns the delivery rather than memorising one speaker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_backbone(device: str = "cuda"):
    """Load Chatterbox and freeze everything except what LoRA will adapt."""
    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device)
    for module in (model.s3gen, model.ve):
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()
    return model


class SpeechCollator:
    """Build a T3 training batch from packed dataset rows.

    Each row carries a 24 kHz mono waveform and its transcript. The collator
    tokenises the text, extracts S3 speech tokens from the audio, and pads both
    to the batch maximum, returning the lengths ``T3.loss`` needs to mask
    padding out of the cross-entropy.
    """

    def __init__(self, backbone, max_speech_tokens: int = 600) -> None:
        self.model = backbone
        self.tokenizer = backbone.tokenizer
        self.max_speech_tokens = max_speech_tokens
        self.device = backbone.device

    @staticmethod
    def _to_16k(wav: np.ndarray, src_sr: int) -> np.ndarray:
        """Resample to S3_SR.

        Both the S3 tokenizer and the voice encoder operate at 16 kHz, while
        the packed speech is 24 kHz (S3GEN_SR, the rate S3Gen *emits*). Feeding
        24 kHz to either produces plausible-looking but wrong output, with no
        error, so the conversion is explicit here rather than assumed.
        """
        import librosa

        from chatterbox.models.s3tokenizer import S3_SR

        if src_sr == S3_SR:
            return wav
        return librosa.resample(wav, orig_sr=src_sr, target_sr=S3_SR)

    def _speech_tokens(self, wavs16: list[np.ndarray]) -> list[torch.Tensor]:
        """S3 tokens for each 16 kHz waveform, via the frozen tokenizer."""
        with torch.no_grad():
            tokens, lens = self.model.s3gen.tokenizer.forward(wavs16)
        return [tokens[i, : lens[i]] for i in range(len(wavs16))]

    def __call__(self, rows: list[dict]) -> dict[str, Any] | None:
        wavs16, texts = [], []
        for row in rows:
            audio = row["audio"]
            wav = np.asarray(audio["array"], dtype=np.float32)
            if wav.size == 0:
                continue
            wavs16.append(self._to_16k(wav, int(audio["sampling_rate"])))
            texts.append(row["text"])
        if not wavs16:
            return None

        from chatterbox.tts import punc_norm

        hp = self.model.t3.hp
        speech = self._speech_tokens(wavs16)

        # Match inference exactly: normalise punctuation, then wrap with the
        # start/stop text tokens. T3.forward asserts the start token is present,
        # and training on unwrapped sequences would teach a different input
        # distribution from the one used at generation time.
        text_ids = []
        for t in texts:
            ids = self.tokenizer.text_to_tokens(punc_norm(t)).squeeze(0)
            ids = torch.as_tensor(ids, dtype=torch.long)
            text_ids.append(torch.cat([
                torch.tensor([hp.start_text_token], dtype=torch.long),
                ids,
                torch.tensor([hp.stop_text_token], dtype=torch.long),
            ]))

        # Speech tokens get the same treatment so the model learns where an
        # utterance ends rather than running to the token cap at inference.
        speech = [
            torch.cat([
                torch.tensor([hp.start_speech_token], dtype=torch.long),
                s.cpu().long(),
                torch.tensor([hp.stop_speech_token], dtype=torch.long),
            ])
            for s in speech
        ]

        keep = [
            i for i, s in enumerate(speech)
            if 0 < len(s) <= self.max_speech_tokens and len(text_ids[i]) > 0
        ]
        if not keep:
            return None
        speech = [speech[i] for i in keep]
        text_ids = [text_ids[i] for i in keep]

        # T3.loss asserts the padded width equals the max length, so pad to
        # exactly that rather than to a fixed bucket.
        t_len = torch.tensor([len(t) for t in text_ids], dtype=torch.long)
        s_len = torch.tensor([len(s) for s in speech], dtype=torch.long)
        text_pad = torch.zeros(len(keep), int(t_len.max()), dtype=torch.long)
        speech_pad = torch.zeros(len(keep), int(s_len.max()), dtype=torch.long)
        for i, (t, s) in enumerate(zip(text_ids, speech)):
            text_pad[i, : len(t)] = t
            speech_pad[i, : len(s)] = s

        # Speaker conditioning from the batch's own audio: the voice encoder is
        # frozen, so this is just a feature, not something being learned. It
        # also wants 16 kHz, hence the already-resampled waveforms.
        from chatterbox.models.s3tokenizer import S3_SR

        with torch.no_grad():
            spk = self.model.ve.embeds_from_wavs(
                [wavs16[i] for i in keep], sample_rate=S3_SR
            )
            spk = torch.from_numpy(np.asarray(spk)).float()

        return {
            "text_tokens": text_pad,
            "text_token_lens": t_len,
            "speech_tokens": speech_pad,
            "speech_token_lens": s_len,
            "speaker_emb": spk,
        }


IGNORE_ID = -100


def _causal_lm_losses(t3, *, cond, text_tokens, text_lens,
                      speech_tokens, speech_lens):
    """Next-token cross-entropy over T3's text and speech heads.

    Chatterbox ships a ``T3.loss`` but it cannot be used as-is:

    1. It calls ``F.cross_entropy(logits, targets)`` with logits shaped
       ``(B, seq, vocab)``. PyTorch reads that as ``(N, C, d)``, so it demands
       a target of ``(B, vocab)`` and raises on any real batch. The logits need
       transposing to ``(B, vocab, seq)``.
    2. It scores the logit at position *i* against the token at position *i*.
       For a causal transformer that is the identity -- the hidden state at
       *i* already encodes token *i* -- so the objective would be trivially
       satisfied without learning to predict anything. The targets have to be
       shifted by one.

    Both are fixed here. Padding is masked with ``IGNORE_ID`` using the true
    lengths, so padded positions contribute nothing.
    """
    import torch.nn.functional as F

    out = t3.forward(
        t3_cond=cond,
        text_tokens=text_tokens,
        text_token_lens=text_lens,
        speech_tokens=speech_tokens,
        speech_token_lens=speech_lens,
        training=True,
    )

    def head_loss(logits, tokens, lens):
        # Predict token t+1 from the latent at t.
        logits = logits[:, :-1, :]
        targets = tokens[:, 1:].clone()
        positions = torch.arange(targets.size(1), device=targets.device)
        # A target at index j corresponds to token j+1, which is padding once
        # j+1 >= len.
        targets[positions[None, :] + 1 >= lens[:, None]] = IGNORE_ID
        return F.cross_entropy(
            logits.transpose(1, 2), targets, ignore_index=IGNORE_ID
        )

    return (head_loss(out.text_logits, text_tokens, text_lens),
            head_loss(out.speech_logits, speech_tokens, speech_lens))


@dataclass
class ChatterboxFinetuner:
    """LoRA training loop over ``T3.loss``."""

    backbone: Any
    lora_r: int = 32
    lora_alpha: int = 64
    lr: float = 1e-4
    warmup: int = 200
    speech_loss_weight: float = 1.0
    text_loss_weight: float = 0.25

    def __post_init__(self) -> None:
        from peft import LoraConfig, get_peft_model

        t3 = self.backbone.t3
        for p in t3.parameters():
            p.requires_grad_(False)

        # Adapt attention and MLP projections of the Llama backbone. The
        # embedding and output heads stay frozen: the token inventory is
        # unchanged, only the mapping between text and delivery moves.
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        t3.tfmr = get_peft_model(t3.tfmr, config)
        self.t3 = t3

        trainable = [p for p in t3.parameters() if p.requires_grad]
        n = sum(p.numel() for p in trainable)
        print(f"trainable parameters: {n/1e6:.1f}M")

        self.opt = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=0.01)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / max(self.warmup, 1))
        )
        self._micro = 0

    def _forward_loss(self, batch: dict):
        """Weighted loss for one batch, without touching gradients."""
        from chatterbox.models.t3.modules.cond_enc import T3Cond

        device = self.backbone.device
        cond = T3Cond(
            speaker_emb=batch["speaker_emb"].to(device),
            cond_prompt_speech_tokens=None,
            emotion_adv=0.5 * torch.ones(len(batch["speaker_emb"]), 1, 1,
                                         device=device),
        )
        loss_text, loss_speech = _causal_lm_losses(
            self.t3,
            cond=cond,
            text_tokens=batch["text_tokens"].to(device),
            text_lens=batch["text_token_lens"].to(device),
            speech_tokens=batch["speech_tokens"].to(device),
            speech_lens=batch["speech_token_lens"].to(device),
        )
        # Speech tokens carry the delivery; the text head is kept in the loss
        # at low weight only to stop the shared backbone drifting.
        return (self.speech_loss_weight * loss_speech
                + self.text_loss_weight * loss_text)

    def loss_only(self, batch: dict | None) -> float:
        """Loss without a backward pass, for evaluation."""
        if batch is None:
            return float("nan")
        with torch.no_grad():
            return float(self._forward_loss(batch).detach())

    def step(self, batch: dict | None, accumulate: bool = False) -> float:
        """One micro-batch. Returns the loss, or nan for a skipped batch."""
        if batch is None:
            return float("nan")

        loss = self._forward_loss(batch)
        loss.backward()
        if not accumulate:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.t3.parameters() if p.requires_grad], 1.0
            )
            self.opt.step()
            self.sched.step()
            self.opt.zero_grad(set_to_none=True)
        return float(loss.detach())

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.t3.tfmr.save_pretrained(str(path))
        print(f"  saved adapter -> {path}")
