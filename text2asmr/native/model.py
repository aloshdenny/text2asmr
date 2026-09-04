"""The T2A paper's own architecture: BERT -> FiLM-conditioned 1D U-Net -> DDPM.

Reimplements Section IV.B of the paper rather than finetuning a pretrained
backbone. Architecture, matched to the paper's Figures 1-4:

  Text -> BertTokenizer -> BERT -> Conditioning Network -> conditioned features
  Waveform -> DDPM noising -> noisy waveform
  (noisy waveform, conditioned features) -> UNet-FiLM -> predicted noise
  iterate t steps -> denoised waveform

One addition beyond what the paper's figures show: a sinusoidal timestep
embedding, summed with the BERT-derived conditioning before the FiLM
shift/scale projection. The paper's Fig. 4 doesn't depict it, but a DDPM
denoiser structurally cannot tell noise levels apart without seeing t
somewhere -- this is the standard, necessary way to supply it, not an
embellishment.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    """Sinusoidal position embedding for the diffusion timestep."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class ConditioningNetwork(nn.Module):
    """BERT text embedding -> conditioned features, matching Fig. 1.

    BERT is frozen: the paper conditions on BERT's representations, not on a
    finetuned language model, and freezing keeps the trainable model exactly
    where the paper's does -- in the conditioning network and the U-Net.
    """

    def __init__(self, cond_dim: int, bert_name: str = "bert-base-uncased") -> None:
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(bert_name)
        for p in self.bert.parameters():
            p.requires_grad_(False)
        self.bert.eval()

        self.project = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    @torch.no_grad()
    def _bert_embed(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output  # (B, hidden)

    def forward(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
        return self.project(self._bert_embed(input_ids, attention_mask))


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation: Fig. 4's shift/scale transforms.

    y = x * (1 + scale) + shift

    The (1 + scale) parameterization (rather than raw scale) means an
    all-zero projection is the identity, so FiLM starts as a no-op and the
    network has to learn to use conditioning rather than fight it from a
    random-scale initialization.
    """

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, channels)
        self.to_shift = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.to_scale(cond)[:, :, None]
        shift = self.to_shift(cond)[:, :, None]
        return x * (1 + scale) + shift


class DownBlock(nn.Module):
    """Fig. 3's downsampling block: Conv -> BatchNorm -> ReLU -> FiLM -> pool."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2)
        self.norm = nn.BatchNorm1d(out_ch)
        self.film = FiLMLayer(cond_dim, out_ch)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.film(x, cond)
        skip = x
        return self.pool(x), skip


class UpBlock(nn.Module):
    """Fig. 3's upsampling block: ConvT -> BatchNorm -> ReLU -> FiLM.

    Takes the skip connection from the matching DownBlock, concatenated
    before the transposed conv -- the U-Net's namesake connection, and what
    lets the network "look back," as the paper's prose describes.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        # The skip connection carries in_ch channels, not out_ch: it comes
        # from the DownBlock whose *output* was in_ch (this block mirrors
        # that block across the bottleneck). Concatenating [upconv(out_ch),
        # skip(in_ch)] gives out_ch + in_ch channels into the conv -- not
        # 2*out_ch, which is only correct when in_ch == out_ch.
        self.conv = nn.Conv1d(out_ch + in_ch, out_ch, kernel_size=5, padding=2)
        self.norm = nn.BatchNorm1d(out_ch)
        self.film = FiLMLayer(cond_dim, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
               cond: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        if x.shape[-1] != skip.shape[-1]:
            # Odd input lengths can shift pool/unpool sizes by one sample;
            # trim rather than error, since this is a boundary artifact, not
            # a real shape bug.
            n = min(x.shape[-1], skip.shape[-1])
            x, skip = x[..., :n], skip[..., :n]
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return self.film(x, cond)


class UNetFiLM(nn.Module):
    """The full noise-prediction network: Fig. 1's 'UNet with FiLM' box."""

    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256, 512),
                cond_dim: int = 256) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.time_embed = TimestepEmbedding(cond_dim)

        self.in_conv = nn.Conv1d(1, channels[0], kernel_size=5, padding=2)
        self.downs = nn.ModuleList([
            DownBlock(channels[i], channels[i + 1], cond_dim)
            for i in range(len(channels) - 1)
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], kernel_size=5, padding=2),
            nn.BatchNorm1d(channels[-1]),
            nn.ReLU(),
        )
        self.ups = nn.ModuleList([
            UpBlock(channels[i + 1], channels[i], cond_dim)
            for i in reversed(range(len(channels) - 1))
        ])
        self.out_conv = nn.Conv1d(channels[0], 1, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
               text_cond: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) noisy waveform. t: (B,) timesteps. text_cond: (B, cond_dim)."""
        cond = text_cond + self.time_embed(t)

        x = self.in_conv(x)
        skips = []
        for down in self.downs:
            x, skip = down(x, cond)
            skips.append(skip)

        x = self.bottleneck(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, cond)

        return self.out_conv(x)


class T2ANative(nn.Module):
    """Ties the conditioning network and U-Net-FiLM together for training."""

    def __init__(self, cond_dim: int = 256,
                channels: tuple[int, ...] = (32, 64, 128, 256, 512),
                bert_name: str = "bert-base-uncased") -> None:
        super().__init__()
        self.conditioning = ConditioningNetwork(cond_dim, bert_name)
        self.unet = UNetFiLM(channels, cond_dim)
        # Downsampling halves the length once per DownBlock; the input length
        # must be a multiple of this or skip connections misalign.
        self.length_multiple = 2 ** (len(channels) - 1)

    def forward(self, noisy_waveform: torch.Tensor, t: torch.Tensor,
               input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cond = self.conditioning(input_ids, attention_mask)
        return self.unet(noisy_waveform, t, cond)
