"""DDPM noising/denoising, matching the paper's Fig. 2 and Section IV.B.

The paper specifies a "linear noise schedule" DDPM (citing Ho et al. 2020)
for computational efficiency over other schedules. This is exactly that:
linear beta schedule, closed-form forward noising, epsilon-prediction
training objective, ancestral sampling for inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4,
                beta_end: float = 0.02, device: str = "cpu") -> None:
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device: str) -> "DDPM":
        for name in ("betas", "alphas", "alphas_cumprod",
                     "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form q(x_t | x_0): the forward process in one step.

        Sampling directly from x_0 (rather than iterating x_1..x_t) is what
        makes DDPM training tractable -- every training step only needs one
        random t and one noise draw, not a simulated trajectory.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_1mac = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_ac * x0 + sqrt_1mac * noise, noise

    def loss(self, model, x0: torch.Tensor, input_ids: torch.Tensor,
             attention_mask: torch.Tensor) -> torch.Tensor:
        """One training step's loss: MSE between predicted and true noise."""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noisy, noise = self.add_noise(x0, t)
        pred = model(noisy, t, input_ids, attention_mask)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, model, shape: tuple[int, ...], input_ids: torch.Tensor,
              attention_mask: torch.Tensor, device: str) -> torch.Tensor:
        """Ancestral sampling: iterate p(x_{t-1} | x_t) from pure noise."""
        x = torch.randn(shape, device=device)
        for t_val in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            pred_noise = model(x, t, input_ids, attention_mask)

            alpha = self.alphas[t_val]
            alpha_cumprod = self.alphas_cumprod[t_val]
            beta = self.betas[t_val]

            mean = (1 / alpha.sqrt()) * (
                x - (beta / (1 - alpha_cumprod).sqrt()) * pred_noise
            )
            if t_val > 0:
                noise = torch.randn_like(x)
                x = mean + beta.sqrt() * noise
            else:
                x = mean
        return x
