"""NullSwap losses: identity cosine, LPIPS, MSE, PatchGAN, and Dynamic Loss Weighting."""
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


def identity_cos_loss(e_clean, e_pert):
    """Cosine similarity between clean and perturbed embeddings. Minimize this (pushes apart)."""
    return F.cosine_similarity(e_clean, e_pert, dim=1).mean()


class DynamicLossWeighter:
    """Adaptive per-recognizer loss weights.

    Paper (Eq. 4):
        w_i(t_e, t_b) = max(1 / max(alpha * sigma_i^2(t_b) + beta(t_e) * (1 + Delta_i(t_b)), eps_d), eps_w)

        sigma_i^2: variance of L_i over last k=30 iterations (batch means).
        Delta_i   = max((L_i(t-1) - L_i(t)) / (L_i(t-1) + eps_p), -1)   # relative progress.
        beta(t_e) = min(beta_init + gamma_dlw * t_e, beta_final)        # paper Eq. 5
                  : grows by +gamma_dlw=0.1 per epoch from 0.5, saturates at 2.0 by epoch 15.

    Final weights normalized via paper Eq. 3 (sum-to-c) so total id pressure is invariant."""

    def __init__(self, n_models, k=30, alpha=3.0, beta_init=0.5, beta_final=2.0,
                 gamma_dlw=0.1, eps_d=1e-6, eps_w=1e-6, eps_p=1e-6):
        self.n = n_models
        self.k = k
        self.alpha = alpha
        self.beta_init = beta_init
        self.beta_final = beta_final
        self.gamma_dlw = gamma_dlw
        self.eps_d = eps_d
        self.eps_w = eps_w
        self.eps_p = eps_p
        self.history = [deque(maxlen=k) for _ in range(n_models)]
        self.last_loss = [None] * n_models

    def step(self, epoch, per_model_losses):
        """epoch: integer epoch index (0-based). per_model_losses: list of floats (batch means).
        Returns: list[float] sum-to-c-normalized weights (paper Eq. 3).
        """
        beta = min(self.beta_init + self.gamma_dlw * epoch, self.beta_final)
        variances = []
        deltas = []
        for i, L in enumerate(per_model_losses):
            self.history[i].append(L)
            if len(self.history[i]) >= 2:
                arr = torch.tensor(list(self.history[i]))
                variances.append(arr.var(unbiased=False).item())
            else:
                variances.append(0.0)
            if self.last_loss[i] is None:
                deltas.append(0.0)
            else:
                # Bulletproof against L(t-1) landing on exactly -eps_p (very improbable
                # with float32, but Python floats raise ZeroDivisionError unlike tensors).
                denom_p = self.last_loss[i] + self.eps_p
                if denom_p == 0.0:
                    denom_p = 1e-9
                d = (self.last_loss[i] - L) / denom_p
                deltas.append(max(d, -1.0))
            self.last_loss[i] = L
        raw_w = []
        for sigma2, delta in zip(variances, deltas):
            denom = max(self.alpha * sigma2 + beta * (1.0 + delta), self.eps_d)
            w = max(1.0 / denom, self.eps_w)
            raw_w.append(w)
        # Paper Eq. 3: ŵ_i = c · w_i / Σ_j w_j  (renormalized so weights sum to c, the
        # number of FR models). Without this, scale drift between recognizers can collapse
        # the effective identity loss.
        c = float(self.n)
        s = sum(raw_w)
        if s <= 0:
            return [c / max(self.n, 1)] * self.n
        return [c * w / s for w in raw_w]


def discriminator_loss(D, real_255, fake_255):
    """BCE-with-logits PatchGAN loss."""
    pred_real = D(real_255)
    pred_fake = D(fake_255.detach())
    l_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
    l_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
    return 0.5 * (l_real + l_fake)


def generator_gan_loss(D, fake_255):
    pred = D(fake_255)
    return F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred))
