"""Levenberg-Marquardt optimizers (standard and trust-region damping).

Written for this project (the authors use the external `torch-levenberg-marquardt`
package; this is an independent implementation of the paper's Algorithm 2).

Update vector (Eq. 15):   s = -(J^T J + gamma I)^{-1} J^T e,   e = y_pred - t
Trust-region ratio (Eq. 17), written with the error norms (the paper's Eq. 17
prints parameter norms ||b||, which we read as a typo):
    r = (||e(b)||^2 - ||e(b + s)||^2) / (||e(b)||^2 - ||J s + e(b)||^2)
Damping update (Eq. 18):  r < 0.25 -> 4 gamma ;  r > 0.75 -> gamma / 2
Acceptance (Eq. 19):      keep b + s only if the error decreases.
Stopping: ||J^T e|| < tau (tau = 1e-6).

"Standard" LM uses the classic Marquardt rule: gamma /= 10 after an accepted
step, gamma *= 10 after a rejected one.

The Jacobian of the residuals w.r.t. all trainable parameters is computed as
per-sample gradients, vmap(jacrev(.)) over the samples (much faster here than
forward-mode jacfwd for trapezoidal MFs, whose sort op is slow in forward mode).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.func import functional_call, jacrev, vmap


@dataclass
class LMState:
    gamma: float = 1.0
    accepted: int = 0
    rejected: int = 0
    converged: bool = False


class LevenbergMarquardt:
    """Call `step(x, y)` once per mini-batch. Parameters of `model` are updated in place."""

    def __init__(self, model: nn.Module, mode: str = "trust_region", gamma_init: float = 1.0,
                 max_tries: int = 10, tau: float = 1e-6, gamma_min: float = 1e-12,
                 gamma_max: float = 1e12):
        if mode not in ("trust_region", "standard"):
            raise ValueError("mode must be 'trust_region' or 'standard'")
        self.model, self.mode = model, mode
        self.max_tries, self.tau = max_tries, tau
        self.gamma_min, self.gamma_max = gamma_min, gamma_max
        self.state = LMState(gamma=gamma_init)
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.shapes = [p.shape for n, p in model.named_parameters() if p.requires_grad]
        self.sizes = [s.numel() for s in self.shapes]

    # --------------------------------------------------------------- helpers
    def _get_flat(self) -> torch.Tensor:
        params = dict(self.model.named_parameters())
        return torch.cat([params[n].detach().reshape(-1) for n in self.names])

    def _unflatten(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        chunks = torch.split(flat, self.sizes)
        return {n: c.reshape(s) for n, c, s in zip(self.names, chunks, self.shapes)}

    def _set_flat(self, flat: torch.Tensor) -> None:
        with torch.no_grad():
            params = dict(self.model.named_parameters())
            for n, v in self._unflatten(flat).items():
                params[n].copy_(v)

    def _residuals(self, flat: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = functional_call(self.model, self._unflatten(flat), (x,))
        return (pred - y).reshape(-1)

    def _jacobian(self, flat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """d(prediction)/d(params), shape (n_samples * out_dim, n_params)."""
        def single(bf: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
            return functional_call(self.model, self._unflatten(bf), (xi.unsqueeze(0),)).reshape(-1)
        J = vmap(jacrev(single), in_dims=(None, 0))(flat, x)   # (n, out_dim, n_params)
        return J.reshape(-1, flat.numel())

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """One LM iteration on (x, y). Returns the MSE after the step."""
        st = self.state
        b = self._get_flat()
        with torch.enable_grad():
            e = self._residuals(b, x, y)
            J = self._jacobian(b, x)                             # (n_res, n_params)
        e64, J64 = e.double(), J.double()
        g = J64.T @ e64                                         # gradient (up to factor 2)
        if g.norm().item() < self.tau:
            st.converged = True
            return e64.pow(2).mean().item()
        H = J64.T @ J64
        I = torch.eye(H.shape[0], dtype=H.dtype)
        sse = e64.pow(2).sum()

        for _ in range(self.max_tries):
            try:
                s = -torch.linalg.solve(H + st.gamma * I, g)
            except RuntimeError:                                # singular -> more damping
                st.gamma = min(st.gamma * 10, self.gamma_max)
                continue
            b_new = b + s.to(b.dtype)
            sse_new = self._residuals(b_new, x, y).double().pow(2).sum()
            improved = bool(torch.isfinite(sse_new)) and sse_new < sse

            if self.mode == "trust_region":
                predicted = sse - (J64 @ s + e64).pow(2).sum()
                r = ((sse - sse_new) / predicted).item() if predicted > 0 else -1.0
                if r < 0.25:
                    st.gamma = min(st.gamma * 4, self.gamma_max)
                elif r > 0.75:
                    st.gamma = max(st.gamma / 2, self.gamma_min)
            else:
                st.gamma = (max(st.gamma / 10, self.gamma_min) if improved
                            else min(st.gamma * 10, self.gamma_max))

            if improved:
                self._set_flat(b_new)
                st.accepted += 1
                return (sse_new / e64.numel()).item()
            st.rejected += 1
        return (sse / e64.numel()).item()
