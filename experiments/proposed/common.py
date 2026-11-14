"""Shared settings for the proposed-method experiments (Checkpoint 5)."""
from __future__ import annotations

INITS = ("random", "lhs", "fcm", "fcm_xy")
INIT_LABEL = {"random": "random (baseline)", "lhs": "lhs (kontrol)", "fcm": "fcm (önerilen)",
              "fcm_xy": "fcm_xy (önerilen varyant)"}
SUCCESS_SINC = 1e-5   # a sinc run "converges" if val MSE < 1e-5 (same threshold as E1b)
