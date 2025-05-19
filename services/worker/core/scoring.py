"""Voice matching – cosine, PLDA, logistic Q‑score & tier logic."""


from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cosine
# from speechbrain.utils.metric_learning import SB_PLDA, score_plda

from speechbrain.processing.PLDA_LDA import PLDA as SB_PLDA, fast_PLDA_scoring as score_plda

from .config import load_thresholds, settings

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
THRESHOLD = 0.75


def _l2_normalize(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(v, p=2.0, dim=-1)


# ------------------------------------------------------------------
# models – lazy singletons to keep memory footprint low
# ------------------------------------------------------------------
def score_embedding(emb: np.ndarray, profile: np.ndarray, tier="default"):
    sims = 1 - np.array([cosine(emb, p) for p in profile])
    score = float(sims.mean())
    return score, bool(score > THRESHOLD)

def _plda_path() -> Path:
    return settings.THRESHOLDS_PATH.with_name("plda.pkl")


@lru_cache(maxsize=1)
def _get_plda() -> SB_PLDA:
    with open(_plda_path(), "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _get_q_model() -> LogisticRegression:
    t = load_thresholds()
    clf = LogisticRegression()
    clf.coef_ = np.asarray([[t["q_coeff"]["alpha"], t["q_coeff"]["beta"]]])
    clf.intercept_ = np.asarray([t["q_coeff"]["gamma"]])
    clf.classes_ = np.asarray([0, 1])  # sklearn internals
    return clf

# ------------------------------------------------------------------
# public API
# ------------------------------------------------------------------


def score_probe_vs_profile(
    probe: torch.Tensor,  # 192‑D
    profile_vecs: torch.Tensor,  # (N, 192)
) -> Tuple[float, float, float]:
    """Return (cosine, plda, q) scores against the *mean* of profile.

    * All scores are similarity (higher → closer).
    * Q ∈ [0,1] – logistic regression fusion.
    """
    probe_n = _l2_normalize(probe.unsqueeze(0))
    mean_vec = _l2_normalize(profile_vecs.mean(0, keepdim=True))

    cosine = float(torch.nn.functional.cosine_similarity(probe_n, mean_vec).item())

    plda = float(
        score_plda(
            _get_plda(),
            probe_n.cpu().numpy(),
            mean_vec.cpu().numpy(),
            mean_vec.cpu().numpy(),  # enrollment stats – simplified (µ)
        )[0]
    )

    q_model = _get_q_model()
    q = float(
        q_model.predict_proba(np.asarray([[cosine, plda]]), )[:, 1].item()
    )

    return cosine, plda, q


def tier_from_q(q: float) -> Literal["confidential", "basic", "no-match"]:
    t = load_thresholds()
    if q >= t["tiers"]["confidential"]:
        return "confidential"
    if q >= t["tiers"]["basic"]:
        return "basic"
    return "no-match"


def verify(
    probe: torch.Tensor,
    profile_vecs: torch.Tensor,
) -> dict:
    cos, plda, q = score_probe_vs_profile(probe, profile_vecs)
    tier = tier_from_q(q)
    return {
        "cosine": cos,
        "plda": plda,
        "q": q,
        "tier": tier,
        "match": tier != "no-match",
    }
