"""Train cosine + PLDA fusion thresholds for SVAS.

* extracts ECAPA x‑vectors for a (sub)sample of VoxCeleb‑1
* trains PLDA (speechbrain.processing.PLDA_LDA)
* fits 2‑feature logistic regression `[cosine, plda] → Q`
* chooses tier thresholds at FAR 0.2 % and 1 %

Run example
-----------
```bash
python services/worker/tools/train_thresholds.py \
       --dataset-root ~/Datasets/voxceleb1/wav \
       --sample-fraction 0.3        # quick dev run (30 %)
```
Outputs `core/plda.pkl` and `core/thresholds.yml`.
Works on SpeechBrain ≥ 0.5.0 (same API in 1.x).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from tqdm import tqdm

from core import extract_embedding, preprocess
from speechbrain.processing.PLDA_LDA import (
    PLDA,
    StatObject_SB,
    fast_PLDA_scoring,
)

###############################################################################
# helpers
###############################################################################

def _collect_pairs(root: Path, frac: float) -> list[tuple[Path, Path, bool]]:
    """Return [(wav_a, wav_b, same_speaker)]."""
    spk2wavs: dict[str, list[Path]] = {}
    for wav in root.rglob("*.wav"):
        spk_id = wav.parent.parent.name  # idXXXXX/video/00001.wav
        spk2wavs.setdefault(spk_id, []).append(wav)

    rng = np.random.default_rng(42)
    same, diff = [], []

    for spk, files in spk2wavs.items():
        if len(files) < 2:
            continue
        a, b = rng.choice(files, 2, replace=False)
        same.append((a, b, True))

    spk_ids = list(spk2wavs)
    for _ in range(len(same)):
        s1, s2 = rng.choice(spk_ids, 2, replace=False)
        a = rng.choice(spk2wavs[s1])
        b = rng.choice(spk2wavs[s2])
        diff.append((a, b, False))

    pairs = same + diff
    if frac < 1.0:
        pairs = rng.choice(pairs, int(len(pairs) * frac), replace=False).tolist()
    return pairs


def _to_stat(vec: np.ndarray) -> StatObject_SB:
    """Wrap single x‑vector for PLDA utils."""
    return StatObject_SB(
        modelset=np.array(["m"], dtype=object),
        segset=np.array(["s"], dtype=object),
        start=np.array([None]),
        stop=np.array([None]),
        stat0=np.ones((1, 1), dtype=float),
        stat1=vec.astype(float),
    )

# one dummy Ndx object (PLDA scorer inspects only .modelset / .segset)
class _OnePairNdx:
    """Stub Ndx with minimal attrs used by PLDA utils."""

    def __init__(self):
        self.modelset = np.array(["m"], dtype=object)
        self.segset = np.array(["s"], dtype=object)
        self.trialmask = np.ones((1, 1), dtype=bool)  # required by fast_PLDA_scoring

    def filter(self, *_, **__) -> "_OnePairNdx":
        return self

_ONE_NDX = _OnePairNdx()

###############################################################################
# main script
###############################################################################

def main() -> None:  # noqa: C901 – fine for CLI
    ap = argparse.ArgumentParser("Train SVAS thresholds")
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("core/thresholds.yml"))
    ap.add_argument("--plda-out", type=Path, default=Path("core/plda.pkl"))
    ap.add_argument("--sample-fraction", type=float, default=1.0)
    args = ap.parse_args()

    # 1) pairs
    print("Collecting pairs …")
    pairs = _collect_pairs(args.dataset_root, args.sample_fraction)
    g = sum(s for *_, s in pairs)
    print(f"Pairs: {len(pairs)} (genuine {g}, impostor {len(pairs)-g})")

    # 2) embeddings
    embs: dict[Path, torch.Tensor] = {}
    for wav in tqdm({w for p in pairs for w in p[:2]}, desc="embeddings"):
        segs = preprocess(wav, denoise=False)
        embs[wav] = torch.stack([extract_embedding(s) for s in segs]).mean(0)

    def _vec(p: Path) -> torch.Tensor:  # 1×192
        return embs[p].unsqueeze(0)

    # cosine array
    X_cos, y = [], []
    for a, b, same in tqdm(pairs, desc="cosine"):
        X_cos.append(float(torch.nn.functional.cosine_similarity(_vec(a), _vec(b)).item()))
        y.append(int(same))
    X_cos = np.asarray(X_cos)
    y = np.asarray(y)

    # 3) train PLDA on all x‑vectors
    print("Training PLDA …")
    plda = PLDA()
    all_vecs = torch.stack(list(embs.values())).numpy()
    stat_obj = StatObject_SB(
        modelset=np.arange(all_vecs.shape[0], dtype=object),
        segset=np.arange(all_vecs.shape[0], dtype=object),
        start=np.array([None] * all_vecs.shape[0]),
        stop=np.array([None] * all_vecs.shape[0]),
        stat0=np.ones((all_vecs.shape[0], 1), dtype=float),
        stat1=all_vecs.astype(float),
    )
    plda.plda(stat_obj)

    # 4) PLDA scores for pairs
    X_plda = []
    for a, b, _ in tqdm(pairs, desc="plda"):
        s_mod = _to_stat(_vec(a).numpy())
        s_seg = _to_stat(_vec(b).numpy())
        scores = fast_PLDA_scoring(s_mod, s_seg, _ONE_NDX,
                           plda.mean, plda.F, plda.Sigma)
        sc = float(scores.scoremat[0, 0])
        X_plda.append(float(sc))
    X_plda = np.asarray(X_plda)

    # 5) logistic fusion [cos, plda] → Q
    X_fuse = np.column_stack([X_cos, X_plda])
    clf = LogisticRegression(max_iter=1000).fit(X_fuse, y)
    q = clf.predict_proba(X_fuse)[:, 1]

    # 6) thresholds @ FAR 0.2 % and 1 %
    fpr, _, thr = roc_curve(y, q)
    q_conf = float(thr[np.argmin(np.abs(fpr - 0.002))])
    q_basic = float(thr[np.argmin(np.abs(fpr - 0.01))])

    cos_thr = float(np.percentile(X_cos[y == 0], 99))
    plda_thr = float(np.percentile(X_plda[y == 0], 1))

    # 7) save artefacts
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.plda_out, "wb") as f:
        pickle.dump(plda, f)

    yaml.safe_dump({
        "model": "ecapa-voxceleb",
        "train_date": "2025-05-XX",
        "cosine_thr": cos_thr,
        "plda_thr": plda_thr,
        "q_coeff": {
            "alpha": float(clf.coef_[0, 0]),
            "beta": float(clf.coef_[0, 1]),
            "gamma": float(clf.intercept_[0]),
        },
        "tiers": {"confidential": q_conf, "basic": q_basic},
    }, args.out.open("w"), sort_keys=False)

    print("✓ thresholds →", args.out)
    print("✓ PLDA model →", args.plda_out)


if __name__ == "__main__":
    main()
