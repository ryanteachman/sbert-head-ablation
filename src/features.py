"""Phase 4 — build the C0..C9 classifier-head inputs from cached embeddings.

Given cached unique-sentence embeddings, this module:
  * gathers per-pair ``u`` and ``v`` (fp16 in memory, upcast to fp32 per batch),
  * builds feature blocks  ``u``, ``v``, ``diff = |u-v|``, ``prod = u*v``,
    ``absprod = |u*v|``, ``rand = concat(u,v) @ W_r``  (W_r ~ N(0, 1/1536),
    768 cols, drawn from the run seed; C4 and C9 share it),
  * standardizes each block per-dim using **training-split** statistics,
  * assembles a condition's feature matrix by concatenating its blocks in order.

See PROTOCOL.md sections 4 and 7.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

EMB_DIM = 768
_STD_EPS = 1e-6

# Condition -> ordered block list. Mirrors config/experiment.yaml `conditions`.
CONDITIONS: dict[str, list[str]] = {
    "C0": ["u", "v"],
    "C1": ["u", "v", "diff"],
    "C2": ["u", "v", "prod"],
    "C3": ["u", "v", "diff", "prod"],
    "C4": ["u", "v", "diff", "rand"],
    "C5": ["diff", "prod"],
    "C6": ["diff"],
    "C7": ["prod"],
    "C8": ["u", "v", "diff", "absprod"],
    "C9": ["u", "v", "prod", "rand"],
}

# Blocks that depend on the per-seed random projection.
SEED_DEPENDENT_BLOCKS = {"rand"}


def blocks_for(conditions: list[str]) -> list[str]:
    """Union of blocks needed by the given conditions, in a stable order."""
    order = ["u", "v", "diff", "prod", "absprod", "rand"]
    needed = {b for c in conditions for b in CONDITIONS[c]}
    return [b for b in order if b in needed]


def d_in(condition: str) -> int:
    return EMB_DIM * len(CONDITIONS[condition])


# --------------------------------------------------------------------------- load
def load_pair_embeddings(embed_dir: Path, dataset: str, split: str
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, label) with u, v as fp16 [n_pairs, 768] and label int64."""
    path = Path(embed_dir) / dataset / f"{split}.npz"
    with np.load(path) as z:
        uniq = z["uniq_emb"]                       # [n_unique, 768], fp16 or fp32
        u = np.ascontiguousarray(uniq[z["idx_a"]])
        v = np.ascontiguousarray(uniq[z["idx_b"]])
        label = z["label"].astype(np.int64)
    return u.astype(np.float16), v.astype(np.float16), label


# ------------------------------------------------------------------- random proj
def make_rand_projection(seed: int) -> np.ndarray:
    """W_r ~ N(0, 1/1536), shape [2*768, 768]. First draw from the seed's stream
    so C4 and C9 (same seed) get the same matrix."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, np.sqrt(1.0 / (2 * EMB_DIM)),
                      size=(2 * EMB_DIM, EMB_DIM)).astype(np.float32)


# ------------------------------------------------------------------------ blocks
def build_block(name: str, u: np.ndarray, v: np.ndarray,
                w_r: np.ndarray | None) -> np.ndarray:
    """u, v are fp32 [n, 768]. Returns fp32 [n, 768]."""
    if name == "u":
        return u
    if name == "v":
        return v
    if name == "diff":
        return np.abs(u - v)
    if name == "prod":
        return u * v
    if name == "absprod":
        return np.abs(u * v)
    if name == "rand":
        return np.concatenate([u, v], axis=1) @ w_r
    raise KeyError(name)


class BlockStandardizer:
    """Per-dimension z-score for each block, fit on the training split."""

    def __init__(self) -> None:
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, u16: np.ndarray, v16: np.ndarray, block_names: list[str],
            w_r: np.ndarray | None, chunk: int = 100_000) -> "BlockStandardizer":
        n = len(u16)
        acc = {b: (np.zeros(EMB_DIM), np.zeros(EMB_DIM)) for b in block_names}
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            u = u16[s:e].astype(np.float32)
            v = v16[s:e].astype(np.float32)
            for b in block_names:
                x = build_block(b, u, v, w_r)
                acc[b][0][:] += x.sum(axis=0)
                acc[b][1][:] += (x.astype(np.float64) ** 2).sum(axis=0)
        for b, (ssum, sqsum) in acc.items():
            mean = ssum / n
            var = np.maximum(sqsum / n - mean ** 2, 0.0)
            self.stats[b] = (mean.astype(np.float32),
                             np.sqrt(var + _STD_EPS).astype(np.float32))
        return self

    def apply(self, name: str, x: np.ndarray) -> np.ndarray:
        mean, std = self.stats[name]
        return (x - mean) / std


# ---------------------------------------------------------------------- assemble
def assemble(condition: str, u32: np.ndarray, v32: np.ndarray,
             w_r: np.ndarray | None, std: BlockStandardizer) -> np.ndarray:
    """Standardized feature matrix for `condition`, fp32 [n, d_in(condition)]."""
    parts = [std.apply(b, build_block(b, u32, v32, w_r)) for b in CONDITIONS[condition]]
    return np.concatenate(parts, axis=1).astype(np.float32)
