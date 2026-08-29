"""Phase 4 — build the C0..C9 classifier-head inputs from cached embeddings.

  * gather per-pair ``u`` and ``v`` (fp16) and keep them resident on the run
    device (GPU when available),
  * per mini-batch, build blocks  ``u``, ``v``, ``diff = |u-v|``, ``prod = u*v``,
    ``absprod = |u*v|``, ``rand = concat(u,v) @ W_r``  (W_r ~ N(0, 1/1536),
    768 cols, drawn from the run seed; C4 and C9 share it) — all in torch,
  * standardize each block per-dim using **training-split** statistics,
  * concatenate the condition's blocks in order.

Everything is torch so the per-batch math runs on the GPU. See PROTOCOL.md
sections 4 and 7.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

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


def d_in(condition: str) -> int:
    return EMB_DIM * len(CONDITIONS[condition])


# --------------------------------------------------------------------------- load
def load_pair_embeddings(embed_dir: Path, dataset: str, split: str
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, label): u, v as fp16 [n_pairs, 768], label int64."""
    path = Path(embed_dir) / dataset / f"{split}.npz"
    with np.load(path) as z:
        uniq = z["uniq_emb"]
        u = np.ascontiguousarray(uniq[z["idx_a"]]).astype(np.float16)
        v = np.ascontiguousarray(uniq[z["idx_b"]]).astype(np.float16)
        label = z["label"].astype(np.int64)
    return u, v, label


# ------------------------------------------------------------------- random proj
def make_rand_projection(seed: int) -> np.ndarray:
    """W_r ~ N(0, 1/1536), shape [2*768, 768]. First draw from the seed's stream
    so C4 and C9 (same seed) get the same matrix."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, np.sqrt(1.0 / (2 * EMB_DIM)),
                      size=(2 * EMB_DIM, EMB_DIM)).astype(np.float32)


# ------------------------------------------------------------------------ blocks
def build_block(name: str, u: torch.Tensor, v: torch.Tensor,
                w_r: torch.Tensor | None) -> torch.Tensor:
    """u, v: fp32 [n, 768] on some device. Returns fp32 [n, 768]."""
    if name == "u":
        return u
    if name == "v":
        return v
    if name == "diff":
        return (u - v).abs()
    if name == "prod":
        return u * v
    if name == "absprod":
        return (u * v).abs()
    if name == "rand":
        return torch.cat([u, v], dim=1) @ w_r
    raise KeyError(name)


class BlockStandardizer:
    """Per-dimension z-score for each block, fit on the training split.
    Holds mean/std as fp32 tensors on the run device."""

    def __init__(self) -> None:
        self.stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def fit(self, u16: torch.Tensor, v16: torch.Tensor, block_names: list[str],
            w_r: torch.Tensor | None, chunk: int = 200_000) -> "BlockStandardizer":
        device = u16.device
        n = u16.shape[0]
        acc = {b: [torch.zeros(EMB_DIM, dtype=torch.float64, device=device),
                   torch.zeros(EMB_DIM, dtype=torch.float64, device=device)]
               for b in block_names}
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            u = u16[s:e].float()
            v = v16[s:e].float()
            for b in block_names:
                x = build_block(b, u, v, w_r).double()
                acc[b][0] += x.sum(0)
                acc[b][1] += (x * x).sum(0)
        for b, (ssum, sqsum) in acc.items():
            mean = ssum / n
            var = torch.clamp(sqsum / n - mean * mean, min=0.0)
            self.stats[b] = (mean.float(), torch.sqrt(var + _STD_EPS).float())
        return self

    def apply(self, name: str, x: torch.Tensor) -> torch.Tensor:
        mean, std = self.stats[name]
        return (x - mean) / std


def fit_standardizer(condition: str, u16: torch.Tensor, v16: torch.Tensor,
                     w_r: torch.Tensor | None) -> BlockStandardizer:
    return BlockStandardizer().fit(u16, v16, CONDITIONS[condition], w_r)


# ---------------------------------------------------------------------- assemble
def assemble(condition: str, u32: torch.Tensor, v32: torch.Tensor,
             w_r: torch.Tensor | None, std: BlockStandardizer) -> torch.Tensor:
    """Standardized feature matrix for a batch, fp32 [n, d_in(condition)]."""
    parts = [std.apply(b, build_block(b, u32, v32, w_r)) for b in CONDITIONS[condition]]
    return torch.cat(parts, dim=1)
