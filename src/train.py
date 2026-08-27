"""Phase 4 — train and evaluate one classifier head (a single grid cell).

``train_one`` is deterministic given ``seed``: it controls head initialization,
the training-data shuffle order, and (for C4 / C9) the random projection. For a
fixed seed the init and shuffle are identical across conditions, so per-seed
metric differences are paired. See PROTOCOL.md sections 9-10.

Features are built per mini-batch (never materialized in full — a standardized
NLI-train matrix at d_in=3072 would be ~12 GB).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from features import BlockStandardizer, CONDITIONS, build_block, d_in, make_rand_projection
from heads import make_head, weight_decay_for


@dataclass
class SplitData:
    u16: np.ndarray          # [n, 768] fp16
    v16: np.ndarray
    label: np.ndarray        # [n] int64


# Standardizer cache: non-`rand` blocks are seed-independent, so the fit over the
# (large) training split is reused across seeds. Keyed by (dataset, condition,
# seed-if-rand-else-None). Call clear_std_cache() when moving to a new dataset.
_STD_CACHE: dict = {}


def clear_std_cache() -> None:
    _STD_CACHE.clear()


def _get_standardizer(dataset: str, condition: str, seed: int, w_r,
                      train: "SplitData") -> "BlockStandardizer":
    key = (dataset, condition, seed if "rand" in CONDITIONS[condition] else None)
    if key not in _STD_CACHE:
        _STD_CACHE[key] = BlockStandardizer().fit(
            train.u16, train.v16, CONDITIONS[condition], w_r)
    return _STD_CACHE[key]


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _feature_builder(condition: str, w_r, std: BlockStandardizer):
    blocks = CONDITIONS[condition]

    def build(u16: np.ndarray, v16: np.ndarray, idx: np.ndarray) -> np.ndarray:
        u = u16[idx].astype(np.float32)
        v = v16[idx].astype(np.float32)
        parts = [std.apply(b, build_block(b, u, v, w_r)) for b in blocks]
        return np.concatenate(parts, axis=1)

    return build


@torch.no_grad()
def _evaluate(head: nn.Module, split: SplitData, build, device: str,
              n_classes: int, batch: int = 8192) -> dict:
    head.eval()
    n = len(split.label)
    ce = nn.CrossEntropyLoss(reduction="sum")
    preds = np.empty(n, dtype=np.int64)
    total_loss = 0.0
    for s in range(0, n, batch):
        idx = np.arange(s, min(s + batch, n))
        x = torch.from_numpy(build(split.u16, split.v16, idx)).to(device)
        y = torch.from_numpy(split.label[idx]).to(device)
        logits = head(x)
        total_loss += ce(logits, y).item()
        preds[idx] = logits.argmax(1).cpu().numpy()
    head.train()

    y_true = split.label
    acc = float((preds == y_true).mean())
    out = {
        "acc": acc,
        "loss": total_loss / n,
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "confusion": [[int(x) for x in row]
                      for row in _confusion(y_true, preds, n_classes)],
    }
    if n_classes == 2:
        out["pos_f1"] = float(f1_score(y_true, preds, pos_label=1, average="binary"))
    return out


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=np.int64)
    np.add.at(m, (y_true, y_pred), 1)
    return m


def train_one(cfg: dict, *, dataset: str, condition: str, head_kind: str, seed: int,
              n_classes: int, train: SplitData, val: SplitData, test: SplitData,
              extra_eval: dict[str, SplitData] | None = None,
              device: str = "cpu") -> dict:
    t0 = time.time()
    tcfg = cfg["train"]
    set_determinism(seed)

    din = d_in(condition)
    head = make_head(head_kind, din, n_classes, cfg).to(device)   # init uses the seed

    w_r = make_rand_projection(seed) if "rand" in CONDITIONS[condition] else None
    std = _get_standardizer(dataset, condition, seed, w_r, train)
    build = _feature_builder(condition, w_r, std)

    opt = torch.optim.AdamW(
        head.parameters(), lr=float(tcfg["lr"]),
        betas=tuple(tcfg["betas"]), eps=float(tcfg["eps"]),
        weight_decay=weight_decay_for(head_kind, cfg),
    )
    ce = nn.CrossEntropyLoss()
    clip = float(tcfg["grad_clip_norm"])
    batch = int(tcfg["batch_size"])
    n = len(train.label)
    steps_per_epoch = (n + batch - 1) // batch
    eval_every = max(1, int(steps_per_epoch * float(tcfg["eval_every_frac_epoch"])))
    patience = int(tcfg["early_stopping"]["patience_evals"])

    gen = torch.Generator().manual_seed(seed)
    best = {"val_acc": -1.0, "val_loss": np.inf, "state": None, "step": 0}
    since_improve = 0
    step = 0
    stopped = False
    epochs_done = 0

    for epoch in range(int(tcfg["max_epochs"])):
        epochs_done = epoch + 1
        perm = torch.randperm(n, generator=gen).numpy()
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            x = torch.from_numpy(build(train.u16, train.v16, idx)).to(device)
            y = torch.from_numpy(train.label[idx]).to(device)
            loss = ce(head(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), clip)
            opt.step()
            step += 1

            if step % eval_every == 0:
                vm = _evaluate(head, val, build, device, n_classes)
                improved = (vm["acc"] > best["val_acc"] + 1e-6) or (
                    abs(vm["acc"] - best["val_acc"]) <= 1e-6 and vm["loss"] < best["val_loss"])
                if improved:
                    best.update(val_acc=vm["acc"], val_loss=vm["loss"], step=step,
                                state={k: v.detach().cpu().clone()
                                       for k, v in head.state_dict().items()})
                    since_improve = 0
                else:
                    since_improve += 1
                    if since_improve >= patience:
                        stopped = True
                        break
        if stopped:
            break

    if best["state"] is not None:
        head.load_state_dict(best["state"])

    tm = _evaluate(head, test, build, device, n_classes)
    row = {
        "dataset": dataset, "condition": condition, "head": head_kind, "seed": seed,
        "d_in": din, "n_train": n, "n_test": len(test.label),
        "test_acc": tm["acc"], "test_loss": tm["loss"], "test_macro_f1": tm["macro_f1"],
        "test_pos_f1": tm.get("pos_f1"), "test_confusion": tm["confusion"],
        "val_acc_best": best["val_acc"], "val_loss_best": best["val_loss"],
        "best_step": best["step"], "epochs_trained": epochs_done,
        "early_stopped": stopped, "wall_s": round(time.time() - t0, 1),
    }
    for name, sd in (extra_eval or {}).items():
        em = _evaluate(head, sd, build, device, n_classes)
        row[f"{name}_acc"] = em["acc"]
        row[f"{name}_macro_f1"] = em["macro_f1"]
    return row
