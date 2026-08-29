"""Phase 4 — train and evaluate one classifier head (a single grid cell).

``train_one`` is deterministic given ``seed``: it controls head initialization
and the training-data shuffle order. For a fixed seed the init and shuffle are
identical across conditions, so per-seed metric differences are paired.
See PROTOCOL.md sections 9-10.

Features are built per mini-batch, in torch, on the run device — with ``u``/``v``
resident on the GPU the block math (`|u-v|`, `u*v`, the projection,
standardization) is effectively free, so a full grid cell is seconds.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from features import CONDITIONS, assemble, d_in
from heads import make_head, weight_decay_for


@dataclass
class SplitTensors:
    u: torch.Tensor          # [n, 768] fp16, on the run device
    v: torch.Tensor
    y: np.ndarray            # [n] int64 (CPU)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=np.int64)
    np.add.at(m, (y_true, y_pred), 1)
    return m


@torch.no_grad()
def _evaluate(head: nn.Module, sd: SplitTensors, condition: str, w_r, std,
              n_classes: int, batch: int = 16384) -> dict:
    head.eval()
    n = len(sd.y)
    ce = nn.CrossEntropyLoss(reduction="sum")
    preds = np.empty(n, dtype=np.int64)
    total_loss = 0.0
    for s in range(0, n, batch):
        e = min(s + batch, n)
        x = assemble(condition, sd.u[s:e].float(), sd.v[s:e].float(), w_r, std)
        logits = head(x)
        yb = torch.from_numpy(sd.y[s:e]).to(x.device)
        total_loss += ce(logits, yb).item()
        preds[s:e] = logits.argmax(1).cpu().numpy()
    head.train()

    acc = float((preds == sd.y).mean())
    out = {
        "acc": acc,
        "loss": total_loss / n,
        "macro_f1": float(f1_score(sd.y, preds, average="macro")),
        "confusion": [[int(v) for v in row] for row in _confusion(sd.y, preds, n_classes)],
    }
    if n_classes == 2:
        out["pos_f1"] = float(f1_score(sd.y, preds, pos_label=1, average="binary"))
    return out


def train_one(cfg: dict, *, dataset: str, condition: str, head_kind: str, seed: int,
              n_classes: int, tr: SplitTensors, va: SplitTensors, te: SplitTensors,
              extra: dict[str, SplitTensors], w_r, std, device: str = "cpu") -> dict:
    t0 = time.time()
    tcfg = cfg["train"]
    set_determinism(seed)

    din = d_in(condition)
    head = make_head(head_kind, din, n_classes, cfg).to(device)   # init consumes the seed

    opt = torch.optim.AdamW(
        head.parameters(), lr=float(tcfg["lr"]),
        betas=tuple(tcfg["betas"]), eps=float(tcfg["eps"]),
        weight_decay=weight_decay_for(head_kind, cfg),
    )
    ce = nn.CrossEntropyLoss()
    clip = float(tcfg["grad_clip_norm"])
    batch = int(tcfg["batch_size"])
    n = len(tr.y)
    y_train = torch.from_numpy(tr.y).to(device)
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
        perm = torch.randperm(n, generator=gen).to(device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            x = assemble(condition, tr.u[idx].float(), tr.v[idx].float(), w_r, std)
            loss = ce(head(x), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), clip)
            opt.step()
            step += 1

            if step % eval_every == 0:
                vm = _evaluate(head, va, condition, w_r, std, n_classes)
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

    tm = _evaluate(head, te, condition, w_r, std, n_classes)
    row = {
        "dataset": dataset, "condition": condition, "head": head_kind, "seed": seed,
        "d_in": din, "n_train": n, "n_test": len(te.y),
        "test_acc": tm["acc"], "test_loss": tm["loss"], "test_macro_f1": tm["macro_f1"],
        "test_pos_f1": tm.get("pos_f1"), "test_confusion": tm["confusion"],
        "val_acc_best": best["val_acc"], "val_loss_best": best["val_loss"],
        "best_step": best["step"], "epochs_trained": epochs_done,
        "early_stopped": stopped, "wall_s": round(time.time() - t0, 1),
    }
    for name, sd in extra.items():
        em = _evaluate(head, sd, condition, w_r, std, n_classes)
        row[f"{name}_acc"] = em["acc"]
        row[f"{name}_macro_f1"] = em["macro_f1"]
    return row
