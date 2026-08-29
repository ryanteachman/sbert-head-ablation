"""Phase 5 / 6 — run the condition grid.

  Pilot:  python src/run_grid.py --pilot --embed-dir <dir>
          3 seeds x linear head x 10 conditions x 3 datasets = 90 cells

  Full:   python src/run_grid.py --embed-dir <dir>
          10 seeds x {linear, mlp} x 10 conditions x 3 datasets = 600 cells

For each (dataset, condition) the standardized feature matrices are built **once**
(x_train as a disk memmap) and reused by every seed/head cell — the per-batch
rebuild is ~90% of the cost otherwise. `rand` conditions (C4, C9) rebuild per
seed since W_r is seed-dependent.

Resumable: cells already in the output parquet are skipped; results flush after
every cell.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from features import (CONDITIONS, fit_standardizer, load_pair_embeddings,
                      make_rand_projection, standardized_matrix)
from train import FeatureSet, train_one

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "experiment.yaml"
CELL_KEYS = ["dataset", "condition", "head", "seed"]
NLI_EXTRA = ["mnli_val_matched", "mnli_val_mismatched"]


def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _device(req: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if req == "auto" else req


def _feature_set(condition, w_r, std, raw: dict, work_dir: Path, tag: str) -> FeatureSet:
    """Build standardized matrices for every split of one (dataset, condition[, seed])."""
    def mat(split, out_path=None):
        u16, v16, _ = raw[split]
        return standardized_matrix(condition, u16, v16, w_r, std, out_path=out_path)

    mm_path = work_dir / f"{tag}_train.npy"
    return FeatureSet(
        x_train=mat("train", out_path=str(mm_path)), y_train=raw["train"][2],
        x_val=mat("val"), y_val=raw["val"][2],
        x_test=mat("test"), y_test=raw["test"][2],
        extra={k: (mat(k), raw[k][2]) for k in raw if k in NLI_EXTRA},
    )


def _free(feats: FeatureSet, work_dir: Path, tag: str) -> None:
    del feats
    gc.collect()
    p = work_dir / f"{tag}_train.npy"
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass                                   # Windows: memmap handle may linger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="3 seeds, linear head only")
    ap.add_argument("--embed-dir", required=True, help="dir with <dataset>/<split>.npz")
    ap.add_argument("--out", default=str(ROOT / "results" / "runs.parquet"))
    ap.add_argument("--work-dir", default=str(ROOT / ".featcache"),
                    help="scratch dir for x_train memmaps (needs ~12 GB free)")
    ap.add_argument("--datasets", default="nli,qqp,paws")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--heads", default=None, help="override; default linear (pilot) or linear,mlp")
    ap.add_argument("--seeds", default=None, help="override, comma list")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=None, help="stop after N new cells")
    args = ap.parse_args()

    cfg = _cfg()
    device = _device(args.device)
    embed_dir = Path(args.embed_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    heads = ([h.strip() for h in args.heads.split(",")] if args.heads
             else (["linear"] if args.pilot else ["linear", "mlp"]))
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else (cfg["pilot_seeds"] if args.pilot else cfg["seed_base"]))

    done: set[tuple] = set()
    rows: list[dict] = []
    if out.exists():
        prev = pd.read_parquet(out)
        rows = prev.to_dict("records")
        done = {tuple(r[k] for k in CELL_KEYS) for r in prev[CELL_KEYS].to_dict("records")}
        print(f"resuming — {len(done)} cells already done")

    planned = [(d, c, h, s) for d in datasets for c in conditions for h in heads for s in seeds]
    todo = [cell for cell in planned if cell not in done]
    print(f"device: {device} | planned {len(planned)} | remaining {len(todo)}"
          + (f" | limit {args.limit}" if args.limit else ""), flush=True)

    n_new = 0
    stop = False
    for dataset in datasets:
        ds_cells = [c for c in todo if c[0] == dataset]
        if not ds_cells or stop:
            continue
        n_classes = int(cfg["datasets"][dataset]["num_classes"])
        splits = ["train", "val", "test"] + (NLI_EXTRA if dataset == "nli" else [])
        print(f"\n=== {dataset} ({n_classes}-way) — loading embeddings ===", flush=True)
        raw = {s: load_pair_embeddings(embed_dir, dataset, s) for s in splits}
        print("    " + " | ".join(f"{s} {len(raw[s][2]):,}" for s in splits), flush=True)

        for condition in conditions:
            cond_cells = [(h, s) for (_, c, h, s) in ds_cells if c == condition]
            if not cond_cells:
                continue
            has_rand = "rand" in CONDITIONS[condition]
            groups = ({sd: [(h, s) for (h, s) in cond_cells if s == sd]
                       for sd in sorted({s for _, s in cond_cells})}
                      if has_rand else {None: cond_cells})

            for gseed, gcells in groups.items():
                w_r = make_rand_projection(gseed) if has_rand else None
                std = fit_standardizer(condition, raw["train"][0], raw["train"][1], w_r)
                tag = f"{dataset}_{condition}_{gseed}"
                feats = _feature_set(condition, w_r, std, raw, work_dir, tag)
                print(f"  [{dataset}/{condition}"
                      + (f" seed {gseed}" if has_rand else "")
                      + f"] features ready (d_in={feats.x_train.shape[1]}), {len(gcells)} cells",
                      flush=True)

                for (head, seed) in gcells:
                    row = train_one(cfg, dataset=dataset, condition=condition,
                                    head_kind=head, seed=seed, n_classes=n_classes,
                                    feats=feats, device=device)
                    row["test_confusion"] = json.dumps(row["test_confusion"])
                    rows.append(row)
                    pd.DataFrame(rows).to_parquet(out, index=False)
                    n_new += 1
                    print(f"    {condition:<3} {head:<6} s{seed}  acc={row['test_acc']:.4f}"
                          f" f1={row['test_macro_f1']:.4f}  [{row['epochs_trained']}ep"
                          f" {row['wall_s']}s]  ({n_new}/{len(todo)})", flush=True)
                    if args.limit and n_new >= args.limit:
                        stop = True
                        break

                _free(feats, work_dir, tag)
                if stop:
                    break
            if stop:
                break

        del raw
        gc.collect()

    for leftover in work_dir.glob("*_train.npy"):
        try:
            leftover.unlink()
        except OSError:
            pass

    print(f"\nwrote {out}  ({len(rows)} rows total, {n_new} new)")
    if n_new:
        df = pd.DataFrame(rows)
        lin = df[df["head"] == "linear"]
        if not lin.empty:
            piv = lin.pivot_table(index="dataset", columns="condition",
                                  values="test_acc", aggfunc="mean").round(4)
            print("\nmean linear-head test accuracy (so far):")
            print(piv.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
