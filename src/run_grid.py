"""Phase 5 / 6 — run the condition grid.

  Pilot:  python src/run_grid.py --pilot --embed-dir <dir>
          3 seeds x linear head x 10 conditions x 3 datasets = 90 cells

  Full:   python src/run_grid.py --embed-dir <dir>
          15 seeds x {linear, mlp} x 10 conditions x 3 datasets = 900 cells

Per dataset, ``u``/``v`` for every split are loaded once onto the run device
(GPU when available). Features are then built per mini-batch in torch — the
block math is free on a GPU. Standardizer stats are fit once per
(dataset, condition[, seed if `rand`]).

Resumable: cells already in the output parquet are skipped; results flush after
every cell.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from features import CONDITIONS, fit_standardizer, load_pair_embeddings, make_rand_projection
from train import SplitTensors, train_one

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


def _load_tensors(embed_dir: Path, dataset: str, split: str, device: str) -> SplitTensors:
    u16, v16, label = load_pair_embeddings(embed_dir, dataset, split)
    return SplitTensors(
        u=torch.from_numpy(u16).to(device),
        v=torch.from_numpy(v16).to(device),
        y=label,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="3 seeds, linear head only")
    ap.add_argument("--embed-dir", required=True, help="dir with <dataset>/<split>.npz")
    ap.add_argument("--out", default=str(ROOT / "results" / "runs.parquet"))
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
        print(f"\n=== {dataset} ({n_classes}-way) — loading embeddings to {device} ===", flush=True)
        T = {s: _load_tensors(embed_dir, dataset, s, device) for s in splits}
        extra = {k: T[k] for k in NLI_EXTRA if k in T}
        print("    " + " | ".join(f"{s} {len(T[s].y):,}" for s in splits), flush=True)

        for condition in conditions:
            cond_cells = [(h, s) for (_, c, h, s) in ds_cells if c == condition]
            if not cond_cells:
                continue
            has_rand = "rand" in CONDITIONS[condition]
            groups = ({sd: [(h, s) for (h, s) in cond_cells if s == sd]
                       for sd in sorted({s for _, s in cond_cells})}
                      if has_rand else {None: cond_cells})

            for gseed, gcells in groups.items():
                w_r = (torch.from_numpy(make_rand_projection(gseed)).to(device)
                       if has_rand else None)
                std = fit_standardizer(condition, T["train"].u, T["train"].v, w_r)
                print(f"  [{dataset}/{condition}"
                      + (f" seed {gseed}" if has_rand else "")
                      + f"]  {len(gcells)} cells", flush=True)

                for (head, seed) in gcells:
                    row = train_one(cfg, dataset=dataset, condition=condition,
                                    head_kind=head, seed=seed, n_classes=n_classes,
                                    tr=T["train"], va=T["val"], te=T["test"],
                                    extra=extra, w_r=w_r, std=std, device=device)
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
                del std, w_r
                if device == "cuda":
                    torch.cuda.empty_cache()
                if stop:
                    break
            if stop:
                break

        del T, extra
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nwrote {out}  ({len(rows)} rows total, {n_new} new)")
    if n_new:
        df = pd.DataFrame(rows)
        for h in [x for x in ("linear", "mlp") if x in df["head"].unique()]:
            piv = df[df["head"] == h].pivot_table(index="dataset", columns="condition",
                                                  values="test_acc", aggfunc="mean").round(4)
            print(f"\nmean {h}-head test accuracy (so far):")
            print(piv.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
