"""Phase 5 / 6 — run the condition grid.

  Pilot:  python src/run_grid.py --pilot --embed-dir <dir>
          3 seeds x linear head x 10 conditions x 3 datasets = 90 cells

  Full:   python src/run_grid.py --embed-dir <dir>
          10 seeds x {linear, mlp} x 10 conditions x 3 datasets = 600 cells

Resumable: cells already present in the output parquet are skipped. Results are
flushed after every cell.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from features import CONDITIONS
from train import SplitData, clear_std_cache, train_one

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


def _load_split(embed_dir: Path, dataset: str, split: str) -> SplitData:
    from features import load_pair_embeddings
    u16, v16, label = load_pair_embeddings(embed_dir, dataset, split)
    return SplitData(u16=u16, v16=v16, label=label)


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
          + (f" | limit {args.limit}" if args.limit else ""))

    n_new = 0
    for dataset in datasets:
        cells = [c for c in todo if c[0] == dataset]
        if not cells:
            continue
        n_classes = int(cfg["datasets"][dataset]["num_classes"])
        print(f"\n=== {dataset} ({n_classes}-way) — loading embeddings ===")
        train = _load_split(embed_dir, dataset, "train")
        val = _load_split(embed_dir, dataset, "val")
        test = _load_split(embed_dir, dataset, "test")
        extra = ({k: _load_split(embed_dir, dataset, k) for k in NLI_EXTRA}
                 if dataset == "nli" else None)
        print(f"    train {len(train.label):,} | val {len(val.label):,} | test {len(test.label):,}")

        for (_, condition, head, seed) in cells:
            if args.limit and n_new >= args.limit:
                break
            t = time.time()
            row = train_one(cfg, dataset=dataset, condition=condition, head_kind=head,
                            seed=seed, n_classes=n_classes, train=train, val=val,
                            test=test, extra_eval=extra, device=device)
            row["test_confusion"] = json.dumps(row["test_confusion"])
            rows.append(row)
            pd.DataFrame(rows).to_parquet(out, index=False)
            n_new += 1
            print(f"  {dataset:<4} {condition:<3} {head:<6} s{seed}  "
                  f"acc={row['test_acc']:.4f} f1={row['test_macro_f1']:.4f}  "
                  f"[{row['epochs_trained']}ep {row['wall_s']}s]  ({n_new}/{len(todo)})")

        del train, val, test, extra
        clear_std_cache()
        gc.collect()
        if args.limit and n_new >= args.limit:
            break

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
