"""Offline plumbing check for features -> heads -> train -> run_grid.

Fabricates a tiny synthetic embedding cache (label weakly tied to <u, v>), runs a
mini grid twice, and asserts: it produces rows, results are deterministic across
fresh runs, resume skips completed cells, and the interaction-feature conditions
beat the bare [u, v] baseline on the planted signal.

Not a substitute for the Phase 4 smoke test on real embeddings — this only
checks the code path. Run:  python scripts/smoke_pipeline.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def make_fake_cache(root: Path) -> None:
    for dataset, k in (("paws", 2), ("nli", 3)):
        rng = np.random.default_rng(0)
        base = rng.normal(size=(120, 768)).astype(np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        splits = {"train": 400, "val": 80, "test": 80}
        if dataset == "nli":
            splits |= {"mnli_val_matched": 40, "mnli_val_mismatched": 40}
        (root / dataset).mkdir(parents=True, exist_ok=True)
        for split, n in splits.items():
            ia = rng.integers(0, 120, n).astype(np.int32)
            ib = rng.integers(0, 120, n).astype(np.int32)
            dots = (base[ia] * base[ib]).sum(1)
            if k == 2:
                lab = (dots > np.median(dots)).astype(np.int64)
            else:
                lab = np.digitize(dots, np.quantile(dots, [1 / 3, 2 / 3])).astype(np.int64)
            np.savez(root / dataset / f"{split}.npz", uniq_emb=base.astype(np.float16),
                     idx_a=ia, idx_b=ib, label=lab)


def run_grid(embed_dir: Path, out: Path) -> pd.DataFrame:
    cmd = [sys.executable, str(ROOT / "src" / "run_grid.py"), "--pilot",
           "--embed-dir", str(embed_dir), "--out", str(out),
           "--work-dir", str(out.parent / "_featcache"),
           "--datasets", "paws,nli", "--conditions", "C0,C1,C3,C4,C9", "--seeds", "0,1"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return pd.read_parquet(out)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        emb = tmp / "emb"
        emb.mkdir()
        make_fake_cache(emb)

        a = run_grid(emb, tmp / "a.parquet")
        b = run_grid(emb, tmp / "b.parquet")

        assert len(a) == 20, f"expected 20 rows, got {len(a)}"
        key = ["dataset", "condition", "seed"]
        merged = a.merge(b, on=key, suffixes=("_a", "_b"))
        drift = (merged["test_acc_a"] - merged["test_acc_b"]).abs().max()
        assert drift == 0.0, f"non-deterministic: max test_acc drift {drift}"

        # resume: a second run against an existing parquet adds nothing
        before = len(pd.read_parquet(tmp / "a.parquet"))
        run_grid(emb, tmp / "a.parquet")
        assert len(pd.read_parquet(tmp / "a.parquet")) == before, "resume re-ran cells"

        lin = a[a["head"] == "linear"]
        for ds in ("paws", "nli"):
            m = lin[lin.dataset == ds].groupby("condition")["test_acc"].mean()
            assert m["C3"] > m["C0"], f"{ds}: C3 ({m['C3']:.3f}) !> C0 ({m['C0']:.3f})"

        print("smoke OK - 20 rows, deterministic (drift 0.0), resume skips, "
              "C3 > C0 on planted signal for both datasets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
