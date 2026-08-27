"""Phase 2 — download, clean, and split the three datasets.

Writes ``data/processed/<dataset>/<split>.parquet`` with columns
``[text_a, text_b, label]`` (NLI also keeps ``source``), plus
``data/processed/qqp/split_assignment.parquet`` recording the fixed
train/validation carve-out. Prints a summary table of split sizes and class
base rates, checked against the reference numbers in PROTOCOL.md section 5.

No modeling here. Run:  python src/data.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows consoles default to cp1252; make stdout/stderr UTF-8 so summary tables
# with non-ASCII characters print cleanly.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "experiment.yaml"

# Reference split sizes from PROTOCOL.md section 5 (post-cleaning). Used only for
# a sanity check at the end; a mismatch is a warning, not a hard failure.
REFERENCE = {
    ("nli", "train"): 942_069,          # SNLI train + MNLI train, minus "-"
    ("nli", "val"): 9_842,              # SNLI validation, minus "-"
    ("nli", "test"): 9_824,             # SNLI test, minus "-"
    ("nli", "mnli_val_matched"): 9_815,
    ("nli", "mnli_val_mismatched"): 9_832,
    ("qqp", "train"): 345_653,          # GLUE QQP train, minus the 5% val carve-out
    ("qqp", "val"): 18_193,             # 5% stratified carve-out
    ("qqp", "test"): 40_430,            # GLUE QQP validation (labels public)
    ("paws", "train"): 49_401,
    ("paws", "val"): 8_000,
    ("paws", "test"): 8_000,
}

NLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _out_dir(cfg: dict, dataset: str) -> Path:
    d = ROOT / cfg["paths"]["data_dir"] / "processed" / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hf_cache(cfg: dict) -> str:
    c = ROOT / cfg["paths"]["data_dir"] / "raw" / "hf"
    c.mkdir(parents=True, exist_ok=True)
    return str(c)


def _write(df: pd.DataFrame, path: Path) -> None:
    df = df.reset_index(drop=True)
    df.to_parquet(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(df):,} rows)")


# --------------------------------------------------------------------------- NLI
def process_nli(cfg: dict, cache: str) -> dict[str, pd.DataFrame]:
    print("\n[NLI]  SNLI + MultiNLI")
    snli = load_dataset("stanfordnlp/snli", cache_dir=cache)
    mnli = load_dataset("nyu-mll/multi_nli", cache_dir=cache)

    def frame(ds, source: str) -> pd.DataFrame:
        df = ds.to_pandas()[["premise", "hypothesis", "label"]]
        df = df[df["label"].isin([0, 1, 2])].copy()          # drop "-" (label == -1)
        df = df.rename(columns={"premise": "text_a", "hypothesis": "text_b"})
        df["source"] = source
        return df

    train = pd.concat(
        [frame(snli["train"], "snli"), frame(mnli["train"], "mnli")],
        ignore_index=True,
    )
    out = {
        "train": train,
        "val": frame(snli["validation"], "snli"),
        "test": frame(snli["test"], "snli"),
        "mnli_val_matched": frame(mnli["validation_matched"], "mnli"),
        "mnli_val_mismatched": frame(mnli["validation_mismatched"], "mnli"),
    }
    d = _out_dir(cfg, "nli")
    for split, df in out.items():
        _write(df[["text_a", "text_b", "label", "source"]], d / f"{split}.parquet")
    return out


# --------------------------------------------------------------------------- QQP
def process_qqp(cfg: dict, cache: str) -> dict[str, pd.DataFrame]:
    print("\n[QQP]  GLUE / QQP")
    qqp = load_dataset("nyu-mll/glue", "qqp", cache_dir=cache)
    vspec = cfg["datasets"]["qqp"]["val_from_train"]

    full_train = qqp["train"].to_pandas()[["question1", "question2", "label"]]
    full_train = full_train.rename(columns={"question1": "text_a", "question2": "text_b"})
    full_train = full_train[full_train["label"].isin([0, 1])].reset_index(drop=True)

    tr_idx, val_idx = train_test_split(
        full_train.index.to_numpy(),
        test_size=vspec["fraction"],
        stratify=full_train["label"].to_numpy() if vspec["stratified"] else None,
        random_state=vspec["seed"],
        shuffle=True,
    )
    assignment = pd.DataFrame({"row": full_train.index})
    assignment["split"] = "train"
    assignment.loc[assignment["row"].isin(val_idx), "split"] = "val"

    test = qqp["validation"].to_pandas()[["question1", "question2", "label"]]
    test = test.rename(columns={"question1": "text_a", "question2": "text_b"})
    test = test[test["label"].isin([0, 1])]

    out = {
        "train": full_train.loc[sorted(tr_idx)],
        "val": full_train.loc[sorted(val_idx)],
        "test": test,
    }
    d = _out_dir(cfg, "qqp")
    for split, df in out.items():
        _write(df[["text_a", "text_b", "label"]], d / f"{split}.parquet")
    _write(assignment, d / "split_assignment.parquet")
    return out


# -------------------------------------------------------------------------- PAWS
def process_paws(cfg: dict, cache: str) -> dict[str, pd.DataFrame]:
    print("\n[PAWS]  PAWS-Wiki / labeled_final")
    paws = load_dataset("google-research-datasets/paws", "labeled_final", cache_dir=cache)

    def frame(ds) -> pd.DataFrame:
        df = ds.to_pandas()[["sentence1", "sentence2", "label"]]
        df = df.rename(columns={"sentence1": "text_a", "sentence2": "text_b"})
        return df[df["label"].isin([0, 1])].copy()

    out = {"train": frame(paws["train"]), "val": frame(paws["validation"]), "test": frame(paws["test"])}
    d = _out_dir(cfg, "paws")
    for split, df in out.items():
        _write(df, d / f"{split}.parquet")
    return out


# ----------------------------------------------------------------------- summary
def summarize(all_splits: dict[str, dict[str, pd.DataFrame]]) -> bool:
    print("\n" + "=" * 78)
    print(f"{'dataset':<7} {'split':<20} {'rows':>10} {'ref':>10} {'d%':>7}  class base rates")
    print("-" * 78)
    ok = True
    for dataset, splits in all_splits.items():
        for split, df in splits.items():
            n = len(df)
            ref = REFERENCE.get((dataset, split))
            if ref:
                delta = 100.0 * (n - ref) / ref
                flag = "" if abs(delta) < 1.0 else "  <-- CHECK"
                if flag:
                    ok = False
                refs = f"{ref:>10,}"
                ds = f"{delta:>+6.2f}"
            else:
                refs, ds, flag = f"{'-':>10}", f"{'-':>7}", ""
            counts = df["label"].value_counts().sort_index()
            names = NLI_LABELS if dataset == "nli" else {}
            rates = ", ".join(
                f"{names.get(k, k)}={v / n:.3f}" for k, v in counts.items()
            )
            print(f"{dataset:<7} {split:<20} {n:>10,} {refs} {ds:>7}  {rates}{flag}")
    print("=" * 78)
    print("OK — all split sizes within 1% of PROTOCOL.md section 5" if ok
          else "WARNING — some split sizes differ from the protocol reference (see above)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["nli", "qqp", "paws"], help="process a single dataset")
    args = ap.parse_args()

    cfg = _cfg()
    cache = _hf_cache(cfg)
    runners = {"nli": process_nli, "qqp": process_qqp, "paws": process_paws}
    if args.only:
        runners = {args.only: runners[args.only]}

    all_splits = {name: fn(cfg, cache) for name, fn in runners.items()}
    ok = summarize(all_splits)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
