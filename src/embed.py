"""Phase 3 — encode every dataset sentence once with the frozen backbone.

For each processed split, collects the unique sentences across ``text_a`` and
``text_b``, encodes them with ``all-mpnet-base-v2`` (frozen, eval mode), and
caches to ``<embed_dir>/<dataset>/<split>.npz``:

    uniq_emb : float32 [n_unique, 768]   unique-sentence embeddings
    idx_a    : int32   [n_pairs]         row of uniq_emb for text_a
    idx_b    : int32   [n_pairs]         row of uniq_emb for text_b
    label    : int64   [n_pairs]

``<embed_dir>/meta.json`` records the model + resolved commit, dims, library
versions, per-split counts, and the L2-norm check.

This is the only GPU-heavy step. Run on Colab:  python src/embed.py
Locally, just sanity-check:                     python src/embed.py --verify
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Reduce CUDA fragmentation on small GPUs (must be set before torch inits CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import yaml

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "experiment.yaml"
NORM_TOL = 1e-3


def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _processed_dir(cfg: dict, dataset: str) -> Path:
    return ROOT / cfg["paths"]["data_dir"] / "processed" / dataset


def _embed_dir(cfg: dict, override: str | None) -> Path:
    d = Path(override) if override else ROOT / cfg["paths"]["embed_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_commit(model_name: str) -> str | None:
    """Best-effort HEAD commit of the model repo. No large downloads."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_name).sha
    except Exception:
        pass
    try:
        from huggingface_hub import constants

        ref = (Path(constants.HF_HUB_CACHE)
               / f"models--{model_name.replace('/', '--')}" / "refs" / "main")
        if ref.is_file():
            return ref.read_text().strip()
    except Exception:
        pass
    return None


def _load_model(cfg: dict, device: str):
    from sentence_transformers import SentenceTransformer

    name = cfg["backbone"]["model"]
    torch.manual_seed(0)
    model = SentenceTransformer(name, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if cfg["backbone"].get("max_seq_length"):
        model.max_seq_length = cfg["backbone"]["max_seq_length"]
    modules = [type(m).__name__ for m in model]
    return model, name, modules


def _encode(model, sentences: list[str], batch_size: int) -> np.ndarray:
    """Always returns float32 (fp16 casting happens after the norm check)."""
    emb = model.encode(
        sentences,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,   # the model's Normalize module handles it
    )
    return np.ascontiguousarray(emb, dtype=np.float32)


def _norm_report(emb: np.ndarray) -> tuple[float, np.ndarray]:
    norms = np.linalg.norm(emb, axis=1)
    return float(np.abs(norms - 1.0).max()), norms


# --------------------------------------------------------------------------- verify
def verify(cfg: dict, device: str) -> int:
    print(f"device: {device}")
    model, name, modules = _load_model(cfg, device)
    print(f"model : {name}")
    print(f"commit: {_resolve_commit(name)}")
    print(f"module stack: {modules}")
    has_norm = any("Normalize" in m for m in modules)
    print(f"Normalize module present: {has_norm}")

    probe = [
        "A man is playing a guitar.", "A person plays an instrument.",
        "The cat sat on the mat.", "Stocks rallied on Friday.",
        "Stocks fell sharply on Friday.", "",
        "Two dogs run across a field.", "Nobody is outside.",
    ]
    e1 = _encode(model, probe, 8)
    e2 = _encode(model, probe, 8)
    identical = np.array_equal(e1, e2)
    max_norm_dev, _ = _norm_report(e1)
    print(f"determinism (bit-identical re-encode): {identical}")
    print(f"max |‖emb‖ - 1|: {max_norm_dev:.2e}  ({'unit-norm OK' if max_norm_dev < NORM_TOL else 'NOT unit-norm'})")
    print(f"embedding dim: {e1.shape[1]}")

    # cache round-trip
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "rt.npz"
        np.savez(tmp, uniq_emb=e1, idx_a=np.array([0, 2], np.int32),
                 idx_b=np.array([1, 3], np.int32), label=np.array([1, 0], np.int64))
        with np.load(tmp) as z:
            ok_rt = np.array_equal(z["uniq_emb"][z["idx_a"]], e1[[0, 2]])
    print(f"cache round-trip: {ok_rt}")

    ok = identical and max_norm_dev < NORM_TOL and ok_rt and e1.shape[1] == cfg["backbone"]["embedding_dim"]
    print("\nVERIFY PASSED" if ok else "\nVERIFY FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------- run
_ENCODE_CHUNK = 100_000          # sentences per encode+cast pass (caps peak RAM)


def _chunk_span(ci: int, n: int) -> tuple[int, int]:
    return ci * _ENCODE_CHUNK, min((ci + 1) * _ENCODE_CHUNK, n)


def encode_unique(model, sentences: list[str], batch_size: int, dtype: str,
                  work_dir: Path | None = None):
    """Encode in 100k-sentence chunks; cast each to the storage dtype at once so
    the full fp32 matrix (~3.5 GB for NLI) is never held.

    With ``work_dir``, each finished chunk is written there as ``chunk_<i>.npy``
    plus a ``state.json`` — so an OOM / disconnect resumes from the last chunk
    instead of restarting the whole split.
    """
    n = len(sentences)
    n_chunks = math.ceil(n / _ENCODE_CHUNK)
    store = np.float16 if dtype == "float16" else np.float32
    cuda = torch.cuda.is_available()

    ck = Path(work_dir) if work_dir else None
    st_path = ck / "state.json" if ck else None
    st = json.loads(st_path.read_text()) if st_path and st_path.exists() else {}
    done = set(st.get("done", []))
    max_dev = float(st.get("max_dev", 0.0))
    explicit = bool(st.get("explicit", False))
    dim = st.get("dim")
    if ck:
        ck.mkdir(parents=True, exist_ok=True)

    out = None                                    # in-memory accumulator (ck is None)
    for ci in range(n_chunks):
        s, e = _chunk_span(ci, n)
        if ci in done and ck and (ck / f"chunk_{ci}.npy").exists():
            if n_chunks > 1:
                print(f"    chunk {ci + 1}/{n_chunks}  (cached)", flush=True)
            continue
        if n_chunks > 1:
            print(f"    chunk {ci + 1}/{n_chunks}  ({e - s:,} sentences)", flush=True)
        block = model.encode(sentences[s:e], batch_size=batch_size, convert_to_numpy=True,
                             show_progress_bar=True, normalize_embeddings=False)
        block = np.asarray(block, dtype=np.float32)
        norms = np.linalg.norm(block, axis=1)
        dev = float(np.abs(norms - 1.0).max())
        max_dev = max(max_dev, dev)
        if dev >= NORM_TOL:                       # PROTOCOL section 6 fallback
            explicit = True
            block = block / norms[:, None]
        block = block.astype(store)
        dim = block.shape[1]
        if ck:
            np.save(ck / f"chunk_{ci}.npy", block)
            done.add(ci)
            st_path.write_text(json.dumps({"done": sorted(done), "max_dev": max_dev,
                                           "explicit": explicit, "dim": dim,
                                           "n_chunks": n_chunks}))
        else:
            if out is None:
                out = np.empty((n, dim), dtype=store)
            out[s:e] = block
        del block
        if cuda:
            torch.cuda.empty_cache()

    if ck:                                        # assemble from chunk files
        out = np.empty((n, dim), dtype=store)
        for ci in range(n_chunks):
            s, e = _chunk_span(ci, n)
            out[s:e] = np.load(ck / f"chunk_{ci}.npy")
    return out, max_dev, explicit


def _pair_indices(df: pd.DataFrame):
    sents = pd.unique(pd.concat([df["text_a"], df["text_b"]], ignore_index=True))
    pos = pd.Series(np.arange(len(sents), dtype=np.int32), index=sents)
    idx_a = pos.reindex(df["text_a"]).to_numpy(np.int32)
    idx_b = pos.reindex(df["text_b"]).to_numpy(np.int32)
    return list(sents), idx_a, idx_b


def _cache_ok(path: Path, expected_pairs: int) -> bool:
    """A cached .npz counts as done only if it loads and matches the parquet."""
    try:
        with np.load(path) as z:
            return (set(z.files) >= {"uniq_emb", "idx_a", "idx_b", "label"}
                    and len(z["label"]) == expected_pairs
                    and len(z["idx_a"]) == expected_pairs
                    and int(z["idx_a"].max()) < z["uniq_emb"].shape[0])
    except Exception:
        return False


def _save_npz(out: Path, **arrays) -> None:
    """Atomic: write to <stem>.tmp.npz, then rename. A kill never leaves a file
    that looks complete. (tmp keeps the .npz suffix so np.savez doesn't append.)"""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, out)


def run(cfg: dict, device: str, datasets: list[str], only_splits: set[str] | None,
        embed_dir: Path, force: bool, dtype: str) -> int:
    model, name, modules = _load_model(cfg, device)
    commit = _resolve_commit(name)
    print(f"device: {device} | model: {name} | commit: {commit} | out: {embed_dir}")

    for stale in embed_dir.rglob("*.tmp.npz"):
        print(f"removing stale {stale.name}")
        stale.unlink()

    meta_path = embed_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "model": name, "commit": commit, "module_stack": modules,
        "embedding_dim": cfg["backbone"]["embedding_dim"],
        "encode_batch_size": cfg["backbone"]["encode_batch_size"],
        "storage_dtype": dtype,
        "torch": torch.__version__,
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    meta.setdefault("splits", {})
    batch_size = cfg["backbone"]["encode_batch_size"]

    for dataset in datasets:
        pdir = _processed_dir(cfg, dataset)
        files = sorted(f for f in pdir.glob("*.parquet") if f.stem != "split_assignment")
        for f in files:
            split = f.stem
            if only_splits and split not in only_splits:
                continue
            key = f"{dataset}/{split}"
            out = embed_dir / dataset / f"{split}.npz"
            df = pd.read_parquet(f)

            if out.exists() and not force and _cache_ok(out, len(df)):
                print(f"skip   {key}  (cached, {len(df):,} pairs)")
                continue
            if out.exists():
                print(f"redo   {key}  (missing/invalid cache)")

            t0 = time.time()
            sents, idx_a, idx_b = _pair_indices(df)
            print(f"encode {key}  ({len(df):,} pairs, {len(sents):,} unique)", flush=True)
            work = embed_dir / dataset / f"_wip_{split}"
            emb, dev, expl = encode_unique(model, sents, batch_size, dtype, work_dir=work)
            print(f"  encoded in {time.time() - t0:.0f}s | norm dev {dev:.1e}"
                  + ("  [explicit-normalized]" if expl else ""), flush=True)

            t1 = time.time()
            print(f"  saving {out.name} ({emb.nbytes / 1e6:.0f} MB)...", flush=True)
            _save_npz(out, uniq_emb=emb, idx_a=idx_a, idx_b=idx_b,
                      label=df["label"].to_numpy(np.int64))
            shutil.rmtree(work, ignore_errors=True)
            meta["splits"][key] = {
                "n_pairs": int(len(df)), "n_unique": int(emb.shape[0]),
                "max_norm_dev": dev, "explicit_normalization": bool(expl),
                "bytes": int(out.stat().st_size),
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            print(f"  saved in {time.time() - t1:.0f}s  "
                  f"({emb.shape[0]:,} unique, {out.stat().st_size / 1e6:.0f} MB)", flush=True)

    print("\nmeta.json:")
    print(meta_path.read_text())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="sanity check only, no real encoding")
    ap.add_argument("--datasets", default="nli,qqp,paws", help="comma list: nli,qqp,paws")
    ap.add_argument("--splits", default=None, help="comma list to restrict, e.g. val,test")
    ap.add_argument("--embed-dir", default=None, help="override output dir (e.g. a Drive path)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                    help="cache storage dtype (encoding + norm check are always fp32)")
    ap.add_argument("--encode-batch", type=int, default=None,
                    help="override encode_batch_size (lower it if the GPU OOMs)")
    ap.add_argument("--force", action="store_true", help="re-encode splits even if cached")
    args = ap.parse_args()

    cfg = _cfg()
    device = _device(args.device)
    if args.encode_batch:
        cfg["backbone"]["encode_batch_size"] = args.encode_batch

    if args.verify:
        return verify(cfg, device)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    only = {s.strip() for s in args.splits.split(",")} if args.splits else None
    return run(cfg, device, datasets, only, _embed_dir(cfg, args.embed_dir),
               args.force, args.dtype)


if __name__ == "__main__":
    sys.exit(main())
