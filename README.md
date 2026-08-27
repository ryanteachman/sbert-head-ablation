# SBERT Classifier-Head Ablation (`|u−v|` vs `u*v`)

A controlled ablation isolating whether an SBERT-style classifier head fed
`[u, v, |u−v|, u*v]` — carrying **both** the element-wise product `u*v` and the
absolute difference `|u−v|` — beats feeding just one of the two interaction
terms, using a **frozen** `all-mpnet-base-v2` backbone and direct evaluation on
NLI / QQP / PAWS.

**Read [`PROTOCOL.md`](PROTOCOL.md) first** — it is the frozen pre-registration
and the single source of truth for every design decision. `config/experiment.yaml`
encodes its constants for the code.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu124  # match cluster CUDA
pip install -r requirements.txt
pip freeze > requirements.lock.txt      # lock the resolved versions
```

## Layout

| Path | Purpose |
|---|---|
| `PROTOCOL.md` | frozen experimental protocol (§1–§20) |
| `config/experiment.yaml` | frozen constants used by the code |
| `src/data.py` | download datasets, clean labels, carve QQP val split (Phase 2) |
| `src/embed.py` | load frozen backbone, dedup + encode + cache (Phase 3) |
| `src/features.py` | build C0–C9 feature vectors, standardization, random projection |
| `src/heads.py` | linear + fixed MLP heads |
| `src/train.py` | single-run training loop + early stopping |
| `src/run_grid.py` | orchestrate the 600-run grid → `results/runs.parquet` |
| `src/analyze.py` | primary family + corrections + effect sizes; factorial; figures |
| `data/`, `embeddings/`, `results/` | regenerated artifacts (gitignored) |
| `ECE684_Paraphrase/` | prior BetBank code, reference only (gitignored) |

## Execution phases (see PROTOCOL.md §17)

1. Repo + env + scaffold  ← **done**
2. `data.py` — datasets, splits, label cleaning, base rates
3. `embed.py` — verify L2-norm + determinism, cache embeddings
4. Smoke test — one cell (C1, NLI, linear, seed 0) end to end
5. Pilot — 3 seeds, linear head, full 10×3 grid (90 runs), sanity checks only
6. Full grid — 10 seeds, both heads (600 runs)
7. `analyze.py` — stats, tables, figures
8. `paper/` — draft
