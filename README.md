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
| `src/embed.py` | load frozen backbone, dedup + encode + cache as fp16 `.npz` (Phase 3) |
| `notebooks/embed_colab.ipynb` | run Phase 3 on Colab GPU, embeddings → Google Drive |
| `src/features.py` | build C0–C9 feature vectors, per-block standardization, random projection |
| `src/heads.py` | linear + fixed MLP heads |
| `src/train.py` | single grid cell: train head, early-stop, evaluate |
| `src/run_grid.py` | orchestrate the grid (`--pilot` = 90 cells, full = 900) → `results/*.parquet` |
| `src/analyze.py` | primary family + corrections + effect sizes; factorial; figures (Phase 7) |
| `scripts/smoke_pipeline.py` | offline plumbing check on synthetic embeddings |
| `notebooks/pilot_colab.ipynb` | Phase 4-5 smoke test + pilot on Colab |
| `data/`, `embeddings/`, `results/{figures,tables}/` | regenerated artifacts (gitignored); `results/*.parquet` IS tracked |
| `ECE684_Paraphrase/` | prior BetBank code, reference only (gitignored) |

## Execution phases (see PROTOCOL.md §17)

1. Repo + env + scaffold  ← **done**
2. `data.py` — datasets, splits, label cleaning, base rates  ← **done** (all splits match PROTOCOL §5 exactly)
3. `embed.py` — verify L2-norm + determinism, cache embeddings  ← code + local verify **done**; full encode on Colab (`notebooks/embed_colab.ipynb`)
4. Smoke test — one cell end to end  ← **done** (offline `scripts/smoke_pipeline.py`; real run in pilot notebook)
5. Pilot — 3 seeds, linear, 90 cells  ← **done** — `results/pilot_runs.parquet`; mechanics/magnitudes/ordering OK
6. Full grid — 15 seeds, both heads (900 cells)  ← code **done**, ~1-2 h on a Colab GPU; run via `notebooks/full_grid_colab.ipynb`
7. `analyze.py` — stats, tables, figures  ← **not written**; local, off `results/runs.parquet`
8. `paper/` — draft

**Compute:** phases 3-6 run on Colab (see `notebooks/`), reading/writing a Google
Drive folder; only the small `results/*.parquet` comes back to the repo. Phases
7-8 run locally.
