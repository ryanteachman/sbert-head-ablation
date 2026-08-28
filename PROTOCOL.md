# Experimental Protocol — SBERT Classifier-Head Ablation (`|u−v|` vs `u*v`)

**Status:** APPROVED / frozen (2026-08-27). All §18 design decisions are
resolved. The values here are fixed and are not changed after any results are
seen. Deviations get logged in §20 with date and reason.

**Last updated:** 2026-08-27

---

## 1. Research question

In an SBERT-style classifier head, does an input of `[u, v, |u−v|, u*v]` —
carrying **both** the element-wise product `u*v` and the absolute difference
`|u−v|` — improve downstream classification accuracy over carrying just one of
them (`[u, v, |u−v|]` or `[u, v, u*v]`)?

Secondary: is any observed gain attributable to the *information* in `u*v` or
merely to the *added head capacity* of a wider feature vector? And is the useful
part of `u*v` its *sign* (directional/polarity agreement) or only its magnitude?

## 2. Contribution statement

This project does **not** claim the difference/product comparison is novel — a
version appears in Reimers & Gurevych (2019, Table 6), and the underlying
"heuristic matching" `[a, b, |a−b|, a⊙b]` predates SBERT (Mou et al. 2016; Nie &
Bansal 2017; InferSent; USE). The contribution is a **more direct isolation** of
the same question:

1. **Frozen backbone** — `u` and `v` are byte-identical across all conditions, so
   any accuracy difference is attributable to the head's input representation,
   not to backbone compensation.
2. **Direct downstream accuracy** on NLI/QQP/PAWS rather than an STS/cosine proxy.
3. **Parameter-count-matched random-feature controls** (one per direction), so a
   win for the both-terms head can be attributed to information rather than
   capacity.
4. **Paired significance testing across seeds** with effect sizes, rather than
   seed-averaging alone.

The SBERT paper's Table 6 finding (adding `u*v` to `(u, v, |u−v|)` *slightly
hurts*: 80.78 → 80.44) is treated as the leading prior expectation to be
confirmed, complicated, or contradicted.

## 3. Prior results for reference (Reimers & Gurevych 2019, Table 6)

MEAN pooling, trained on NLI, evaluated by cosine similarity on STSb dev
(a *different* measurement from this study — reproduced here only as a reference
column):

| Config | Score |
|---|---|
| (u, v) | 66.04 |
| (\|u−v\|) | 69.78 |
| (u\*v) | 70.54 |
| (\|u−v\|, u\*v) | 78.37 |
| (u, v, u\*v) | 77.44 |
| (u, v, \|u−v\|) | 80.78 |
| (u, v, \|u−v\|, u\*v) | 80.44 |

---

## 4. Conditions

`u`, `v` are the (L2-normalized) sentence embeddings. Feature blocks:
`diff = |u − v|`, `prod = u * v`, `absprod = |u * v|`,
`rand = randproj(concat(u, v))` (see §7).

| ID | Feature vector | Input dim¹ | Role |
|----|----------------|-----------|------|
| **C0** | `[u, v]` | 1536 | baseline; factorial cell (no interaction, +uv) |
| **C1** | `[u, v, \|u−v\|]` | 2304 | **PRIMARY** — difference term only |
| **C2** | `[u, v, u*v]` | 2304 | **PRIMARY** — product term only |
| **C3** | `[u, v, \|u−v\|, u*v]` | 3072 | **PRIMARY** — both interaction terms |
| **C4** | `[u, v, \|u−v\|, rand]` | 3072 | capacity-matched random control for C3 |
| **C5** | `[\|u−v\|, u*v]` | 1536 | BetBank's actual head; factorial cell |
| **C6** | `[\|u−v\|]` | 768 | factorial cell; SBERT Table 6 row |
| **C7** | `[u*v]` | 768 | factorial cell; SBERT Table 6 row |
| **C8** | `[u, v, \|u−v\|, \|u*v\|]` | 3072 | sign-ablation of C3 |
| **C9** | `[u, v, u*v, rand]` | 3072 | capacity-matched random control for C3 vs C2 |

¹ mpnet embedding dim = 768.

**Naming.** C3 (`[u, v, \|u−v\|, u*v]`) carries **both interaction terms** — the
absolute difference `|u−v|` and the element-wise product `u*v`. C1 and C2 are its
single-term ablations (difference-only, product-only). We refer to C3 as the
**both-terms head**, not "the concatenated head": every condition here
concatenates several blocks, so that label would not distinguish C3.

**Grid:** 10 conditions × 3 datasets × 2 head types × 10 seeds = **600 runs**.
All runs are small heads trained on cached frozen embeddings.

### 4.1 The embedded 2×3 factorial

C1/C2/C3 (with `u, v`) and C6/C7/C5 (without) form a full factorial:

| | diff | product | both |
|---|---|---|---|
| **with `u,v`** | C1 | C2 | C3 |
| **without `u,v`** | C6 | C7 | C5 |

This lets us estimate two main effects (does the interaction feature help? does
keeping raw `u,v` help?) and their interaction. C0 (`[u,v]`, no interaction) and
the random-feature controls C4 / C9 sit outside the factorial.

---

## 5. Datasets

Each dataset is run independently with the same 10-condition comparison. Standard
published splits only — no custom splits, no cross-validation.

### 5.1 NLI (3-way: entailment / neutral / contradiction)

- **Train:** SNLI train + MultiNLI train, pooled. Drop pairs with
  `gold_label == "-"`. Expected ≈ 550,152 + 392,702 minus "-" ≈ **~942k**.
- **Early-stopping / model-selection split:** SNLI dev, drop "-" (≈ 9,842).
- **Held-out test:** SNLI test, drop "-" (≈ 9,824). Reported as the headline NLI
  number.
- **Secondary test:** MultiNLI dev-matched (≈ 9,815) and dev-mismatched
  (≈ 9,832), reported but not in the primary statistical family (MultiNLI test
  labels are not public).
- **Label map (fixed):** entailment→0, neutral→1, contradiction→2.
- `u` = premise, `v` = hypothesis (order preserved — the task is asymmetric).
- **Sources:** `stanfordnlp/snli`, `nyu-mll/multi_nli` (HuggingFace).

### 5.2 QQP (binary: duplicate / not-duplicate)

- GLUE QQP test labels are hidden → **GLUE QQP dev (≈ 40,430) is the effective
  test set.**
- **Validation split:** a single fixed 5% stratified holdout carved from GLUE QQP
  train (seed-independent — the same split for all 10 seeds). Expected ≈ 18k val,
  ≈ 345k train.
- **Label map (fixed):** not_duplicate→0, duplicate→1. Positive base rate ≈ 36.8%.
- `u`, `v` = question1, question2 (order preserved as given).
- **Primary metric addition:** binary F1 on the positive class, reported
  alongside accuracy (class-imbalanced).
- **Known caveat (log in paper):** GLUE QQP has documented train/dev
  near-duplicate leakage. Affects all conditions equally.
- **Source:** `nyu-mll/glue`, config `qqp`.

### 5.3 PAWS-Wiki (binary: paraphrase / not-paraphrase)

- **PAWS-Wiki, `labeled_final`** only. Standard splits: train 49,401 / dev 8,000 /
  test 8,000. Positive base rate ≈ 44%.
- Dev for model selection, test for the headline number.
- `u`, `v` = sentence1, sentence2 (order preserved).
- **Source:** `google-research-datasets/paws`, config `labeled_final`.

### 5.4 Notes common to all datasets

- Report class base rates for every split.
- QQP/PAWS are symmetric tasks with arbitrary pair order; we keep the given order
  and do not add order-swap augmentation (would be a design change). `|u−v|` and
  `u*v` are order-invariant regardless; only the `[u, v]` block sees order.
- All text is embedded with the backbone's default 384-word-piece truncation
  (§6). Uniform across conditions.

---

## 6. Backbone and embeddings

- **Model:** `sentence-transformers/all-mpnet-base-v2`, **fully frozen**
  (`eval()` mode, `torch.no_grad()`, no parameter updates).
- **Pooling / normalization:** the published stack is Transformer → mean pooling →
  **L2 Normalize** (confirmed in the model's `modules.json`), so `.encode()`
  returns unit-norm vectors. **Phase 1 verification:** confirm output vectors are
  unit-norm and record the module stack + model revision hash. **If** a future
  revision drops the Normalize module, apply L2 normalization explicitly — unit-
  norm `u`, `v` are part of the fixed protocol (the scale of `|u−v|` and `u*v`
  depends on it).
- **Precision:** encoding and the unit-norm check run in fp32. **Encode batch
  size:** 512 (throughput only — mean-pool + normalize + the transformer forward
  are per-sentence, so the cached embedding of a sentence does not depend on
  batch size). Max sequence length: model default (384). Encoding is done in
  sentence chunks of 100k so the full fp32 matrix is never materialized.
- **Determinism:** fixed RNG seeds; `torch.use_deterministic_algorithms(True)`
  where feasible. Encoding in `eval()` mode is deterministic regardless.
  Verification: encode one batch twice, assert bit-identical. (Confirmed
  2026-08-27: module stack `[Transformer, Pooling, Normalize]`, re-encode
  bit-identical, max ‖emb‖−1 ≈ 1.2e-7.)
- **Dedup:** encode the set of *unique* sentences per split, then gather into
  `(u, v)` pairs. (Dedup is modest in practice — NLI train: 942k pairs →
  1.15M unique sentences; QQP train: 346k → 474k.)
- **Cache:** one `.npz` per dataset split holding `uniq_emb` (unique-sentence
  embeddings), `idx_a` / `idx_b` (per-pair row indices), and `label`; plus
  `meta.json` recording model name + resolved commit + module stack + library
  versions + per-split counts + the norm check. Embeddings are computed **once**
  and reused by every run.
- **Cache dtype: fp16.** Stored embeddings are cast to float16 (fp32 total ≈
  5.6 GB → fp16 ≈ 2.8 GB). The cast introduces ~1e-3 relative error applied
  *identically to every condition*, so it cannot bias any C-vs-C contrast;
  block standardization and the `|u−v|` / `u*v` features are recomputed in fp32
  from the fp16 cache. `--dtype float32` is available if a robustness check is
  wanted.

---

## 7. Feature construction

Given cached `u`, `v` (already L2-normalized):

- `diff = |u − v|`
- `prod = u * v`
- `absprod = |u * v|`  (C8 only)
- `rand` (C4 and C9): `r = concat(u, v) @ W_r`, where `W_r ∈ ℝ^{1536×768}` has iid
  `N(0, 1/1536)` entries. **`W_r` is redrawn per seed** (tied to the run seed) so
  the control averages over random projections rather than depending on one draw.
  C4 and C9 use the *same* `W_r` for a given seed.
- **Block-wise standardization:** for every block used by a condition (including
  the `u` and `v` blocks), compute per-dimension mean and std on that run's
  **training split** and apply to train/val/test. Standardization stats are
  recomputed per dataset; they do not depend on the seed except for the `rand`
  block.
- The condition's feature vector is the concatenation of its blocks in the order
  listed in §4.

---

## 8. Classifier heads

Both heads output `C` logits (`C = 3` for NLI, `2` for QQP/PAWS) and are trained
with softmax cross-entropy. `bias=True`. Default PyTorch initialization (seeded).

### 8.1 Linear probe — PRIMARY

- `nn.Linear(d_in, C)`. No hidden layer, no dropout, no weight decay.
- This is the instrument for the headline result: it reproduces the SBERT paper's
  classification objective and is the most sensitive probe of what information the
  feature vector carries (a high-capacity head can relearn interaction terms and
  wash out the effect).
- **Precedent:** SBERT's own classification objective is a single softmax layer on
  `[u, v, |u−v|]` (Reimers & Gurevych 2019); SentEval's default transfer-task
  classifier is logistic regression (`nhid = 0`; Conneau & Kiela 2018); Hewitt &
  Liang (2019) show linear probes are substantially more *selective* (memorize
  less) than MLP probes and are the more faithful readout instrument.

### 8.2 Fixed MLP — SECONDARY (robustness only)

- `Linear(d_in, 256) → ReLU → Dropout(0.1) → Linear(256, C)`.
- Hidden width **256, fixed regardless of `d_in`**, so the MLP does not itself
  become a capacity confound across conditions.
- Weight decay `1e-4`.
- **One config, chosen a priori, never swept or tuned.** Reported as an appendix
  robustness check ("does the effect survive a head that can learn its own
  interactions?"), never as the headline.
- **Precedent for the shape:** a single hidden layer is the universal choice for
  MLP heads/probes on sentence embeddings — InferSent uses 1 hidden layer of 512
  for NLI (Conneau et al. 2017); Conneau et al. (2018) and SentEval use 1 hidden
  layer with width tuned in {50, 100, 200}; Eger et al. (2020) sweep hidden ∈
  {50, 100, 200}, dropout ∈ {0, 0.1, 0.2}, L2 ∈ {1e-5 … 1e-1}. Width 256 sits
  just above that probing range and matches BetBank's head, while staying below
  InferSent's 512 — a deliberately modest capacity check. ReLU replaces the
  historical sigmoid/tanh (modern default; does not affect the robustness
  conclusion). Fixing rather than sweeping the config follows Eger et al. (2020)
  and Hewitt & Liang (2019), who both show probe hyperparameters materially move
  results.
- **Relation to the random controls (C4, C9):** each is the feature-space
  analogue of Hewitt & Liang's (2019) *control task* — an added block that can
  only help a probe via capacity/memorization, not via real signal. C3 > C4 is
  the selectivity check for `u*v` (vs C1); C3 > C9 is the selectivity check for
  `|u−v|` (vs C2).

---

## 9. Training

Fixed constants, identical for every cell:

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β = 0.9, 0.999; ε = 1e-8) |
| Learning rate | 1e-3, **constant** (no schedule) |
| Weight decay | 0.0 (linear) / 1e-4 (MLP) |
| Batch size | 256 |
| Max epochs | 30 (raise if any cell early-stops at the ceiling in the pilot) |
| Gradient clipping | 1.0 (global norm) |
| Loss | CrossEntropyLoss, no class weighting |
| Precision | fp32 |

- **Learning-rate schedule — constant 1e-3, decided.** Rationale: the linear-probe
  objective is convex, no pretrained weights are at risk (frozen backbone), and
  best-checkpoint restoration handles convergence. A schedule would add two
  hyperparameters (warmup length, decay horizon) — the latter awkward under early
  stopping. **Contingency:** if the pilot (Phase 5) shows noisy loss/accuracy
  curves or per-seed metric SD > ~0.3 pp, switch to a linear decay to 0 over a
  fixed step budget — decided from pilot diagnostics only, logged in §20.
- **Early stopping / checkpoint selection:** evaluate on the validation split
  every half-epoch; track validation accuracy (tie-break: lower validation loss);
  stop after `patience = 5` consecutive evaluations without improvement; restore
  the best checkpoint. Test metrics are computed **only** on that restored
  checkpoint.
- **No class weighting** even on imbalanced QQP — imbalance is handled at
  reporting time via F1; consistency across conditions matters more.

---

## 10. Seeds

- **10 seeds:** `{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}`.
- Each seed controls: head parameter initialization, training-data shuffle order,
  and the `W_r` draw for C4 / C9.
- Each seed does **not** control: dataset splits (fixed), the QQP validation
  carve-out (fixed), embedding computation (deterministic).
- Pairing: seed `i` uses the same RNG stream across all conditions, so per-seed
  metric differences between conditions are **paired**.

---

## 11. Metrics

Per (dataset × condition × head × seed) run, on the held-out test split:

- **Accuracy** — primary for all datasets.
- **Macro-F1** — NLI.
- **Binary F1 (positive class)** — primary secondary metric for QQP; reported for
  PAWS too.
- Per-class precision / recall — retained for error analysis.
- Best-checkpoint step, epochs trained, wall-clock time, config hash — logged.

Aggregation: mean ± sample std (ddof = 1) across the 10 seeds, plus a t-based 95%
CI. A **test-set bootstrap** CI (resample test examples within a run, 10,000
draws) is reported alongside to reflect test-set sampling noise that seed
variance alone misses. (Distinct from the **seed-level bootstrap** of paired
differences in §12.1, which resamples the 10 seeds.)

---

## 12. Statistical analysis plan

**Unit of analysis:** the per-seed test metric. Comparisons are paired on seed.

Every reported comparison carries all three layers, following standard NLP
practice (Reimers & Gurevych 2017; Dror et al. 2018):

1. **Descriptive:** per-condition mean ± sample std across the 10 seeds, a t-based
   95% CI, and the full per-seed distribution (table + strip plot). A single
   score is not reported alone — seed choice alone can produce large, significant
   swings (Reimers & Gurevych 2017).
2. **Effect estimate:** the paired mean difference between conditions, in
   accuracy points, with its 95% CI. This *is* the "comparison of averages" and
   is the quantity of primary interest.
3. **Inference:** the significance tests below — do the differences in (2) exceed
   seed noise.

Plus a standardized effect size (Cohen's `d_z`) on every primary contrast, and
an optional one-number cross-dataset summary (mean effect across the 3 datasets,
caveated as thin at n = 3 datasets).

### 12.1 Primary family (pre-registered, Holm-Bonferroni corrected, α = 0.05 FWER)

On the **linear head**, **accuracy**, for each dataset ∈ {NLI, QQP, PAWS}:

| Hypothesis | Contrast | Question |
|---|---|---|
| H1 | C3 vs C1 | does adding `u*v` on top of `[u, v, \|u−v\|]` help? |
| H1c | C3 vs C4 | …is that gain information rather than added capacity? |
| H2 | C3 vs C2 | does adding `\|u−v\|` on top of `[u, v, u*v]` help? |
| H2c | C3 vs C9 | …is that gain information rather than added capacity? |

→ 4 contrasts × 3 datasets = **12 hypotheses** in the primary family.

- **Tests, both reported:** paired two-sided t-test **and** Wilcoxon signed-rank.
  The t-test is more powerful and yields the effect CI in (2); Wilcoxon is the
  distribution-free cross-check but is discreteness-limited at n = 10 (two-sided
  floor p ≈ 0.002; one flipped seed → p ≈ 0.02). A seed-level bootstrap of the
  paired difference (10,000 resamples) is reported as a third, assumption-light
  CI.
- **Decision rule (pre-registered):** a primary hypothesis is called
  **confirmed** only if (a) the Holm-corrected paired t-test has p < 0.05 **and**
  (b) Wilcoxon agrees in sign with uncorrected p < 0.05. Disagreement is reported
  as "mixed / suggestive," not confirmed. Stricter than either test alone;
  guards against the t-test's outlier sensitivity and Wilcoxon's discreteness.
- **Effect size (always reported):** Cohen's `d_z = mean(diff) / sd(diff)`, plus
  the raw mean accuracy-point difference with 95% CI.
- **Interpretation rule:** an improvement is only read as evidence for the
  *information* value of the added block if the paired capacity control also
  holds — i.e. H1 counts only with H1c (C3 > C4), and H2 counts only with H2c
  (C3 > C9). C4 and C9 are parameter-count-matched to C3 (all 3072-dim), so each
  control isolates information from head capacity for its contrast.
- **Symmetry:** H1 (does `u*v` add?) is the paper's thesis; H2 (does `|u−v|`
  add?) is the mirror image and a well-established expectation — reported for
  completeness and because C9 makes it cheap to control properly.

### 12.2 Power / minimum detectable effect

With n = 10, paired t, 80% power: detectable `d_z ≈ 1.0` at an uncorrected
two-sided α = 0.05, rising to `d_z ≈ 1.2` at the Holm worst-case per-test
α ≈ 0.05/12 ≈ 0.004. In both cases the mean difference must be roughly 1–1.2 SD
of the per-seed differences. Given per-seed metric SD with frozen embeddings is
typically ~0.1–0.3 pp, the MDE in accuracy points is roughly **0.1–0.4 pp**. A
null result is reported as "no detectable effect above ~X pp," never as "no
effect." The exact per-contrast MDE is recomputed from the observed SDs after
the full run.

### 12.3 Secondary / exploratory analyses (labeled as such, not corrected)

- MLP-head replication of the four primary contrasts (H1 / H1c / H2 / H2c).
- **2×3 factorial** (§4.1): two-way repeated-measures ANOVA (seed as the repeated
  unit; Greenhouse–Geisser corrected) per dataset on accuracy; report main
  effects, interaction, and partial η². Cell-means table is the primary artifact;
  the ANOVA is supplementary.
- C1 vs C2 (difference vs product head-to-head — the SBERT Table 6 question).
- **C3 vs C8** (sign ablation): if the C3 benefit survives with `|u*v|`, the
  useful part is magnitude; if it collapses toward C1, the useful part is sign /
  polarity.
- C5 vs C6, C5 vs C7 (BetBank head retrospective).
- Full row-by-row mapping of the Table 6 conditions (C0–C3, C5–C7) onto SBERT
  Table 6.
- QQP F1 versions of all contrasts.
- MultiNLI dev-matched / dev-mismatched.

---

## 13. What we will and will not claim

### 13.1 Supported (given significant, sign-consistent results)

- **If C3 > C1 and C3 > C4:** with a frozen `all-mpnet-base-v2` encoder and a
  linear classifier, adding `u*v` alongside `|u−v|` (the both-terms head) gives a
  small but reliable downstream-accuracy gain over `|u−v|` alone that exceeds a
  parameter-matched random-feature control — i.e., it reflects complementary
  information, not head capacity.
- **If C3 ≈ C1 or C3 ≯ C4:** no reliable benefit from `u*v` once `|u−v|` is
  present in this regime; confirms/sharpens SBERT Table 6.
- Dataset-level pattern description (with only 3 datasets: descriptive, not a
  generalization).
- Head-capacity robustness (linear vs MLP agree → not a capacity artifact;
  disagree → effect is capacity-dependent, itself a finding).
- The `u,v`-inclusion main effect from the factorial.
- Whether `|u−v|` adds information beyond `[u, v, u*v]` (C3 vs C2, controlled by
  C9) — the mirror-image contrast; expected positive, reported for completeness.
- Whether the useful component of `u*v` is sign or magnitude (C3 vs C8).
- A suggestive-only retrospective on whether BetBank's `u*v` was load-bearing.
- Whether the SBERT Table 6 qualitative pattern reproduces under the changed
  conditions.

### 13.2 Not supported — stated explicitly as limitations

- Anything about the **end-to-end / fine-tuned** setting (separate follow-up;
  backbone compensation may not preserve the frozen result).
- Other backbones, embedding dims, pooling, or non-normalized embeddings.
- Regression / STS tasks (dropped by design).
- Other interaction operators (signed difference, bilinear, cosine features,
  learned pooling) — we test `|u−v|` and `u*v` only.
- A full mechanistic account of *why* `u*v` helps beyond the sign-vs-magnitude
  split from C8 (would need a dedicated probing study).
- Generalization beyond English sentence-pair semantic classification.
- Leaderboard-relative magnitude — frozen embeddings sit below fine-tuned SOTA;
  we measure deltas between head conditions, framed as such.
- Practical "you should always include `u*v`" — effect sizes are reported; the
  reader judges whether a small gain justifies a 33% wider head.
- Strong null claims — n = 10 is modest; report CIs and MDE.

---

## 14. Threats to validity (all affect conditions equally unless noted)

- GLUE QQP train/dev near-duplicate leakage.
- QQP validation is a random (not question-grouped) carve-out from train, so a
  question may appear on both sides — mildly optimistic validation estimate.
  Acceptable: the set is used only for early stopping / checkpoint selection
  (never reported), and any bias is symmetric across all 10 conditions.
- NLI train (SNLI+MNLI) vs test (SNLI only) domain mismatch — mitigated by also
  reporting MultiNLI dev-matched.
- 384-word-piece truncation during embedding.
- Class base rates differ across datasets (reported).
- Seed variance captures only head-init + shuffle variance; fixed splits mean CIs
  understate uncertainty over possible datasets — mitigated by the reported
  test-set bootstrap CI.
- Standardization statistics estimated on train and applied to test (standard;
  negligible leakage risk at this scale).

---

## 15. Compute and experiment tracking

- **Hardware:** single GPU (record model). Embedding precompute is the main cost
  (NLI: ~1.9M sentence encodes, dedup first). Head training is seconds–minutes
  per run. Full 600-run grid is feasible within ~1 day after embeddings are
  cached.
- **Results store:** one tidy row per run in `results/runs.parquet` — dataset,
  condition, head, seed, all metrics, best-checkpoint step, epochs trained, wall
  time, config hash, git commit.
- **Config:** a single YAML; every run logs its resolved config and the git
  commit hash. Re-runs are bit-reproducible.

---

## 16. Repository structure

```
SBERT/
  PROTOCOL.md              # this document
  requirements.txt
  config/experiment.yaml
  src/
    data.py                # download, splits, label cleaning, QQP val carve-out
    embed.py               # backbone load, dedup, encode, cache + verification
    features.py            # build C0..C9 vectors, standardization, random proj
    heads.py               # linear + fixed MLP
    train.py               # single-run training loop + early stopping
    run_grid.py            # orchestrate the 600 runs -> runs.parquet
    analyze.py             # stats, tables, figures
  data/                    # raw + processed (gitignored)
  embeddings/              # cached arrays (gitignored)
  results/                 # runs.parquet, tables/, figures/
  paper/                   # writeup
```

---

## 17. Execution phases

1. **Repo + env** — `git init`, pin deps, scaffold structure.
2. **`data.py`** — download all three datasets; verify split sizes against the
   numbers in §5; clean labels; carve and save the QQP validation split; print
   base rates. No modeling.
3. **`embed.py`** — load backbone; verify L2-normalized output and determinism;
   dedup + encode + cache; record model revision.
4. **`features.py` + `heads.py` + `train.py`** — one smoke-test cell end to end
   (C1, NLI, linear, seed 0); confirm the absolute number is sane (linear head on
   frozen mpnet NLI should land in the ~70s–80s).
5. **Pilot — 3 seeds `{0, 1, 2}`, linear head only.** Run the full 10-condition ×
   3-dataset grid (90 runs). Purpose is pipeline shakeout and sanity-checking,
   **not** statistics. Checks: (a) absolute numbers are in a believable range;
   (b) conditions are ordered plausibly (e.g. C1/C2/C3 ≳ C0; C6/C7 < their
   with-`u,v` counterparts; C4 ≈ C1 and C9 ≈ C2); (c) per-seed variance is small
   (~0.1–0.3 pp); (d) early stopping triggers before max epochs; (e) wall-clock
   per run is as expected. No significance tests are run on 3 seeds.
6. **`run_grid.py`** — full grid: all 10 seeds, both heads → `runs.parquet`.
7. **`analyze.py`** — primary family + corrections + effect sizes; factorial;
   exploratory contrasts; tables and figures.
8. **`paper/`** — draft.

---

## 18. Decisions log (all resolved)

1. **Learning-rate schedule — RESOLVED:** constant 1e-3, with a pilot-gated
   contingency to a linear decay if curves are noisy or per-seed SD > ~0.3 pp
   (see §9).
2. **NLI model-selection split — RESOLVED:** SNLI dev only. Keeps MultiNLI
   dev-matched / dev-mismatched fully held out as clean secondary test sets, and
   aligns checkpoint selection with the headline test distribution (SNLI test).
   ~9,800 examples is ample for early-stopping a small head.
3. **QQP validation fraction — RESOLVED:** 5% stratified (~18k). Already a
   large, low-variance selection set; 10% would trade training data for a
   negligible variance gain. Random (not question-grouped) split; acceptable
   because the set is selection-only and the effect is condition-symmetric
   (noted in §14).
4. **MLP config — RESOLVED:** `d_in → 256 → C`, ReLU, dropout 0.1, weight decay
   1e-4, same optimizer / LR / early-stopping as the linear head. One config,
   fixed a priori. Citable basis in §8.2.
5. **Pilot — RESOLVED:** 3 seeds `{0, 1, 2}` on the linear head across the full
   10×3 condition grid (90 runs, Phase 5), before the full 10-seed / both-head
   run (Phase 6).
6. **Primary statistical test — RESOLVED:** paired t-test *and* Wilcoxon both
   reported; a hypothesis is "confirmed" only if both agree (Holm-corrected t
   p < 0.05 and Wilcoxon sign-agreeing at uncorrected p < 0.05); seed bootstrap
   as a third CI. See §12.1.

---

## 19. References

- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks.* EMNLP. — backbone, `[u, v, |u−v|]` head, Table 6
  ablation.
- Conneau, A. et al. (2017). *Supervised Learning of Universal Sentence
  Representations from Natural Language Inference Data* (InferSent). EMNLP. —
  origin of the `(u, v, |u−v|, u*v)` combination; 1-hidden-layer (512) NLI head.
- Conneau, A. & Kiela, D. (2018). *SentEval: An Evaluation Toolkit for Universal
  Sentence Representations.* LREC. — default transfer classifier = logistic
  regression; optional 1-hidden-layer MLP.
- Conneau, A. et al. (2018). *What you can cram into a single vector: Probing
  sentence embeddings for linguistic properties.* ACL. — logistic regression +
  1-sigmoid-hidden-layer MLP, width tuned.
- Eger, S., Daxenberger, J. & Gurevych, I. (2020). *How to Probe Sentence
  Embeddings in Low-Resource Languages: On Structural Design Choices for Probing
  Task Evaluation.* CoNLL. — MLP hidden {50,100,200}, dropout {0,0.1,0.2}, L2
  {1e-5…1e-1}; classifier choice materially affects probing results.
- Hewitt, J. & Liang, P. (2019). *Designing and Interpreting Probes with Control
  Tasks.* EMNLP. — linear probes more selective than MLPs; control tasks and
  selectivity (basis for the C4 random-feature control).
- Mou, L. et al. (2016). *Natural Language Inference by Tree-Based Convolution and
  Heuristic Matching.* ACL (arXiv 2015). — earliest `[a, b, a−b, a⊙b]` heuristic
  matching.
- Nie, Y. & Bansal, M. (2017). *Shortcut-Stacked Sentence Encoders for
  Multi-Domain Inference.* RepEval. — `[v_p, v_h, |v_p−v_h|, v_p⊙v_h]`.
- Cer, D. et al. (2018). *Universal Sentence Encoder.* arXiv:1803.11175. — also
  uses the `(u, v, |u−v|, u*v)`-style matching for pair tasks.
- Reimers, N. & Gurevych, I. (2017). *Reporting Score Distributions Makes a
  Difference: Performance Study of LSTM-networks for Sequence Tagging.* EMNLP. —
  report seed distributions, not single scores.
- Dror, R. et al. (2018). *The Hitchhiker's Guide to Testing Statistical
  Significance in Natural Language Processing.* ACL. — significance-test
  selection; pair descriptive results with an appropriate test.
- Bowman, S. et al. (2015). *A large annotated corpus for learning natural
  language inference* (SNLI). EMNLP.
- Williams, A. et al. (2018). *A Broad-Coverage Challenge Corpus for Sentence
  Understanding through Inference* (MultiNLI). NAACL.
- Wang, A. et al. (2019). *GLUE: A Multi-Task Benchmark and Analysis Platform for
  Natural Language Understanding.* ICLR. — QQP split conventions.
- Zhang, Y. et al. (2019). *PAWS: Paraphrase Adversaries from Word Scrambling.*
  NAACL.

---

## 20. Protocol deviations

_(none yet — log dated entries here as they occur)_
