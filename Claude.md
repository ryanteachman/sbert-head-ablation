# Project: SBERT Classifier Head Ablation Study

## Background

This project follows on from **BetBank**, an SBERT-based (`all-mpnet-base-v2`)
paraphrase-detection system for matching equivalent sports bets across
sportsbooks. BetBank's classifier head concatenated an element-wise
multiplication vector (`u * v`) alongside the standard difference vector
(`|u - v|`) used in the original SBERT paper's classification objective. That
project reported strong results with the concatenated approach, but never ran
a controlled ablation isolating whether the multiplication vector actually
helped — all reported BetBank numbers already included both vectors combined.

This project designs and runs that missing ablation properly, using
recognized benchmark datasets, so the result is comparable to published work
and can be written up as a short paper for arXiv (and possibly a workshop
submission, e.g. ACL/EMNLP student research track).

## Research Question

Does concatenating an element-wise multiplication vector (`u * v`) with the
standard difference vector (`|u - v|`) in an SBERT-style classifier head
improve performance over either vector used alone?

## Related Work / Prior Art (checked before starting implementation)

The original SBERT paper (Reimers & Gurevych, 2019, Table 6, Section 6
"Ablation Study") already reports a version of this comparison. With MEAN
pooling, trained on NLI, and **evaluated via cosine similarity on the STSb
dev set** (not downstream NLI/QQP/PAWS accuracy), they report:

| Configuration | Score |
|---|---|
| (u, v) | 66.04 |
| (\|u−v\|) | 69.78 |
| (u\*v) | 70.54 |
| (\|u−v\|, u\*v) | 78.37 |
| (u, v, u\*v) | 77.44 |
| (u, v, \|u−v\|) | 80.78 |
| (u, v, \|u−v\|, u\*v) | 80.44 |

Notably, adding `u*v` to `(u, v, |u-v|)` — 80.78 → 80.44 — **slightly hurts**
performance in their setup, contrary to the BetBank hypothesis. They also
used 10 seeds (this project uses 5; consider raising to 10 if compute
allows, for closer comparability). Their result is trained end-to-end (not
frozen-backbone) and measured via STSb/cosine-similarity rather than direct
downstream task accuracy — so it answers a related but distinct question
from this project's.

A predecessor combination — `[v_p, v_h, |v_p−v_h|, v_p⊗v_h]` — appears
earlier in Nie & Bansal (2017, "Shortcut-Stacked Sentence Encoders," citing
Mou et al. 2015's "heuristic matching"), and is also used by InferSent and
the Universal Sentence Encoder. This is likely the actual origin of the
concat-multiplication idea and should be cited as related/prior work
regardless of this project's outcome.

**Reframed contribution statement:** this project does not claim the
difference/multiplication comparison is novel. Instead, it isolates the same
question more directly than prior reporting by: (a) freezing the backbone to
remove backbone-compensation confounds present in end-to-end training, (b)
measuring direct downstream classification accuracy on NLI/QQP/PAWS rather
than an STSb/cosine-similarity proxy, and (c) using paired significance
testing across seeds rather than raw seed-averaging. The SBERT paper's Table
6 result (multiplication slightly hurts) should be treated as the leading
prior expectation to be confirmed, complicated, or contradicted — not
ignored — in the writeup's related work and discussion sections.

## Core Design

**Backbone:** `all-mpnet-base-v2`, **fully frozen** for the primary study.
Embeddings `u` and `v` are identical across all three conditions below, so
any accuracy difference is attributable to the classifier head's input
representation, not to backbone drift. (Full end-to-end fine-tuning, matching
the original SBERT paper's procedure, is planned as a **secondary follow-up
study** — see below — not part of the core result.)

**Classifier head variants (3-way comparison), matching the SBERT paper's
input format of also including `u` and `v` themselves:**
1. `[u, v, |u - v|]` — difference only (SBERT paper baseline)
2. `[u, v, u * v]` — multiplication only
3. `[u, v, |u - v|, u * v]` — concatenated (BetBank's approach)

**Datasets (each run independently, same 3-way head comparison on each):**
- **NLI** (SNLI + MultiNLI combined for training, matching the SBERT paper) —
  3-way entailment / contradiction / neutral classification. Use SNLI's
  public test set for held-out eval (MultiNLI's official test labels aren't
  public; use MultiNLI dev-matched if desired). Drop "-" labeled pairs.
- **QQP** — binary duplicate detection. GLUE's QQP test labels aren't public;
  use GLUE QQP dev set as the effective test set, and carve a validation
  split out of train. Report F1 alongside accuracy (QQP is class-imbalanced).
- **PAWS** (PAWS-Wiki, labeled final) — binary paraphrase detection with high
  lexical overlap. Good stress test for whether multiplication adds signal
  beyond difference. Has standard public train/dev/test splits.

Use each dataset's own standard published splits — no custom splits, no
cross-validation. (Cross-validation would break comparability with published
splits; multiple seeds already provide the stability check CV is normally
used for.)

**Seeds:** 5 seeds per (dataset × variant) cell, varying linear head
initialization and data shuffling order. Total: 3 datasets × 3 variants × 5
seeds = 45 runs. Since the backbone is frozen, embeddings can be precomputed
and cached once per dataset — each run only trains a small linear head, so
this is cheap.

**Training:** Precompute/cache embeddings per dataset. Cross-entropy loss
(3-way for NLI; binary or 2-way softmax for QQP/PAWS, consistent across
variants). AdamW, small-head-appropriate learning rate (e.g. 1e-3–1e-4). Use
dev split for early stopping / checkpoint selection; report final numbers
only on held-out test split.

**Evaluation & statistics:** Report mean ± std across seeds per (dataset,
variant) cell. Paired significance test (paired t-test or Wilcoxon
signed-rank, given small seed count) comparing concatenated vs.
difference-only and concatenated vs. multiplication-only, per dataset. Be
upfront in the writeup about limited statistical power with ~5 seeds; report
effect sizes alongside p-values.

## Explicitly Out of Scope (for this core study)

- The BetBank-specific error-pattern targeted fine-tuning pass — not part of
  this ablation.
- STS-B — dropped entirely. In the original SBERT paper, STS-B is evaluated
  via plain cosine similarity between embeddings with no classifier head
  involved at all, so it can't test this project's actual question (head
  input representation). Retrofitting a regression head onto STS-B was
  considered and rejected as scope creep relative to using NLI (the
  classifier head's native task) plus QQP/PAWS.

## Planned Follow-Up (separate, later report)

Repeat the same 3-dataset × 3-variant × 5-seed design with the SBERT backbone
**fully fine-tuned end-to-end** (matching the original SBERT paper's actual
training procedure), to see whether the effect observed under the frozen,
fully-controlled setting persists, shrinks, or disappears when the backbone
is jointly trainable. This reintroduces a known confound (the backbone can
partially compensate for a weaker head) and is intended as a robustness
check / second paper, not as evidence for the core causal claim.

## Honesty / Reporting Principles

- Report results honestly even if the effect is negative, marginal, or
  inconsistent across datasets.
- Be explicit in the writeup about deviations from the original SBERT
  paper's procedure (frozen backbone in the core study) and why.
- Frame this as a focused replication/ablation stepping-stone artifact, not
  a novel architecture claim.

## ECE684_Paraphrase 

The directory contains the code originally used to train and analyze the model that 
powers BetBank. For reference, please look into this folder