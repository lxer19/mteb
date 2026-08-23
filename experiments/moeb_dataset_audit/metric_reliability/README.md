# Classification metric-reliability audit

Reproduces the label-distribution and leaderboard findings behind the MOEB metric-reliability
audit. The central question is not whether a dataset is imbalanced, but whether its configured
`main_score` remains interpretable under MTEB's evaluation protocol.

MTEB classification probes train on at most eight examples per label, producing a balanced
training sample, and then evaluate on the natural test distribution. For accuracy-scored tasks,
the relevant trivial baseline is therefore the test split's majority-class share. A negative
`majority_gap = best published score - majority share` means a constant predictor outscores every
published model.

## Run it

From an MTEB checkout:

```bash
python experiments/moeb_dataset_audit/metric_reliability/metric_reliability_audit.py \
  --with-results --json metric_reliability.json
```

The offline portion reads only `mteb/descriptive_stats/**/*.json` and task metadata. It downloads
no datasets and runs no model inference. `--with-results` additionally reads the public MTEB result
cache—the same scores displayed by the leaderboard—and requires at least five submissions before
attaching empirical evidence.

Restrict the same analysis to a registered benchmark:

```bash
python experiments/moeb_dataset_audit/metric_reliability/metric_reliability_audit.py \
  --benchmark 'MTEB(Multilingual, v1)' --benchmark MMTEB \
  --with-results --json mmteb_metric_reliability.json
python experiments/moeb_dataset_audit/metric_reliability/metric_reliability_audit.py \
  --benchmark 'MIEB(eng)' --benchmark 'MIEB(Multilingual)' --benchmark 'MIEB(Img)' \
  --with-results --json mieb_metric_reliability.json
python experiments/moeb_dataset_audit/metric_reliability/metric_reliability_audit.py \
  --benchmark MAEB --benchmark 'MAEB(audio-only)' \
  --with-results --json maeb_metric_reliability.json
python experiments/moeb_dataset_audit/metric_reliability/metric_reliability_audit.py \
  --benchmark MVEB --benchmark 'MVEB(text, video)' --benchmark 'MVEB(video)' \
  --with-results --json mveb_metric_reliability.json
python experiments/moeb_dataset_audit/metric_reliability/metric_reliability_audit.py \
  --benchmark 'MTEB(eng, v1)' --benchmark 'MTEB(eng)' \
  --with-results --json mteb_eng_metric_reliability.json
```

List empirically inverted tasks with `jq`:

```bash
jq -r '.audits[] | select(.evidence.majority_gap < 0) |
  [.task, .distribution.majority_share, .evidence.best, .evidence.majority_gap] | @tsv' \
  metric_reliability.json
```

## Signals and thresholds

- **R1 — accuracy baseline inversion.** Majority share `>=0.90` is HIGH, `0.70–0.90` is
  REVIEW, and `0.50–0.70` is NOTE. The 0.70 break is empirically calibrated; above it the
  observed inversion rate jumps sharply.
- **R2 — macro resolution.** HIGH when effective macro N is below 50 or any class is a
  singleton; REVIEW below 200. Effective N is `k² / sum(1/m_c)`, and the reported macro noise
  floor is `(1/2k) * sqrt(sum(1/m_c))`. These thresholds are provisional.
- **R3 — AP comparability.** HIGH when positive prevalence is at least 0.90 and REVIEW at
  0.75. A random ranker's expected AP equals prevalence, so raw AP is not comparable across
  tasks with different prevalences. These thresholds are provisional.

The axes remain separate categories rather than being collapsed into a scalar. All five core
distribution statistics—majority share, accuracy budget, effective macro N, macro noise floor,
and minority count—come from label counts already committed to the repository.

## Expected umbrella-benchmark finding

Retrieved 2026-08-22. Public-result counts and best scores will change as submissions land.

Five single-label tasks in MMTEB, MAEB, or MVEB had `majority_gap < 0`:

```text
MELDVideoClassification          MVEB   -0.312512
VoxCelebSA                       MAEB   -0.235995
CommonLanguageAgeDetection       MAEB   -0.214400
MELDVideoZeroShot                MVEB   -0.020996
SwissJudgementClassification     MMTEB  -0.013737
```

No negative gap was found in MIEB or the English MTEB v1/v2 suites. Separately,
`ToxicConversationsClassification` is R1 HIGH in both MMTEB and English MTEB, but its best
published model still clears the majority baseline by `+0.0550`.

## Caveats

- This script intentionally excludes multilabel classification. A label's frequency share is
  not the constant-predictor baseline for multilabel exact-match accuracy, so subtracting it from
  a model score would create a meaningless `majority_gap`.
- A skewed distribution is not a data defect. R1 diagnoses a metric/protocol mismatch.
- R2 can flag a singleton class even when the majority share is low; that is a macro-resolution
  problem, not baseline inversion.
- Empirical evidence never changes a distribution-derived tier, and tasks without five public
  submissions remain auditable offline.
