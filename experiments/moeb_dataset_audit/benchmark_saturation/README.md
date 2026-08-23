# Frontier-saturation probe

Reproduces the numbers in [#5271 — Measuring frontier saturation in MOEB tasks](https://github.com/embeddings-benchmark/mteb/issues/5271).

Measures whether a task still separates models **at the frontier**, or whether the strongest models
are compressed against the metric ceiling. A task can stay useful for weak-vs-strong comparisons long
after it stops ranking frontier models apart.

## Run it

```bash
pip install mteb          # tested on mteb==2.19.5
python saturation_probe.py
```

Reads the **public MTEB result cache** — the same scores the leaderboard displays. No model
inference, no dataset downloads, no credentials. A cold run takes a few minutes (it fetches the
results repo into `~/.cache/mteb`); after that it's seconds.

## Expected output

```
| task                            | N  | best  | dist-to-ceiling | top-10 spread | top-10 spread (fam) | within 2pp  | tied at best | verdict                |
|---------------------------------|----|-------|-----------------|---------------|---------------------|-------------|--------------|------------------------|
| VidoreSyntheticDocQAAIRetrieval | 92 | 1.000 | 0.00 pp         | 0.37 pp       | 0.37 pp             | 34/92 (37%) | 6            | SATURATED AT FRONTIER  |
| VidoreTatdqaRetrieval           | 92 | 0.857 | 14.33 pp        | 2.43 pp       | 3.55 pp             | 8/92 (9%)   | 1            | HEALTHY DISCRIMINATION |
```

Retrieved 2026-08-22. **N grows as new submissions land**, so re-running later will shift the
numbers — pin the retrieval date when quoting them.

Other usages:

```bash
python saturation_probe.py VidoreDocVQARetrieval --top 20            # any task by name
python saturation_probe.py --benchmark "MIEB(lite)" --json out.json  # rank a whole benchmark
```

Sweeping all 51 MIEB(lite) tasks gives **1 saturated, 3 frontier-compressed, 2 mixed, 45 healthy** —
saturation is a per-task property, not a benchmark-wide one.

## Signals

| signal | definition |
|---|---|
| `dist_to_ceiling` | `1 - best`, in pp. Assumes the main score is bounded at 1.0 (true for the ndcg/ap/accuracy main scores MTEB uses). |
| `topK_spread` | `best - Kth_best`, in pp. |
| `within_Xpp` | submissions scoring `>= best - X pp`. |
| `tied at best` | submissions at the exact best score. |
| family-dedup (`fam`) | the same signals after collapsing model families, so stacked variants (`2b`/`4b`/`9b`, `bf16`, `v0`) can't make the frontier look artificially tight. `Argus-Colqwen3.5-9b-v0-bf16` → `datascience-uibk/argus-colqwen3.5`. Coarse regex, not a curated taxonomy — it only has to stop one model line from occupying the whole frontier. |

## Caveats

- **Compression ≠ saturation.** The probe flags `NIGHTSI2IRetrieval` (best 0.265, 31/55 within 2pp)
  and `ImageCoDe` (best 0.152, 21/49 within 2pp) as frontier-compressed. Those are *floor* effects —
  task too hard or metric too insensitive — the opposite cause with the same symptom. So
  `dist_to_ceiling` alone would miss them, and `within_Xpp` alone would mislabel them as saturated.
  Any benchmark-health indicator needs both ends.
- **N counts submissions, not distinct models.** Variants of one model line inflate it; that's what
  the family-dedup column is for.
- **Scores pool across `dataset_revision`.** mteb's loader joins *model* revisions but doesn't filter
  by *dataset* revision. For ViDoRe this is safe — the revisions are a benign `vidore/`→`mteb/` org
  re-upload (verified via HF commit history and file sizes: a 303 MB corpus differing by 9 bytes) —
  so pooling matches what the leaderboard shows. It would not be safe for a task whose data changed
  without a task-version bump.
