# NQ-Tables local engineering prototype

Status: **implemented and validated locally on 13 September 2026**. All three splits load through the adapter; the full-corpus random baseline completes. Source integrity and task-quality checks have findings, recorded below without repairs.

This task preserves original NQ-Tables qrels and is an engineering baseline, not the final table-necessary OmniWikipedia benchmark.

## Source and unresolved license discrepancy

The exact source is [`ibm-research/NQTablesRetrieval`](https://huggingface.co/datasets/ibm-research/NQTablesRetrieval/tree/4962c33f5e651c82bc82c013893061202a47685e), revision **`4962c33f5e651c82bc82c013893061202a47685e`**. The [pinned card](https://huggingface.co/datasets/ibm-research/NQTablesRetrieval/blob/4962c33f5e651c82bc82c013893061202a47685e/README.md) identifies this as IBM's TableIR repackaging of Herzig et al.'s [NQ-Tables](https://aclanthology.org/2021.naacl-main.43/). The [original Google release instructions](https://github.com/google-research/tapas/blob/master/DENSE_TABLE_RETRIEVER.md) establish its NQ provenance. This adapter preserves the pinned IBM release's representation; it does not claim byte identity with Google's original interaction JSON.

The IBM card declares `cc-by-4.0`. The upstream [Natural Questions download page](https://ai.google.com/research/NaturalQuestions/download) identifies Creative Commons Share-Alike 3.0, and the [pinned upstream NQ card](https://huggingface.co/datasets/google-research-datasets/natural_questions/blob/e8103d566bef4154c2c12b17c6095ec5275840cc/README.md) declares `cc-by-sa-3.0`. The IBM card does not explain how these declarations relate, whether its declaration covers only added material, or whether another grant applies. A repository's Apache-2.0 code license does not establish this repackaged dataset's license.

**The final dataset license remains unresolved. No definitive claim is made for CC-BY-4.0, CC-BY-SA-3.0, or Apache-2.0.** This validation-only branch contains code, aggregate measurements, and notes; it does not redistribute dataset files or make a final license claim. No publisher was contacted.

`TaskMetadata.license` accepts `None`; the adapter uses that value. The task is deliberately **unregistered**: its class is not exported from `mteb.tasks` or included in any benchmark. Import it explicitly. Metadata schema/language validation passes, but publication metadata completeness is false. Besides the unresolved license, date, dialect, and BibTeX fields remain unset. No invented placeholder or global metadata-test exemption was added.

The measured run used local downloaded source files and Arrow caches. The committed JSON files contain aggregate measurements and hashes, not dataset copies. No dataset files were uploaded or redistributed.

## Inspection and implementation

Preflight inspected the simple `nq_retrieval.py` task, the table-as-Markdown `green_node_table_markdown_retrieval.py` task, and the table-adjacent `wiki_sql_retrieval.py` task. The current `AbsTaskRetrieval` and `RetrievalDatasetLoader` implementations use `dataset["default"][split]`. The standard loader's corpus/query split assumptions and qrel column names do not match this release, and its query filtering is inappropriate for source-preserving loading.

The executed plan was to add a task-local adapter, route each release split explicitly, share the corpus, test preservation/integrity behavior, and validate against the full release. Registration from the earlier plan was omitted to keep the unresolved metadata local.

Files:

- [Adapter](../../mteb/tasks/retrieval/eng/nq_tables_retrieval.py): `NQTablesRetrieval`, text-to-text Retrieval, English Latin script, `ndcg_at_10`, beta metadata, pinned source, `license=None`.
- [Focused tests](../../tests/test_tasks/test_nq_tables_retrieval.py): 14 offline tests covering default/explicit/incremental split routing, shared corpus identity, stable IDs and exact text, repeated and unjudged queries, multiple positives, dangling references, and prevention of silent conflicting-ID/qrel overwrites. Small artificial fixtures test the schema; they are not benchmark data.
- [Validation runner](validate_nq_tables.py): loads every split, checks preservation, computes MTEB statistics, invokes existing quality checks, and runs the existing random encoder.
- [Trained-model evaluation runner](run_nq_tables_trained_baseline.py): second validation script; reruns the random encoder and the pinned `intfloat/multilingual-e5-small` on the full test corpus over a deduplicated evaluation view and appends `trained_model_sanity_baseline` to `validation.json`.
- [Measured validation and baseline results](nqtables_validation/validation.json).
- [MTEB descriptive statistics for all splits](nqtables_validation/descriptive_stats.json).

No core retrieval loader, task registry, benchmark grouping, or unrelated file was modified. The pre-existing staged image-statistics changes were left untouched.

## Schema, serialization, and loader adaptations

| Role | HF configuration | HF split | Source fields | Source file |
|---|---|---|---|---|
| Shared corpus | `corpus_md` | `corpus_md` | `_id`, `title`, `text` | `corpus_md.jsonl` |
| Queries | `queries` | `{train,dev,test}_queries` | `_id`, `text` | `{split}_queries.jsonl` |
| Qrels | `default` | `train`, `dev`, `test` | `qid`, `did`, `score` | `{split}_qrels.jsonl` |

The corpus is **exactly the released `corpus_md.text` strings**: supplied page/table heading followed by the existing Markdown table, preserving Unicode, whitespace, headers, cells, repetitions, and trailing newlines. All 169,898 supplied `title` fields are empty strings and remain empty. Neither alternate corpus serialization is loaded.

The adapter only renames query/corpus `_id` columns to MTEB `id` and maps each `(qid, did, score)` into `relevant_docs[qid][did]`. Source ID strings are not normalized. All qrel rows are also retained in `task.source_qrels`. No score is changed; all measured source scores are `1`. No duplicate qrel pairs exist in this revision, so the nested mapping loses no qrel rows.

Repeated identical query-ID rows remain in the query datasets. Conflicting query texts under one ID, duplicate corpus IDs, or repeated qrel pairs would raise rather than silently overwrite records; none occur in this pinned release. **Dangling references remain in the qrels** and are reported by validation. An initial fail-fast integrity check was replaced with this preservation behavior when the real release exposed dangling references.

The loader defaults to test. `filter_eval_splits(["train", "dev", "test"])` explicitly requests all splits. It routes by release split names, never by ID prefixes: for example, dev questions have `train_` source prefixes and test questions have `dev_` prefixes. Repeated loads reuse already loaded splits; later explicit loads can add splits using the same corpus object.

One downstream distinction is material: MTEB's existing `_corpus_to_dict` calls `.strip()` when preparing encoder inputs and descriptive statistics. All 169,898 corpus strings differ at that stage because of boundary whitespace. **The adapter's stored corpus remains byte-for-byte equivalent in UTF-8 text content to the release.** This standard model-input preprocessing was measured and retained; no core code or serialized table was rewritten. Corpus text contains 417,919,068 raw characters and 417,749,170 characters after standard preprocessing.

## Measured counts and integrity

All measurements cover the complete pinned release. Corpus counts refer to the same shared corpus, not three distinct copies.

| Measurement | train | dev | test |
|---|---:|---:|---:|
| Query rows | 9,594 | 1,068 | 966 |
| Unique query IDs | 9,534 | 1,067 | 959 |
| Corpus items / unique corpus IDs | 169,898 | 169,898 | 169,898 |
| Qrel rows | 9,594 | 1,068 | 966 |
| Unique queries with 1 positive | 9,474 | 1,066 | 952 |
| Unique queries with 2 positives | 60 | 1 | 7 |
| Repeated query IDs | 60 | 1 | 7 |
| Extra query rows beyond first occurrence | 60 | 1 | 7 |
| Conflicting query IDs | 0 | 0 | 0 |
| Duplicate corpus IDs / duplicate qrel pairs | 0 / 0 | 0 / 0 | 0 / 0 |
| Empty or whitespace-only queries / tables | 0 / 0 | 0 / 0 | 0 / 0 |
| Queries without qrels | 0 | 0 | 0 |
| Dangling qrel query rows | 0 | 0 | 0 |
| Dangling qrel corpus rows | **4** | **1** | **1** |
| Distinct missing corpus IDs within split | 2 | 1 | 1 |

There are 11,560 unique query IDs and 11,628 query/qrel rows across all splits. Query-ID overlap between every pair of splits is zero. Positives count the original score-positive qrels, including dangling references; the counts do not imply semantic validity or corpus presence. Unmarked tables are not declared negative by this audit.

Every qrel query ID exists: **pass**. Every qrel corpus ID exists: **fail** in all three splits. The six dangling rows contain quoted/doubled-quote document IDs in the source qrels. Their source-layer presence is observed; a quoting/export mistake is a plausible explanation, not an established diagnosis. They were neither normalized nor dropped.

All query rows compare equal to the source after reversing the column rename. Ordered corpus record hashes match before/after adaptation:

`fa3a30712de3dc44e6d804f800760a69c5e4aa6dad30a212d2210b9ee3ec97ee`

The runner hashes ordered JSON records using sorted keys, unescaped UTF-8 strings, and 8-byte length framing; this is a logical-record digest, not the downloaded file's byte hash. Per-split query hashes and source qrel hashes are in `validation.json`. Every source qrel triple matches the mapped triples, including multiplicity. Cache paths recorded in that file resolve to the pinned revision.

The previously traced Italy query `dev_3052797144690241914_0_0` remains unchanged with gold ID `Italian_general_election,_2018_B19E32FB82D4E3D7` and score `1`. Its table-text SHA-256 still equals `329109192b73a661d39027e5ddc7a78ceddaa1843167f5fbc5e4e465921046ba`, including the released `TBD` content. This is a preservation regression check, not a renewed semantic audit.

## Statistics and applicable quality checks

The runner invokes MTEB's `_calculate_descriptive_statistics_from_split` for each split and the existing `_split_quality` checks, with the repository's normal warning classification. Statistics are saved locally instead of the registered-task statistics directory. No findings were exempted or repaired.

- **Four errors:** `missing_qrel_corpus_ids` in train/dev/test; `duplicate_text:queries_text_statistics` in test (966 rows, 919 distinct question strings). Distinct question strings and distinct IDs are different counts.
- **Three warnings:** `long_text:documents_text_statistics`, once per split for the shared corpus. Maximum model-input table length is 285,798 characters versus mean 2,458.823 and minimum 39. This heuristic warning does not establish an extraction error.
- Every model-input table text is distinct. Query lengths in characters are train 26–99 (mean 46.210), dev 27–97 (mean 45.996), and test 25–97 (mean 46.539).
- Metadata schema validation passes; publication completeness does not. The global registered-task metadata test is not claimed to pass for this unregistered prototype.
- **14 focused tests passed.** Ruff lint and formatting checks passed on the three added Python files.

The test environment's `pytest-rerunfailures` plugin tried to bind a localhost socket, which the sandbox disallows. Tests ran with that plugin and its configured retry arguments disabled; no test cases were skipped. Task modules are excluded from errors by the repository's existing mypy configuration, so no meaningful task type-check pass is claimed.

## Full-corpus engineering baseline

Ran the existing `mteb/baseline-random-encoder`, revision `1`, with 32-dimensional float32 embeddings, deterministic text-conditioned random vectors, task seed 42, cosine similarity, batch size 256, two Torch threads, and the standard `SearchEncoderWrapper` with corpus chunks of 10,000. No model download or training was needed.

Scope: all **169,898 tables**, all **966 test query rows / 959 query IDs**, and all **966 original test qrels**, including the dangling one. Requested top-k was 1,000. This was an exhaustive full-corpus search, not a sampled-corpus run. `AbsTaskRetrieval.evaluate` completed; source query rows and qrels remained unchanged after evaluation.

| Metric | Measured value |
|---|---:|
| nDCG@10 / main score | 0.00000 |
| Recall@10 | 0.00000 |
| nDCG@100 | 0.00022 |
| Recall@100 | 0.00104 |
| nDCG@1000 | 0.00040 |
| Recall@1000 | 0.00261 |
| MRR@1000 | 0.0000423977 |
| Hit rate@1000 | 0.00313 |

**Observed retrieval limitation:** 952 query IDs received 1,000 distinct results, while the seven duplicated query IDs received only 500. The existing wrapper inserts results from both identical query rows into the same bounded heap, then collapses duplicate document IDs into the output dictionary. Thus this is a successful execution baseline with a top-1,000 result-depth limitation for those seven IDs, not proof of correct top-1,000 retrieval for every query. The reported metrics use those actual outputs. No query deduplication or wrapper repair was introduced.

Some nAUC metrics are undefined for all-zero effectiveness vectors; nonfinite values are recorded as JSON `null`, not substituted with zero. Runtime and all returned metrics are in `validation.json`. Scores are engineering observations under the original qrels, not evidence of table necessity, model usefulness, or label validity.

## Trained-model sanity baseline

**Completed locally on 13 September 2026.** The trained model shows clear retrieval signal above random, while its main nDCG@10 and Recall@10 are far from perfect. This supports the **“non-random, non-trivial” sanity-check item for this engineering baseline**, not dataset validity or completion of the other PR checklist items.

Metrics below are MTEB's returned values on a 0–1 scale; each comparison is **E5 / random rerun on the identical evaluation scope**:

- **nDCG@10:** 0.52530 / 0.00000.
- **Recall@10:** 0.70229 / 0.00000.
- **Recall@100:** 0.90250 / 0.00104.
- **Recall@1000:** 0.98123 / 0.00261.
- **MRR@10:** 0.4694138239 / 0.0000000000.
- **MRR@1000:** 0.4783773524 / 0.0000423977.
- **Evaluation wall time:** 1,500.297 seconds (25 minutes) / 6.756 seconds. E5 loading from the local cache took another 0.916 seconds; dataset loading and integrity checks are outside these evaluation timers.

Recall@1000 is high (98.123%); the non-triviality conclusion rests on top-10 ranking performance, especially the task's nDCG@10 main score, rather than claiming every metric is far from perfect. These are descriptive single-run results, without significance testing. The random rerun's listed effectiveness metrics equal the historical random run, despite restoring full result depth for the seven repeated query IDs.

The complete returned metrics, exact model configuration, package versions, per-chunk encoding counts/times, source hashes, and scope are recorded under `trained_model_sanity_baseline` in [validation.json](nqtables_validation/validation.json). All 169,898 documents and all 959 logical queries were encoded by E5, every observed embedding was finite and 384-dimensional, and **both runs returned 1,000 distinct documents for each of the 959 queries**. Post-evaluation assertions confirmed that source query rows were restored and every original qrel remained unchanged. No reduced-corpus evaluation was run.

Model: **`intfloat/multilingual-e5-small`**, pinned revision **`fd1525a9fd15316a2d503bf26ab031a61d056e98`**. No substitute model is used. The [pinned model card](https://huggingface.co/intfloat/multilingual-e5-small/blob/fd1525a9fd15316a2d503bf26ab031a61d056e98/README.md) and MTEB's `mteb/models/model_implementations/e5_models.py` specify `query: ` for questions and `passage: ` for documents. MTEB's registered model wrapper supplies these prefixes; the stored source strings are unchanged.

Scope: the same `ibm-research/NQTablesRetrieval@4962c33f5e651c82bc82c013893061202a47685e` release, **all 169,898 shared Markdown tables**, **959 logical test query IDs**, and **all 966 original test qrels**, including the one dangling corpus reference. Seven repeated, identical ID/text rows are collapsed only in the temporary evaluation view. Distinct IDs with identical question text remain distinct; there are 919 distinct question strings. The source query dataset is retained and restored afterward. Qrels are neither filtered nor repaired. Source corpus, query, and qrel hashes are asserted equal to the previous validation report before evaluating.

Configuration: Apple M4 Max with 36 GiB unified memory; PyTorch MPS, float32 model parameters and returned embeddings; 117,653,760 parameters; 384 dimensions; mean pooling including prefix tokens; L2-normalized E5 embeddings; cosine similarity; model maximum 512 tokens; E5 batch size 64; corpus chunks of 10,000; requested top-k 1,000; seed 42; two Torch threads and eight tokenizer threads. The released Markdown is not rewritten or manually shortened. Standard MTEB boundary-whitespace handling remains as described above, and the model's normal 512-token truncation can omit later table cells.

The [evaluation runner](run_nq_tables_trained_baseline.py) calls the normal `SearchEncoderWrapper` and `AbsTaskRetrieval.evaluate`; it implements no retrieval metrics. It also reruns the existing 32-dimensional random encoder with batch size 256 on exactly the same deduplicated evaluation view and complete corpus. The historical random result remains in the JSON's original `baseline` entry. The evaluation-only removal of repeated identical ID/text rows eliminates the historical result-depth limitation without modifying the loader or search wrapper.

Validation: the 14 existing adapter tests passed; the runner passes Ruff lint and formatting checks. Additional direct checks confirmed that its evaluation view collapses identical ID/text rows, retains distinct IDs with identical text, leaves source rows unchanged, and rejects conflicting texts for one ID. The release's existing task-quality errors and warnings remain recorded; the trained scores do not exempt them.

Operational notes: an initial attempt lost its process and temporary cache before any E5 result was saved; the cause is unknown. The same pinned inputs were restored under Git-ignored `.cache/nqtables-baseline/`. A subsequent serial-tokenizer attempt was stopped because early long-table batches took about seven seconds each. A timing-only probe on 64 existing long tables found approximately one second per batch with eight tokenizer threads; it computed no retrieval metrics. Float16 was timed but not selected. The final full evaluation retains float32 and uses tokenizer parallelism. Download time, failed attempts, and timing probes are excluded from the separately measured final evaluation runtime.

Reproduce after populating the pinned local caches:

```bash
HF_HOME="$PWD/.cache/nqtables-baseline/hf" \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=8 \
  .venv/bin/python -u -m experiments.omniwiki.run_nq_tables_trained_baseline \
  --cache-dir "$PWD/.cache/nqtables-baseline/datasets" \
  --device mps --batch-size 64
```

Caveats: E5's model card lists NQ among supervised training datasets; exact overlap with these NQ-Tables questions was not audited. This comparison can establish retrieval signal under the original qrels, but cannot establish unseen-data generalization, gold-label validity, table sufficiency or necessity, novelty, paper-score reproduction, benchmark readiness, or a final license. The existing task-quality findings and unresolved license discrepancy remain unchanged.

This task preserves original NQ-Tables qrels and is an engineering baseline, not the final table-necessary OmniWikipedia benchmark.

## Reproduction

Run from the repository root:

```bash
HF_HOME=/private/tmp/nqtables-prototype-cache/hf HF_HUB_DISABLE_XET=1 \
  .venv/bin/python -m experiments.omniwiki.validate_nq_tables \
  --cache-dir /private/tmp/nqtables-prototype-cache/datasets

.venv/bin/python -m pytest -p no:rerunfailures -o addopts='' \
  tests/test_tasks/test_nq_tables_retrieval.py -q

.venv/bin/ruff check mteb/tasks/retrieval/eng/nq_tables_retrieval.py \
  tests/test_tasks/test_nq_tables_retrieval.py experiments/omniwiki/validate_nq_tables.py
.venv/bin/ruff format --check mteb/tasks/retrieval/eng/nq_tables_retrieval.py \
  tests/test_tasks/test_nq_tables_retrieval.py experiments/omniwiki/validate_nq_tables.py
```

The measured validation used `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` after downloading the three pinned configurations into the dedicated cache. Offline datasets emitted its “latest cached version” message; the cache paths identify the specified revision, with no other source revision used. Run without offline flags to populate a fresh cache.

Local use:

```python
from mteb.tasks.retrieval.eng.nq_tables_retrieval import NQTablesRetrieval

task = NQTablesRetrieval()
task.filter_eval_splits(["train", "dev", "test"])
task.load_data(cache_dir="/private/tmp/nqtables-prototype-cache/datasets")
```

The validator used local checkout `96eb0d3394b50f02e2307276aaa6b7a93c162b22`, plus the existing workspace changes and prototype files. Python 3.14.6, datasets 5.0.0, NumPy 2.4.6, Torch 2.11.0, pytest 8.3.5. The checkout declares MTEB 2.20.3; installed distribution metadata reports 2.19.5. Imports resolve to this local checkout, not a separately installed implementation. The exact module path and distribution versions are recorded in the report.

## What this prototype proves

The pinned Markdown release can be adapted to local MTEB retrieval, loaded across train/dev/test with a shared corpus, inspected by existing statistical/quality tools, and evaluated over the full corpus. The adapter preserves source questions, table content, IDs, and every original qrel. Validation exposes concrete source-reference problems and a duplicate-query retrieval limitation without changing them.

## What it does NOT prove

It does not establish that every gold table satisfies its question, that unmarked tables are negative, that all source references are valid, or that the retrieval wrapper returns a complete top 1,000 for every query ID. It does not resolve licensing or authorize publication/redistribution. It does not establish a table-necessary subset, dataset novelty, benchmark readiness, or the final OmniWikipedia benchmark design. No new questions, A/B/C/D filters, repaired labels, added prose, rendered images, live Wikipedia evidence, or dataset uploads were used.
