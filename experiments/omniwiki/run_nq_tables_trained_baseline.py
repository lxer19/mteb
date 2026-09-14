"""Evaluate pinned E5 and random on the same complete NQ-Tables test corpus.

Only the evaluation view collapses identical query ID/text rows. The adapter,
source datasets, table serialization, and all qrels remain unchanged.
"""

# Standalone experiment: assertions check source integrity, not model quality.
# ruff: noqa: S101, T201, PLC2701

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

import mteb
from experiments.omniwiki.validate_nq_tables import _digest, _json_safe
from mteb.models.search_wrappers import SearchEncoderWrapper
from mteb.models.sentence_transformer_wrapper import _resolve_prompt
from mteb.tasks.retrieval.eng.nq_tables_retrieval import NQTablesRetrieval
from mteb.types import PromptType

if TYPE_CHECKING:
    from datasets import Dataset

    from mteb.models.models_protocols import EncoderProtocol
    from mteb.types import Array, RetrievalOutputType

MODEL = "intfloat/multilingual-e5-small"
REVISION = "fd1525a9fd15316a2d503bf26ab031a61d056e98"


def logical_queries(queries: Dataset) -> Dataset:
    """Keep the first row per ID, requiring exact agreement on every field."""
    first = {}
    indices = []
    for index, row in enumerate(queries):
        if row["id"] in first:
            assert row == first[row["id"]], "Conflicting duplicate query ID"
        else:
            first[row["id"]] = row
            indices.append(index)
    return queries.select(indices)


def evaluate(
    task: NQTablesRetrieval,
    model: EncoderProtocol,
    *,
    batch_size: int,
    normalize_embeddings: bool = False,
) -> dict[str, object]:
    """Use MTEB search and metrics unchanged; observe aggregate output depth."""
    wrapper = SearchEncoderWrapper(model, corpus_chunk_size=10_000)
    record = {
        "model": model.mteb_model_meta.name,
        "revision": model.mteb_model_meta.revision,
        "batch_size": batch_size,
        "corpus_chunk_size": 10_000,
        "requested_top_k": task._top_k,
        "similarity": "cosine",
        "normalize_embeddings": normalize_embeddings,
        "encoding_calls": [],
    }
    original_search, original_encode = wrapper.search, model.encode

    def observe_encode(inputs: object, **kwargs: object) -> Array:
        start = time.monotonic()
        embeddings = original_encode(inputs, **kwargs)
        assert torch.isfinite(torch.as_tensor(embeddings)).all()
        call = {
            "prompt_type": kwargs["prompt_type"].value,
            "items": len(embeddings),
            "dimension": embeddings.shape[1],
            "dtype": str(embeddings.dtype),
            "elapsed_seconds": time.monotonic() - start,
        }
        record["encoding_calls"].append(call)
        print("ENCODE", json.dumps(call), flush=True)
        return embeddings

    def observe_search(*args: object, **kwargs: object) -> RetrievalOutputType:
        results = original_search(*args, **kwargs)
        record["result_query_ids"] = len(results)
        record["retrieved_docs_per_query"] = dict(
            Counter(len(docs) for docs in results.values())
        )
        assert len(results) == 959
        assert set(map(len, results.values())) == {1000}
        return results

    encode_kwargs = {"batch_size": batch_size}
    if normalize_embeddings:
        encode_kwargs["normalize_embeddings"] = True
    start = time.monotonic()
    with (
        patch.object(model, "encode", side_effect=observe_encode),
        patch.object(wrapper, "search", side_effect=observe_search),
    ):
        record["scores"] = task.evaluate(
            wrapper, split="test", encode_kwargs=encode_kwargs
        )
    record["elapsed_seconds"] = time.monotonic() - start
    assert (
        sum(
            call["items"]
            for call in record["encoding_calls"]
            if call["prompt_type"] == "document"
        )
        == 169898
    )
    print("RESULT", json.dumps(_json_safe(record)), flush=True)
    return record


def main() -> None:
    """Run both models and append aggregate results to the existing report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Embedding device; 'auto' lets Sentence Transformers select one.",
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=Path("experiments/omniwiki/nqtables_validation/validation.json"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(2)
    report = json.loads(args.validation_json.read_text())
    task = NQTablesRetrieval(seed=42)
    task.load_data(cache_dir=str(args.cache_dir))
    data = task.dataset["default"]["test"]
    original_queries = data["queries"]
    original_qrels_digest = _digest(task.source_qrels["test"])
    mapped_qrels_digest = _digest(
        {"qid": qid, "did": did, "score": score}
        for qid, docs in data["relevant_docs"].items()
        for did, score in docs.items()
    )
    assert dict(task.metadata.dataset) == report["source"]
    assert original_qrels_digest == report["splits"]["test"]["qrel_source_sha256"]
    assert (
        _digest(original_queries.rename_column("id", "_id"))
        == (report["splits"]["test"]["queries"]["source_sha256"])
    )
    assert (
        _digest(data["corpus"].rename_column("id", "_id"))
        == (report["corpus"]["source_sha256"])
    )
    unique_queries = logical_queries(original_queries)
    assert (len(original_queries), len(unique_queries), len(data["corpus"])) == (
        966,
        959,
        169898,
    )
    baseline = {
        "source": dict(task.metadata.dataset),
        "split": "test",
        "source_query_rows": 966,
        "evaluated_unique_query_ids": 959,
        "deduplication": "First occurrence of each identical query ID/text row; distinct IDs retained even when text matches.",
        "corpus_items": 169898,
        "qrel_rows": len(task.source_qrels["test"]),
        "dangling_corpus_qrels_retained": 1,
        "source_qrels_sha256": original_qrels_digest,
        "source_queries_sha256": _digest(original_queries.rename_column("id", "_id")),
        "evaluation_queries_sha256": _digest(unique_queries.rename_column("id", "_id")),
        "source_corpus_sha256": report["corpus"]["source_sha256"],
        "seed": 42,
        "torch_threads": 2,
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        "requested_device": args.device,
        "python": platform.python_version(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "mteb_module_path": str(Path(mteb.__file__).resolve()),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "mteb",
                "torch",
                "transformers",
                "sentence-transformers",
                "datasets",
                "numpy",
            ]
        },
    }
    data["queries"] = unique_queries
    try:
        random_model = mteb.get_model("mteb/baseline-random-encoder", embed_dim=32)
        baseline["random_same_scope"] = evaluate(task, random_model, batch_size=256)
        start = time.monotonic()
        model = mteb.get_model(
            MODEL,
            revision=REVISION,
            device=None if args.device == "auto" else args.device,
        )
        baseline["model_load_seconds"] = time.monotonic() - start
        baseline["device"] = str(model.model.device)
        prompts = {
            kind.value: _resolve_prompt(model.model_prompts, task.metadata, kind)
            for kind in [PromptType.query, PromptType.document]
        }
        assert prompts == {"query": "query: ", "document": "passage: "}
        assert model.model.get_sentence_embedding_dimension() == 384
        assert model.model.max_seq_length == 512
        assert model.model.similarity_fn_name == "cosine"
        baseline["model_configuration"] = {
            "prompts": prompts,
            "embedding_dimension": model.model.get_sentence_embedding_dimension(),
            "max_seq_length": model.model.max_seq_length,
            "parameter_dtype": str(next(model.model.parameters()).dtype),
            "parameter_count": sum(p.numel() for p in model.model.parameters()),
            "modules": str(model.model),
            "model_config": model.model[0].auto_model.config.to_dict(),
            "pooling_config": model.model[1].get_config_dict(),
            "similarity": model.model.similarity_fn_name,
            "training_overlap_caveat": "E5 reports NQ in supervised training; exact overlap with these NQ-Tables queries was not audited.",
        }
        print("CONFIG", json.dumps(baseline["model_configuration"]), flush=True)
        baseline["trained"] = evaluate(
            task, model, batch_size=args.batch_size, normalize_embeddings=True
        )
    finally:
        data["queries"] = original_queries
    assert _digest(task.source_qrels["test"]) == original_qrels_digest
    assert (
        _digest(
            {"qid": qid, "did": did, "score": score}
            for qid, docs in data["relevant_docs"].items()
            for did, score in docs.items()
        )
        == mapped_qrels_digest
    )
    assert (
        _digest(original_queries.rename_column("id", "_id"))
        == baseline["source_queries_sha256"]
    )
    baseline["source_queries_restored_and_qrels_unchanged"] = True
    baseline["nonfinite_metric_handling"] = "Nonfinite metrics are serialized as null."
    # Preserve the historical random run and every existing validation finding.
    current_report = json.loads(args.validation_json.read_text())
    current_report["trained_model_sanity_baseline"] = _json_safe(baseline)
    args.validation_json.write_text(
        json.dumps(current_report, indent=2, allow_nan=False) + "\n"
    )
    print("Saved aggregate results:", args.validation_json, flush=True)


if __name__ == "__main__":
    main()
