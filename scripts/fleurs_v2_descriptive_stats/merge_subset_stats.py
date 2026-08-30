"""Exact merge of per-language descriptive statistics into an aggregate.

`_calculate_descriptive_statistics_from_split(compute_overall=True)` concatenates every
subset and materialises `corpus["audio"]`, which is 65 GB for FLEURS. Every field it
produces is decomposable, so the aggregate can be assembled from per-language results
plus the per-language content-hash and id sets (~2.5 MB total).
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def _merge_audio(parts: list[dict], counts: list[int], hashes: set[str]) -> dict | None:
    parts = [p for p in parts if p]
    if not parts:
        return None
    total = sum(p["total_duration_seconds"] for p in parts)
    n = sum(counts)
    # JSON round-trips dict keys to strings; the in-memory form uses ints
    rates: Counter[int] = Counter()
    for p in parts:
        for rate, c in p["sampling_rates"].items():
            rates[int(rate)] += c
    return {
        "total_duration_seconds": total,
        "min_duration_seconds": min(p["min_duration_seconds"] for p in parts),
        "average_duration_seconds": total / n,
        "max_duration_seconds": max(p["max_duration_seconds"] for p in parts),
        "unique_audios": len(hashes),
        "average_sampling_rate": sum(r * c for r, c in rates.items()) / n,
        "sampling_rates": dict(rates),
    }


def _merge_text(parts: list[dict], counts: list[int], hashes: set[str]) -> dict | None:
    parts = [p for p in parts if p]
    if not parts:
        return None
    total = sum(p["total_text_length"] for p in parts)
    return {
        "total_text_length": total,
        "min_text_length": min(p["min_text_length"] for p in parts),
        "average_text_length": total / sum(counts),
        "max_text_length": max(p["max_text_length"] for p in parts),
        "unique_texts": len(hashes),
    }


def _merge_relevant(parts: list[dict], n_qrel_queries: int, doc_ids: set[str]) -> dict:
    total = sum(p["num_relevant_docs"] for p in parts)
    return {
        "num_relevant_docs": total,
        "min_relevant_docs_per_query": min(p["min_relevant_docs_per_query"] for p in parts),
        "average_relevant_docs_per_query": total / n_qrel_queries,
        "max_relevant_docs_per_query": max(p["max_relevant_docs_per_query"] for p in parts),
        "unique_relevant_docs": len(doc_ids),
        "num_missing_query_ids": sum(p["num_missing_query_ids"] for p in parts),
        "num_missing_corpus_ids": sum(p["num_missing_corpus_ids"] for p in parts),
    }


def merge_subsets(subsets: dict[str, dict], extras: dict[str, dict]) -> dict[str, Any]:
    """Merge per-language stats into the aggregate `compute_overall=True` would produce.

    `extras[lang]` carries the hash/id sets that cannot be recovered from the stats dict:
    `doc_audio_hashes`, `query_audio_hashes`, `doc_text_hashes`, `query_text_hashes`,
    `qrel_doc_ids`.
    """
    langs = list(subsets)
    n_docs = [subsets[l]["num_documents"] for l in langs]
    n_queries = [subsets[l]["num_queries"] for l in langs]

    def union(key: str) -> set[str]:
        out: set[str] = set()
        for l in langs:
            out |= extras[l][key]
        return out

    merged = {
        "num_samples": sum(subsets[l]["num_samples"] for l in langs),
        "num_queries": sum(n_queries),
        "num_documents": sum(n_docs),
        "number_of_characters": sum(subsets[l]["number_of_characters"] for l in langs),
        "documents_text_statistics": _merge_text(
            [subsets[l]["documents_text_statistics"] for l in langs], n_docs,
            union("doc_text_hashes")),
        "documents_image_statistics": None,
        "documents_audio_statistics": _merge_audio(
            [subsets[l]["documents_audio_statistics"] for l in langs], n_docs,
            union("doc_audio_hashes")),
        "documents_video_statistics": None,
        "queries_text_statistics": _merge_text(
            [subsets[l]["queries_text_statistics"] for l in langs], n_queries,
            union("query_text_hashes")),
        "queries_image_statistics": None,
        "queries_audio_statistics": _merge_audio(
            [subsets[l]["queries_audio_statistics"] for l in langs], n_queries,
            union("query_audio_hashes")),
        "queries_video_statistics": None,
        "relevant_docs_statistics": _merge_relevant(
            [subsets[l]["relevant_docs_statistics"] for l in langs],
            sum(n_queries), union("qrel_doc_ids")),
        "top_ranked_statistics": None,
    }
    return merged
