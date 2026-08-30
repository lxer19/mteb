"""Recompute FLEURS `.v2` descriptive statistics through MTEB's own code path.

One language at a time, checkpointing after each, so an interruption costs at most one
language. Uses `_calculate_descriptive_statistics_from_split` unchanged; the aggregate is
assembled afterwards by `assemble_and_diff.py`, using the merge implemented in
`merge_subset_stats.py`, because the built-in `compute_overall=True` would materialise
all 77,809 decoded clips (~65 GB) at once.

Run through `run_official_stats.sh`, which supplies the FFmpeg library path.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import mteb
from mteb.abstasks._statistics_calculation import (
    compute_audio_hashes,
    compute_text_hashes,
)

SPLIT = "test"
TASKS = ["FleursT2ARetrieval.v2", "FleursA2TRetrieval.v2"]
CKPT = Path(
    os.environ.get("FLEURS_STATS_CKPT_DIR")
    or Path(__file__).parent / "results" / "official_rerun"
)


def _hashes(ds, modality: str) -> list[str]:
    if modality == "audio":
        return compute_audio_hashes(ds["audio"]) if "audio" in ds.column_names else []
    return compute_text_hashes(ds["text"]) if "text" in ds.column_names else []


def process(task_name: str, lang: str) -> dict:
    """Official per-subset statistics plus the hash/id sets the aggregate needs."""
    task = mteb.get_task(task_name, hf_subsets=[lang])
    task.load_data()
    stats = task._calculate_descriptive_statistics_from_split(SPLIT, hf_subset=lang)

    task.convert_v1_dataset_format_to_v2(None)
    sd = task.dataset[lang][SPLIT]
    out = {
        "stats": stats,
        "doc_audio_hashes": sorted(set(_hashes(sd["corpus"], "audio"))),
        "query_audio_hashes": sorted(set(_hashes(sd["queries"], "audio"))),
        "doc_text_hashes": sorted(set(_hashes(sd["corpus"], "text"))),
        "query_text_hashes": sorted(set(_hashes(sd["queries"], "text"))),
        "qrel_doc_ids": sorted(
            {f"{SPLIT}_{lang}_{d}" for rel in sd["relevant_docs"].values() for d in rel}
        ),
    }
    del task, sd
    gc.collect()
    return out


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    langs = list(mteb.get_task(TASKS[0]).metadata.eval_langs)
    todo = [(t, l) for l in langs for t in TASKS]
    done = {p.stem for p in CKPT.glob("*.json")}

    remaining = [(t, l) for t, l in todo if f"{t}__{l}" not in done]
    print(f"{len(todo)} units total, {len(todo) - len(remaining)} already done, "
          f"{len(remaining)} remaining", flush=True)

    t0 = time.time()
    for i, (task_name, lang) in enumerate(remaining, 1):
        key = f"{task_name}__{lang}"
        started = time.time()
        try:
            result = process(task_name, lang)
        except Exception as exc:  # keep going; a failed language is reported at the end
            print(f"[{i}/{len(remaining)}] {key}: FAILED {type(exc).__name__}: {exc}",
                  flush=True)
            continue
        (CKPT / f"{key}.json").write_text(json.dumps(result))
        elapsed = time.time() - started
        rate = (time.time() - t0) / i
        eta = rate * (len(remaining) - i) / 3600
        print(f"[{i}/{len(remaining)}] {key}: "
              f"q={result['stats']['num_queries']} d={result['stats']['num_documents']} "
              f"({elapsed:.0f}s, ETA {eta:.1f}h)", flush=True)

    missing = [f"{t}__{l}" for t, l in todo if not (CKPT / f"{t}__{l}.json").exists()]
    print(f"\ncheckpoints: {len(todo) - len(missing)}/{len(todo)}")
    if missing:
        print("MISSING:", ", ".join(missing[:10]), "..." if len(missing) > 10 else "")
        sys.exit(1)
    print("done")


if __name__ == "__main__":
    main()
