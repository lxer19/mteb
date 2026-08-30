"""Assemble the official FLEURS `.v2` stats from per-language checkpoints.

`_calculate_descriptive_statistics_from_split(compute_overall=True)` would concatenate
every language and materialise `corpus["audio"]` (~65 GB for FLEURS) in one call. Instead,
`rerun_official_stats.py` calls the same underlying function once per language, and this
script merges the per-language results with `merge_subset_stats.merge_subsets` and writes
the result to the committed stats path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from merge_subset_stats import merge_subsets

SPLIT = "test"
CKPT = Path(__file__).parent / "results" / "official_rerun"
STATS = Path("mteb/descriptive_stats/Image/Any2AnyRetrieval")
HASH_KEYS = ("doc_audio_hashes", "query_audio_hashes",
             "doc_text_hashes", "query_text_hashes", "qrel_doc_ids")


def build(task_name: str) -> tuple[dict, int]:
    subsets, extras = {}, {}
    for f in sorted(CKPT.glob(f"{task_name}__*.json")):
        lang = f.stem.split("__", 1)[1]
        d = json.loads(f.read_text())
        subsets[lang] = d["stats"]
        extras[lang] = {k: set(d[k]) for k in HASH_KEYS}
    agg = merge_subsets(subsets, extras)
    agg["hf_subset_descriptive_stats"] = subsets
    return {SPLIT: agg}, len(subsets)


def diff(new: dict, old_path: Path) -> list[str]:
    """Field-by-field differences between `new` and the stats file at `old_path`."""
    old = json.loads(old_path.read_text())[SPLIT]
    cur = new[SPLIT]
    out = []

    def cmp(scope: str, a: dict, b: dict) -> None:
        for k in a:
            if k == "hf_subset_descriptive_stats":
                continue
            x, y = a[k], b.get(k)
            if x == y:
                continue
            if isinstance(x, dict) and isinstance(y, dict):
                for kk in x:
                    # JSON round-trips dict keys to strings on both sides
                    if str(x[kk]) != str(y.get(kk)):
                        out.append(f"{scope}{k}.{kk}: shipped={x[kk]!r} recomputed={y.get(kk)!r}")
            else:
                out.append(f"{scope}{k}: shipped={x!r} recomputed={y!r}")

    cmp("", old, cur)
    for lang in old["hf_subset_descriptive_stats"]:
        cmp(f"[{lang}] ", old["hf_subset_descriptive_stats"][lang],
            cur["hf_subset_descriptive_stats"].get(lang, {}))
    return out


if __name__ == "__main__":
    for task_name in ("FleursT2ARetrieval.v2", "FleursA2TRetrieval.v2"):
        new, n = build(task_name)
        stats_path = STATS / f"{task_name}.json"
        print(f"\n{'=' * 74}\n{task_name}  ({n} subsets)\n{'=' * 74}")
        a = new[SPLIT]
        print(f"  queries={a['num_queries']} documents={a['num_documents']} "
              f"chars={a['number_of_characters']}")
        print(f"  relevant_docs_statistics: {json.dumps(a['relevant_docs_statistics'])}")

        if stats_path.exists():
            diffs = diff(new, stats_path)
            print(f"\n  differences vs committed stats: {len(diffs)}")
            for d in diffs[:20]:
                print(f"    {d}")

        stats_path.write_text(json.dumps(new, indent=4) + "\n")
        print(f"\n  wrote {stats_path}")
