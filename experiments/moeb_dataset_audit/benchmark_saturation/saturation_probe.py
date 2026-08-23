"""Task-level saturation probe for MTEB tasks.

Measures whether a task still separates models *at the frontier*, using only the
public MTEB result cache. A task can keep plenty of signal for weak-vs-strong
comparisons while the top of the field is compressed against the metric ceiling.

Signals reported per task:
  - dist_to_ceiling  : 1 - best_score (how much headroom is left)
  - topK_spread      : best - K-th best, in percentage points
  - within_Xpp       : how many submissions sit within X pp of the best score
  - n_at_best        : submissions tied at the exact best score
  - family-dedup     : all of the above after collapsing model families, so that
                       stacked variants (2b/4b/9b, bf16, ...) cannot fake a tight
                       frontier

Usage:
    pip install mteb
    python saturation_probe.py                                     # default demo pair
    python saturation_probe.py VidoreSyntheticDocQAAIRetrieval VidoreTatdqaRetrieval
    python saturation_probe.py --benchmark "MIEB(lite)" --top 20    # rank a whole benchmark
    python saturation_probe.py STS12 --json out.json

Notes:
  * Scores come from `mteb.ResultCache().load_results(...).to_dataframe()`, i.e.
    exactly the numbers the public leaderboard displays (one row per model, model
    revisions joined by mteb itself).
  * `dist_to_ceiling` assumes the task's main score is bounded at 1.0, which holds
    for the ndcg/ap/accuracy-style main scores MTEB uses.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

import mteb

DEFAULT_TASKS = ["VidoreSyntheticDocQAAIRetrieval", "VidoreTatdqaRetrieval"]
CEILING = 1.0

# Suffixes that mark a size/precision/generation variant of the *same* model line.
_VARIANT_SUFFIX = re.compile(
    r"[-_](?:"
    r"bf16|fp16|fp8|int8|int4|binary|"                 # quantization
    r"\d+\.?\d*b(?:-\w+)?|"                            # parameter counts: 2b, 4.5b, 9b-instruct
    r"v\d+(?:\.\d+)*|"                                 # version tags
    r"base|large|small|mini|tiny|xs|full|lora|preview|instruct|turbo"
    r")$"
)


def family_key(model_name: str) -> str:
    """Collapse `org/model-4b-bf16-v0` to a coarse family key.

    Heuristic on purpose: it only needs to stop one model line from occupying the
    whole frontier. Keeping the org prefix prevents unrelated orgs from merging.
    """
    org, _, name = model_name.rpartition("/")
    name = name.lower()
    name = re.sub(r"_\(.*?\)", "", name)  # drop "(instruct)"-style annotations
    prev = None
    while prev != name:  # peel stacked suffixes: -9b-bf16-v0 -> -9b-bf16 -> -9b -> ""
        prev = name
        name = _VARIANT_SUFFIX.sub("", name)
        name = re.sub(r"[-_]+$", "", name)
    return f"{org.lower()}/{name or model_name.rpartition('/')[2].lower()}"


def load_scores(tasks: list[str] | None, benchmark: str | None) -> pd.DataFrame:
    """One row per (model, task) from the public result cache."""
    cache = mteb.ResultCache()
    results = cache.load_results(tasks=benchmark if benchmark else tasks)
    df = results.to_dataframe(format="long")
    if df.empty:
        raise SystemExit("No results found for the requested tasks.")
    df = df.dropna(subset=["score"])
    df["family"] = df["model_name"].map(family_key)
    return df


def frontier_stats(scores: pd.Series, prefix: str = "") -> dict:
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    n = len(s)
    best = float(s[0])

    def spread(k: int) -> float | None:
        return round(float(best - s[k - 1]) * 100, 3) if n >= k else None

    def within(pp: float) -> dict:
        c = int((s >= best - pp / 100 - 1e-12).sum())
        return {"count": c, "share_pct": round(100 * c / n, 1)}

    return {
        f"{prefix}n": n,
        f"{prefix}best": round(best, 5),
        f"{prefix}dist_to_ceiling_pp": round((CEILING - best) * 100, 3),
        f"{prefix}top5_spread_pp": spread(5),
        f"{prefix}top10_spread_pp": spread(10),
        f"{prefix}within_1pp": within(1.0),
        f"{prefix}within_2pp": within(2.0),
        f"{prefix}n_at_best": int((s >= best - 1e-12).sum()),
        f"{prefix}median": round(float(np.median(s)), 5),
    }


def classify(sub: dict, fam: dict) -> str:
    """Coarse triage. The family-dedup frontier spread is the deciding signal:
    it is what survives after variant stacking is removed."""
    d2c = sub["dist_to_ceiling_pp"]
    fam_spread = fam["fam_top10_spread_pp"] or fam["fam_top5_spread_pp"] or 0.0
    if d2c <= 1.0 and (sub["top10_spread_pp"] or 0) <= 1.0:
        return "SATURATED AT FRONTIER"
    if d2c <= 1.0 or sub["within_2pp"]["share_pct"] >= 25:
        return "FRONTIER COMPRESSED"
    if fam_spread >= 2.0:
        return "HEALTHY DISCRIMINATION"
    return "MIXED / INSPECT"


def analyse(df: pd.DataFrame, task: str, top: int) -> dict:
    t = df[df["task_name"] == task].sort_values("score", ascending=False)
    sub = frontier_stats(t["score"])
    fam_scores = t.drop_duplicates("family", keep="first")["score"]
    fam = frontier_stats(fam_scores, prefix="fam_")
    return {
        "task": task,
        "classification": classify(sub, fam),
        "n_families": int(t["family"].nunique()),
        "submission_level": sub,
        "family_dedup_level": fam,
        "top": [
            {"rank": i + 1, "model": r.model_name, "score": round(r.score, 5)}
            for i, r in enumerate(t.head(top).itertuples())
        ],
    }


def print_report(reports: list[dict], top: int) -> None:
    hdr = [
        "task", "N", "best", "dist-to-ceiling", "top-10 spread",
        "top-10 spread (fam)", "within 2pp", "tied at best", "verdict",
    ]
    rows = []
    for r in reports:
        s, f = r["submission_level"], r["family_dedup_level"]
        rows.append([
            r["task"],
            str(s["n"]),
            f"{s['best']:.3f}",
            f"{s['dist_to_ceiling_pp']:.2f} pp",
            "n/a" if s["top10_spread_pp"] is None else f"{s['top10_spread_pp']:.2f} pp",
            "n/a" if f["fam_top10_spread_pp"] is None else f"{f['fam_top10_spread_pp']:.2f} pp",
            f"{s['within_2pp']['count']}/{s['n']} ({s['within_2pp']['share_pct']:.0f}%)",
            str(s["n_at_best"]),
            r["classification"],
        ])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(hdr)]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    print()
    print(line(hdr))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print(line(row))

    for r in reports:
        print(f"\n--- {r['task']} · top {min(top, r['submission_level']['n'])} "
              f"({r['n_families']} distinct families in field) ---")
        for e in r["top"]:
            print(f"  {e['rank']:>3}. {e['score']:.5f}  {e['model']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tasks", nargs="*", default=None, help="task names (default: ViDoRe demo pair)")
    p.add_argument("--benchmark", default=None, help='rank every task in a benchmark, e.g. "MIEB(lite)"')
    p.add_argument("--top", type=int, default=10, help="how many leaders to list per task")
    p.add_argument("--json", dest="json_out", default=None, help="write the full report to this path")
    args = p.parse_args()

    tasks = args.tasks or (None if args.benchmark else DEFAULT_TASKS)
    df = load_scores(tasks, args.benchmark)
    task_list = sorted(df["task_name"].unique()) if args.benchmark else [t for t in tasks if t in set(df["task_name"])]
    missing = set(tasks or []) - set(df["task_name"])
    if missing:
        print(f"warning: no results for {sorted(missing)}", file=sys.stderr)
    if not task_list:
        raise SystemExit("Nothing to analyse.")

    reports = [analyse(df, t, args.top) for t in task_list]
    # most-saturated first
    reports.sort(key=lambda r: (r["submission_level"]["dist_to_ceiling_pp"],
                                r["submission_level"]["top10_spread_pp"] or 0))
    print_report(reports, args.top)

    if args.json_out:
        payload = {"mteb_version": mteb.__version__, "ceiling": CEILING, "reports": reports}
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
