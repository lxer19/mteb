# ruff: noqa: T201
"""Audit whether classification main scores are reliable under MTEB's protocol.

This is not an imbalance checker. It reads committed descriptive statistics and asks:
R1 whether accuracy can be beaten by a constant predictor, R2 whether macro scores are
too noisy, and R3 whether AP scores are comparable across prevalences.

No dataset is downloaded and no statistics are regenerated. ``--with-results`` adds
the observed best-model minus trivial-baseline gap from public results.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import warnings
from collections import Counter
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

import mteb
from mteb.benchmarks import get_benchmark

Level = Literal["HIGH", "REVIEW", "NOTE"]
REPO_ROOT = Path(__file__).resolve().parents[3]
STATS_ROOT = REPO_ROOT / "mteb" / "descriptive_stats"
DEFAULT_OUTPUT = Path(__file__).with_name("metric_reliability_audit.json")
LEVEL_ORDER = {"HIGH": 0, "REVIEW": 1, "NOTE": 2}
ACCURACY_METRICS = {"accuracy", "max_accuracy"}
AP_METRICS = {"ap", "max_ap"}

# R1 is empirically calibrated. R2 and R3 are provisional.
R1_HIGH, R1_REVIEW, R1_NOTE = 0.90, 0.70, 0.50
R2_NEFF_HIGH, R2_NEFF_REVIEW = 50.0, 200.0
R3_PREVALENCE_HIGH, R3_PREVALENCE_REVIEW = 0.90, 0.75
MIN_PUBLIC_MODELS = 5


def distribution_stats(labels: dict[str, Any]) -> dict[str, Any]:
    """Compute the five core statistics and positive prevalence from label counts."""
    parsed = {
        str(label): value["count"] if isinstance(value, dict) else value
        for label, value in labels.items()
    }
    counts = [
        count for count in parsed.values() if isinstance(count, int) and count > 0
    ]
    if len(counts) < 2:
        raise ValueError("a label distribution needs at least two non-empty classes")
    n, k = sum(counts), len(counts)
    majority, minority = max(counts), min(counts)
    inverse_support = sum(1.0 / count for count in counts)
    positive = parsed.get("1")
    return {
        "n": n,
        "k": k,
        "majority_share": majority / n,
        "accuracy_budget": n - majority,
        "n_eff_macro": (k * k) / inverse_support,
        "macro_noise_floor": (0.5 / k) * math.sqrt(inverse_support),
        "minority_count": minority,
        "positive_prevalence": positive / n
        if isinstance(positive, int) and positive > 0
        else None,
    }


def _tier(
    value: float, high: float, review: float, note: float | None = None
) -> Level | None:
    if value >= high:
        return "HIGH"
    if value >= review:
        return "REVIEW"
    if note is not None and value >= note:
        return "NOTE"
    return None


def assess_risks(stats: dict[str, Any], main_score: str) -> list[dict[str, Any]]:
    """Return independent diagnoses; never collapse them into a scalar score."""
    risks: list[dict[str, Any]] = []
    majority = stats["majority_share"]
    if main_score in ACCURACY_METRICS:
        level = _tier(majority, R1_HIGH, R1_REVIEW, R1_NOTE)
        if level:
            risks.append(
                {
                    "axis": "R1",
                    "level": level,
                    "message": (
                        f"majority baseline {majority:.4f}; {stats['accuracy_budget']} "
                        "non-majority examples carry the entire accuracy budget"
                    ),
                }
            )

    n_eff, minority = stats["n_eff_macro"], stats["minority_count"]
    if n_eff < R2_NEFF_HIGH or minority == 1:
        level = "HIGH"
    elif n_eff < R2_NEFF_REVIEW:
        level = "REVIEW"
    else:
        level = None
    if level:
        risks.append(
            {
                "axis": "R2",
                "level": level,
                "message": (
                    f"effective N for macro metrics {n_eff:.1f} of {stats['n']}; "
                    f"minority count {minority}; macro noise floor "
                    f"~{stats['macro_noise_floor']:.4f}"
                ),
            }
        )

    if main_score in AP_METRICS and stats["positive_prevalence"] is not None:
        prevalence = stats["positive_prevalence"]
        level = _tier(prevalence, R3_PREVALENCE_HIGH, R3_PREVALENCE_REVIEW)
        if level:
            risks.append(
                {
                    "axis": "R3",
                    "level": level,
                    "message": (
                        f"positive prevalence {prevalence:.4f}; a random ranker's expected "
                        "AP is at that level, so cross-task means are not comparable"
                    ),
                }
            )
    return risks


def _label_distributions(
    split_stats: dict[str, Any],
) -> Iterable[tuple[str | None, dict[str, Any]]]:
    """Yield direct and per-HF-subset label dictionaries."""
    for key, value in split_stats.items():
        if key == "hf_subset_descriptive_stats" or not isinstance(value, dict):
            continue
        labels = value.get("labels")
        if isinstance(labels, dict) and labels:
            yield None, labels
    subsets = split_stats.get("hf_subset_descriptive_stats", {})
    if isinstance(subsets, dict):
        for subset, subset_stats in subsets.items():
            if isinstance(subset_stats, dict):
                for _, labels in _label_distributions(subset_stats):
                    yield str(subset), labels


def _task_distribution(split_stats: dict[str, Any]) -> dict[str, Any] | None:
    """Return the aggregate distribution, combining subsets only as a fallback."""
    direct = [
        labels for subset, labels in _label_distributions(split_stats) if subset is None
    ]
    if direct:
        return direct[0]

    combined: Counter[str] = Counter()
    for subset, labels in _label_distributions(split_stats):
        if subset is None:
            continue
        for label, value in labels.items():
            count = value.get("count") if isinstance(value, dict) else value
            if isinstance(count, int) and count > 0:
                combined[str(label)] += count
    return dict(combined) or None


def collect_audits(task_names: set[str] | None = None) -> list[dict[str, Any]]:
    """Collect every scored split/subset represented by committed label statistics."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tasks = {
                task.metadata.name: task
                for task in mteb.get_tasks(
                    exclude_superseded=False, exclude_aggregate=True, exclude_beta=False
                )
            }
    finally:
        logging.disable(previous)

    audits = []
    for path in sorted(STATS_ROOT.rglob("*.json")):
        task = tasks.get(path.stem)
        if (
            task is None
            or "Classification" not in task.metadata.type
            or "MultilabelClassification" in task.metadata.type
            or (task_names is not None and task.metadata.name not in task_names)
        ):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for split in task.metadata.eval_splits or []:
            split_stats = payload.get(split)
            if not isinstance(split_stats, dict):
                continue
            labels = _task_distribution(split_stats)
            if labels is None:
                continue
            try:
                stats = distribution_stats(labels)
            except ValueError:
                continue
            audits.append(
                {
                    "task": task.metadata.name,
                    "split": split,
                    "task_type": task.metadata.type,
                    "main_score": task.metadata.main_score,
                    "distribution": stats,
                    "evidence": None,
                    "risks": assess_risks(stats, task.metadata.main_score),
                }
            )
            break  # risk calibration and leaderboard evidence are task-level
    return audits


def load_evidence(task_names: list[str]) -> dict[str, dict[str, Any]]:
    """Load public main scores once, retaining enough information to audit the gap."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = (
                mteb.ResultCache()
                .load_results(tasks=sorted(set(task_names)), only_main_score=True)
                .to_dataframe()
            )
    finally:
        logging.disable(previous)
    evidence = {}
    for _, row in frame.iterrows():
        scores = []
        for column, value in row.items():
            if column in {"task_name", "is_public"}:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                scores.append(number)
        if len(scores) < MIN_PUBLIC_MODELS:
            continue
        scores.sort(reverse=True)
        evidence[str(row["task_name"])] = {
            "best": scores[0],
            "median": median(scores),
            "n_models": len(scores),
            "top10_spread": scores[0] - scores[min(9, len(scores) - 1)],
        }
    return evidence


def attach_evidence(
    audits: list[dict[str, Any]], evidence: dict[str, dict[str, Any]]
) -> None:
    """Attach empirical scores without changing any distribution-derived tier."""
    for audit in audits:
        item = evidence.get(audit["task"])
        if item is not None:
            audit["evidence"] = dict(item)
            if audit["main_score"] in ACCURACY_METRICS:
                audit["evidence"]["majority_gap"] = (
                    item["best"] - audit["distribution"]["majority_share"]
                )


def _print_summary(audits: list[dict[str, Any]], minimum: Level) -> None:
    counts = Counter(
        (risk["axis"], risk["level"]) for audit in audits for risk in audit["risks"]
    )
    print(f"Audited {len(audits)} scored task/split/subset distributions")
    print("\naxis      HIGH  REVIEW  NOTE")
    for axis in ("R1", "R2", "R3"):
        print(
            f"{axis:<8}{counts[axis, 'HIGH']:>6}{counts[axis, 'REVIEW']:>8}{counts[axis, 'NOTE']:>6}"
        )

    cutoff = LEVEL_ORDER[minimum]
    flagged = []
    for audit in audits:
        risks = [
            risk for risk in audit["risks"] if LEVEL_ORDER[risk["level"]] <= cutoff
        ]
        if risks:
            flagged.append((audit, risks))
    flagged.sort(
        key=lambda item: (
            min(LEVEL_ORDER[risk["level"]] for risk in item[1]),
            -item[0]["distribution"]["majority_share"],
            item[0]["task"],
        )
    )
    print(f"\n{minimum} or above ({len(flagged)} rows)")
    for audit, risks in flagged:
        print(f"\n{audit['task']} [{audit['split']}] main_score={audit['main_score']}")
        for risk in risks:
            print(f"  {risk['axis']} {risk['level']}: {risk['message']}")
        if audit["evidence"] and "majority_gap" in audit["evidence"]:
            ev = audit["evidence"]
            print(
                f"  empirical: best={ev['best']:.4f}, majority gap={ev['majority_gap']:+.4f}, models={ev['n_models']}"
            )


def main() -> None:
    """Run the offline audit and write its machine-readable artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-results", action="store_true")
    parser.add_argument(
        "--benchmark",
        action="append",
        help=(
            "restrict the audit to a registered benchmark; repeat to audit a union "
            '(e.g. --benchmark "MVEB(video)" --benchmark "MVEB(text, video)")'
        ),
    )
    parser.add_argument("--level", choices=tuple(LEVEL_ORDER), default="HIGH")
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    task_names = None
    if args.benchmark:
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                task_names = {
                    task.metadata.name
                    for benchmark in args.benchmark
                    for task in get_benchmark(benchmark).tasks
                }
        finally:
            logging.disable(previous)
    audits = collect_audits(task_names)
    if args.with_results:
        attach_evidence(audits, load_evidence([audit["task"] for audit in audits]))
    output = {
        "meta": {
            "n_rows": len(audits),
            "mteb_version": mteb.__version__,
            "benchmark": args.benchmark,
            "leaderboard": args.with_results,
            "thresholds": {
                "R1": {"high": R1_HIGH, "review": R1_REVIEW, "note": R1_NOTE},
                "R2": {
                    "n_eff_high": R2_NEFF_HIGH,
                    "n_eff_review": R2_NEFF_REVIEW,
                    "singleton_high": True,
                },
                "R3": {
                    "prevalence_high": R3_PREVALENCE_HIGH,
                    "prevalence_review": R3_PREVALENCE_REVIEW,
                },
            },
        },
        "audits": audits,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    _print_summary(audits, args.level)
    print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
