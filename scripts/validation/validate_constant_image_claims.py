#!/usr/bin/env python
"""Reproduce and check every claim in the `constant_images` issue.

Run this to verify the numbers rather than trust them. Each claim is declared up
front as an expected value and checked against a fresh scan of the local data, so
the script fails loudly if a figure is wrong or a dataset has changed.

It uses the production helpers (`is_constant_image`,
`count_queries_with_all_gold_constant`) so what is validated is the shipped
behaviour, not a re-implementation of it.

Data is read straight off the local HuggingFace snapshot with pyarrow. Nothing is
downloaded; a dataset that is not cached is reported as SKIPPED and lowers the
coverage line rather than failing the run.

WIT ships a raw per-language pool, so its corpus, queries and qrels are rebuilt
here the way `wit_t2i_retrieval.py` builds them. That reconstruction is a bypass
of the canonical loader, so `--check-loader` diffs it against
`WITT2IRetrieval.load_data()` — corpus ids, query ids and qrels values, per
subset. It is opt-in only because the loader resolves dataset configs over the
network; the default run stays offline.

    python experiments/media_duplicate_audit/validate_constant_image_claims.py
    python experiments/media_duplicate_audit/validate_constant_image_claims.py --check-loader

Exit code 0 if every reachable claim holds, 1 otherwise.
"""

from __future__ import annotations

import io
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from mteb.abstasks._statistics_calculation import (
    count_queries_with_all_gold_constant,
    is_constant_image,
)

HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))

# A solid-colour compressed image is tiny. This bound is deliberately loose, and
# format-aware so nothing uncompressed is ever skipped. Its soundness is checked
# at the end by decoding a random sample of the images it rejected.
COMPRESSED = {"JPEG", "PNG", "WEBP", "GIF", "JPEG2000", "MPO"}
VALIDATION_SAMPLE = 500

# ---------------------------------------------------------------- claims
# (task, repo, side, expected constant images, expected broken queries,
#  expected evaluated queries, expected ceiling)
CLAIMS = [
    ("VisualNewsI2TRetrieval", "mteb/mbeir_visualnews_task3", "queries", 14, 14, 20000, 0.999300),
    ("VisualNewsT2IRetrieval", "mteb/mbeir_visualnews_task0", "corpus", 174, 14, 19995, 0.999300),
    ("WebQAT2ITRetrieval", "mteb/mbeir_webqa_task2", "corpus", 187, 0, None, None),
    ("OVENIT2ITRetrieval", "mteb/mbeir_oven_task8", "corpus", 420, 0, None, None),
    ("RP2kI2IRetrieval", "mteb/rp2k", "corpus", 1, 0, None, None),
]
# WIT ships a raw per-language pool; its loader builds corpus/queries/qrels per
# language and MTEB scores each language as its own hf_subset.
WIT_EXPECTED_CONSTANT = 21
WIT_EXPECTED_BROKEN = 21
WIT_EXPECTED_MEAN_CEILING = 0.997757
WIT_EXPECTED_WORST = ("id", 0.995560)
WIT_EXPECTED_DISTINCT_SOURCES = 13


def candidate(nbytes: int, pixels: int, fmt: str | None) -> bool:
    if fmt not in COMPRESSED:
        return True
    return nbytes <= 1500 + 0.03 * pixels


def snapshot(repo: str) -> Path | None:
    root = HUB / f"datasets--{repo.replace('/', '--')}" / "snapshots"
    snaps = sorted(root.glob("*")) if root.exists() else []
    return snaps[0] if snaps else None


def side_files(snap: Path, side: str) -> list[Path]:
    stems = {
        "corpus": ["corpus.parquet", "corpus-*.parquet", "corpus/*.parquet"],
        "queries": ["query.parquet", "query-*.parquet", "query/*.parquet",
                    "queries.parquet", "queries-*.parquet", "queries/*.parquet"],
        "qrels": ["qrels.parquet", "qrels-*.parquet", "qrels/*.parquet"],
    }[side]
    out: list[Path] = []
    for stem in stems:
        out += sorted(snap.glob(stem)) + sorted(snap.glob(f"*/{stem}"))
    return sorted(set(out))


def read_qrels(snap: Path) -> dict[str, dict[str, int]]:
    q: dict[str, dict[str, int]] = defaultdict(dict)
    for f in side_files(snap, "qrels"):
        d = pq.read_table(f, columns=["query-id", "corpus-id", "score"]).to_pydict()
        for qid, did, score in zip(d["query-id"], d["corpus-id"], d["score"]):
            q[str(qid)][str(did)] = int(score)
    return q


class Rejects:
    """Reservoir of images the prefilter skipped, so it can be audited afterwards."""

    def __init__(self, rng: random.Random) -> None:
        self.rng, self.seen, self.kept = rng, 0, []

    def offer(self, raw: bytes) -> None:
        self.seen += 1
        if len(self.kept) < VALIDATION_SAMPLE:
            self.kept.append(raw)
        else:
            j = self.rng.randrange(self.seen)
            if j < VALIDATION_SAMPLE:
                self.kept[j] = raw

    def missed(self) -> int:
        n = 0
        for raw in self.kept:
            try:
                im = Image.open(io.BytesIO(raw))
                im.load()
                if is_constant_image(im):
                    n += 1
            except Exception:
                pass
        return n


def scan_constants(files: list[Path], rejects: Rejects) -> tuple[set[str], int]:
    """Return the ids of constant images and the number of images seen."""
    found: set[str] = set()
    total = 0
    for f in files:
        pfile = pq.ParquetFile(f)
        names = set(pfile.schema_arrow.names)
        if "image" not in names:
            continue
        id_col = next((c for c in ("id", "corpus-id", "query-id") if c in names), None)
        cols = ["image"] + ([id_col] if id_col else [])
        for batch in pfile.iter_batches(batch_size=256, columns=cols):
            d = batch.to_pydict()
            ids = d.get(id_col) if id_col else None
            for i, cell in enumerate(d["image"]):
                raw = cell if isinstance(cell, bytes) else (cell or {}).get("bytes")
                if raw is None:
                    continue
                total += 1
                item_id = str(ids[i]) if ids is not None else str(total - 1)
                probe = Image.open(io.BytesIO(raw))
                if not candidate(len(raw), probe.width * probe.height, probe.format):
                    rejects.offer(raw)
                    continue
                im = Image.open(io.BytesIO(raw))
                im.load()
                if is_constant_image(im):
                    found.add(item_id)
    return found, total


CHECK_LOADER = "--check-loader" in sys.argv

results: list[tuple[str, bool, str]] = []
skipped: list[str] = []
rng = random.Random(0)
rejects = Rejects(rng)


def check(label: str, got, want, fmt: str = "{}") -> bool:
    ok = got == want
    results.append((label, ok, f"got {fmt.format(got)}, want {fmt.format(want)}"))
    return ok


def close(label: str, got: float, want: float, tol: float = 5e-6) -> bool:
    ok = abs(got - want) <= tol
    results.append((label, ok, f"got {got:.6%}, want {want:.6%}"))
    return ok


print("=" * 78)
print("Validating the constant_images claims against local data")
print("=" * 78)

# ---------------------------------------------------------------- retrieval tasks
print(f"\n{'task':<26}{'side':<9}{'images':>10}{'constant':>10}{'broken':>9}{'ceiling':>12}")
for task, repo, side, exp_const, exp_broken, exp_nq, exp_ceiling in CLAIMS:
    snap = snapshot(repo)
    if snap is None or not side_files(snap, side):
        skipped.append(f"{task} ({repo} not cached)")
        print(f"{task:<26}{side:<9}{'SKIPPED - not cached':>41}")
        continue

    qrels = read_qrels(snap)
    evaluated = set(qrels)
    const_ids, total = scan_constants(side_files(snap, side), rejects)

    if side == "corpus":
        broken = count_queries_with_all_gold_constant(qrels, const_ids)
    else:
        # a constant query is broken outright, but only if it is actually evaluated
        broken = len(const_ids & evaluated)
        check(f"{task}: every constant query is in qrels",
              len(const_ids - evaluated), 0)

    ceiling = 1 - broken / len(evaluated) if evaluated else None
    print(f"{task:<26}{side:<9}{total:>10,}{len(const_ids):>10}{broken:>9}"
          f"{ceiling:>11.4%}" if ceiling is not None else "")

    check(f"{task}: constant images", len(const_ids), exp_const)
    check(f"{task}: broken queries", broken, exp_broken)
    if exp_nq is not None:
        check(f"{task}: evaluated queries", len(evaluated), exp_nq, "{:,}")
    if exp_ceiling is not None:
        close(f"{task}: ceiling", ceiling, exp_ceiling)

# ---------------------------------------------------------------- WIT, per subset
snap = snapshot("mteb/wit")
if snap is None:
    skipped.append("WITT2IRetrieval (mteb/wit not cached)")
    print("\nWITT2IRetrieval  SKIPPED - not cached")
else:
    print("\nWITT2IRetrieval - scored per language subset")
    print(f"  {'lang':<6}{'images':>8}{'queries':>9}{'constant':>10}{'broken':>8}{'ceiling':>12}")
    per_lang, sources = [], set()
    for f in sorted(snap.glob("data/*.parquet")):
        lang = f.name.split("-")[0]
        t = pq.read_table(f, columns=["image_id", "image", "captions"]).to_pydict()
        # exactly how wit_t2i_retrieval.py builds ids and qrels
        qrels, const, n_img = {}, set(), 0
        for image_id, cell, captions in zip(t["image_id"], t["image"], t["captions"]):
            raw = cell if isinstance(cell, bytes) else (cell or {}).get("bytes")
            if raw is None:
                continue
            n_img += 1
            doc_id = "corpus-" + image_id
            for idx in range(len(captions)):
                qrels[f"query-{image_id}-{idx}"] = {doc_id: 1}
            probe = Image.open(io.BytesIO(raw))
            if not candidate(len(raw), probe.width * probe.height, probe.format):
                rejects.offer(raw)
                continue
            im = Image.open(io.BytesIO(raw))
            im.load()
            if is_constant_image(im):
                const.add(doc_id)
                sources.add(image_id)
        broken = count_queries_with_all_gold_constant(qrels, const)
        per_lang.append((lang, n_img, len(qrels), len(const), broken,
                         1 - broken / len(qrels)))

    for lang, n_img, nq, c, b, ceil in sorted(per_lang, key=lambda r: r[5]):
        print(f"  {lang:<6}{n_img:>8,}{nq:>9,}{c:>10}{b:>8}{ceil:>11.4%}")
    mean_ceiling = sum(r[5] for r in per_lang) / len(per_lang)
    worst = min(per_lang, key=lambda r: r[5])
    print(f"  {'mean':<6}{sum(r[1] for r in per_lang):>8,}{sum(r[2] for r in per_lang):>9,}"
          f"{sum(r[3] for r in per_lang):>10}{sum(r[4] for r in per_lang):>8}"
          f"{mean_ceiling:>11.4%}")

    check("WIT: constant images", sum(r[3] for r in per_lang), WIT_EXPECTED_CONSTANT)
    check("WIT: broken queries", sum(r[4] for r in per_lang), WIT_EXPECTED_BROKEN)
    close("WIT: mean subset ceiling", mean_ceiling, WIT_EXPECTED_MEAN_CEILING)
    check("WIT: worst subset", worst[0], WIT_EXPECTED_WORST[0])
    close("WIT: worst subset ceiling", worst[5], WIT_EXPECTED_WORST[1])
    check("WIT: distinct source images behind the broken queries",
          len(sources), WIT_EXPECTED_DISTINCT_SOURCES)

# ------------------------------------------------- WIT reconstruction vs. loader
if CHECK_LOADER and snapshot("mteb/wit") is not None:
    import mteb

    wit = mteb.get_task("WITT2IRetrieval")
    wit.load_data()
    mismatched = []
    for f in sorted(snapshot("mteb/wit").glob("data/*.parquet")):
        lang = f.name.split("-")[0]
        t = pq.read_table(f, columns=["image_id", "captions"]).to_pydict()
        my_corpus, my_qrels = set(), {}
        for image_id, captions in zip(t["image_id"], t["captions"]):
            doc_id = "corpus-" + image_id
            my_corpus.add(doc_id)
            for idx in range(len(captions)):
                my_qrels[f"query-{image_id}-{idx}"] = {doc_id: 1}
        official = {k: dict(v) for k, v in wit.relevant_docs[lang]["test"].items()}
        if (set(wit.corpus[lang]["test"]["id"]) != my_corpus
                or set(wit.queries[lang]["test"]["id"]) != set(my_qrels)
                or official != my_qrels):
            mismatched.append(lang)
    check("WIT reconstruction matches the official loader (11 subsets)", mismatched, [])
elif CHECK_LOADER:
    skipped.append("WIT loader diff (mteb/wit not cached)")

# ---------------------------------------------------------------- prefilter audit
missed = rejects.missed()
check(f"prefilter soundness ({len(rejects.kept)} of {rejects.seen:,} rejects re-decoded)",
      missed, 0)

# ---------------------------------------------------------------- report
print("\n" + "=" * 78)
failed = [r for r in results if not r[1]]
for label, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<58} {detail}")
if skipped:
    print(f"\n  {len(skipped)} dataset(s) skipped, claims unverified:")
    for s in skipped:
        print(f"    - {s}")
print("=" * 78)
print(f"{len(results) - len(failed)}/{len(results)} checks passed"
      + (f", {len(skipped)} dataset(s) skipped" if skipped else ""))
sys.exit(1 if failed else 0)
