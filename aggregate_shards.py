"""Combine per-shard speech_benchmark.py CSVs into one box-level report.

Each per-shard CSV is the combined "summary + per-sample" file written by
speech_benchmark._write_combined_csv. This helper:

  1. Parses every shard CSV.
  2. Sanity-checks that they describe the same (model, dataset, mode-minus-shard-tag).
  3. Computes box-level throughput (xRT, tok/s) using max(wall) across shards,
     since the shards run in parallel in wall-clock time.
  4. Computes box-level WER by micro-averaging — sum of (subs+dels+ins) over
     sum of reference words. Averaging per-shard WER would be wrong because
     shards have different word counts.
  5. Reports p50/p95 latency over the concatenated per-sample stream.

Usage (explicit paths):
    python aggregate_shards.py output/pytorch/foo_shard0of2_*.csv \\
                              output/pytorch/foo_shard1of2_*.csv

Usage (auto-lookup by model + precision; picks the latest matching shard set):
    python aggregate_shards.py --model whisper-large-v3 --precision bf16 \\
                              --mode offline --dataset librispeech
"""

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OUTPUT_ROOT = _HERE / "output"
_BACKEND_DIRS = {
    "pytorch":  _OUTPUT_ROOT / "pytorch",
    "openvino": _OUTPUT_ROOT / "openvino",
}

# Filename written by speech_benchmark._csv_path:
#   {model_slug}_{precision}_{dataset}_{mode_tag}_{YYYYMMDD_HHMM}.csv
# where mode_tag for a sharded run looks like `offline-bAll-shard0of2`.
# Anchor from the right because model_slug may contain '_' (e.g. `openai__whisper-large-v3`).
_SHARD_RE = re.compile(
    r"^(?P<model>.+)_(?P<precision>[^_]+)_(?P<dataset>[^_]+)_"
    r"(?P<mode_tag>[^_]*-shard(?P<i>\d+)of(?P<n>\d+))_"
    r"(?P<stamp>\d{8}_\d{4})\.csv$"
)


def find_shard_set(backend: str, model: str, precision: str,
                   dataset: str | None = None, mode: str | None = None,
                   stamp: str | None = None) -> list[Path]:
    """Glob the backend output dir for sharded CSVs matching model + precision
    (and optional dataset / mode / timestamp) and return one complete shard set.

    If multiple runs match, pick the latest timestamp. Errors out if the set
    is incomplete (e.g. shard1of2 missing) so the caller doesn't silently
    average half a box.
    """
    if backend not in _BACKEND_DIRS:
        raise SystemExit(f"unknown backend {backend!r}; expected one of {sorted(_BACKEND_DIRS)}")
    out_dir = _BACKEND_DIRS[backend]
    if not out_dir.is_dir():
        raise SystemExit(f"backend dir does not exist: {out_dir}")

    model_slug = model.replace("/", "__").replace(" ", "_")

    # group_key -> {shard_index: [(stamp, path), ...]} (multiple historical runs)
    # Stamp is intentionally NOT in the key: shards of the same run can finish
    # in different minutes (speech_benchmark stamps each shard independently
    # with HH:MM resolution), so grouping by stamp would split one run in two.
    groups: dict[tuple, dict[int, list[tuple[str, Path]]]] = defaultdict(lambda: defaultdict(list))
    for p in out_dir.iterdir():
        if not p.is_file() or not p.name.endswith(".csv"):
            continue
        m = _SHARD_RE.match(p.name)
        if not m:
            continue
        if m["model"] != model_slug or m["precision"] != precision:
            continue
        if dataset and m["dataset"] != dataset:
            continue
        if mode and not m["mode_tag"].startswith(mode + "-"):
            continue
        if stamp and m["stamp"] != stamp:
            continue
        n = int(m["n"])
        base_mode_tag = m["mode_tag"][:m["mode_tag"].rindex("-shard")]
        key = (m["dataset"], base_mode_tag, n)
        groups[key][int(m["i"])].append((m["stamp"], p))

    if not groups:
        filters = [f"model={model_slug}", f"precision={precision}"]
        if dataset: filters.append(f"dataset={dataset}")
        if mode:    filters.append(f"mode={mode}")
        if stamp:   filters.append(f"stamp={stamp}")
        raise SystemExit(f"no sharded CSVs in {out_dir} matching: {', '.join(filters)}")

    # Pick the group whose latest-overall shard is most recent.
    def _group_latest_stamp(g: dict[int, list[tuple[str, Path]]]) -> str:
        return max(s for entries in g.values() for s, _ in entries)
    best_key = max(groups, key=lambda k: _group_latest_stamp(groups[k]))
    by_index = groups[best_key]
    _ds, _base_mode, n = best_key

    missing = [i for i in range(n) if i not in by_index]
    if missing:
        raise SystemExit(
            f"incomplete shard set for {best_key}: have {sorted(by_index)}, missing {missing}"
        )
    # For each shard index, take the latest stamp — this is the most recent run.
    picked = {i: max(by_index[i], key=lambda sp: sp[0]) for i in range(n)}
    stamps = sorted({s for s, _ in picked.values()})
    if len(groups) > 1 or any(len(by_index[i]) > 1 for i in range(n)):
        print(f"[info] picking latest shards for {best_key[0]}/{best_key[1]} "
              f"(stamps: {', '.join(stamps)})")
    return [picked[i][1] for i in range(n)]


def parse_combined_csv(path: Path) -> tuple[dict, list[dict]]:
    """Return (summary_row, per_sample_rows) from one combined CSV."""
    with path.open() as f:
        rows = list(csv.reader(f))

    summary, per_sample = None, []
    i = 0
    while i < len(rows):
        row = rows[i]
        if not row:
            i += 1
            continue
        marker = row[0].strip().lower()
        if marker == "# summary":
            header = rows[i + 1]
            values = rows[i + 2]
            summary = dict(zip(header, values))
            i += 3
        elif marker == "# per-sample":
            header = rows[i + 1]
            i += 2
            while i < len(rows) and rows[i] and not rows[i][0].startswith("#"):
                per_sample.append(dict(zip(header, rows[i])))
                i += 1
        else:
            i += 1

    if summary is None or not per_sample:
        raise SystemExit(f"{path}: missing summary or per-sample block")
    return summary, per_sample


def _f(d: dict, k: str, default: float = 0.0) -> float:
    v = d.get(k, "")
    if v in (None, ""):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _i(d: dict, k: str, default: int = 0) -> int:
    v = d.get(k, "")
    if v in (None, ""):
        return default
    try:
        return int(float(v))
    except ValueError:
        return default


def _strip_shard_tag(mode_tag: str) -> str:
    """`offline-bAll-shard0of2` -> `offline-bAll`. Used to verify shards describe
    the same config. Handles both the new hyphen format and the legacy
    underscore format (`offline_shard0of2`)."""
    for sep in ("-shard", "_shard"):
        if sep in mode_tag:
            head, _, tail = mode_tag.rpartition(sep)
            if "of" in tail and tail.split("of")[0].isdigit() and tail.split("of")[1].isdigit():
                return head
    return mode_tag


def aggregate(paths: list[Path]) -> dict:
    parsed = [parse_combined_csv(p) for p in paths]
    summaries = [s for s, _ in parsed]
    sample_sets = [r for _, r in parsed]

    # Sanity: same model, dataset, base mode (BenchReport.mode is the un-tagged mode).
    keys = [(s["model"], s["dataset"], s["mode"]) for s in summaries]
    if len(set(keys)) > 1:
        print(f"[warn] shards describe different configs:")
        for p, k in zip(paths, keys):
            print(f"  {p}: model={k[0]} dataset={k[1]} mode={k[2]}")

    # Sanity: shard tag in filename (BenchReport.mode doesn't carry shard info).
    if not all(("-shard" in p.name) or ("_shard" in p.name) for p in paths):
        print(f"[warn] not every input filename has a 'shardXofN' tag — "
              f"are you sure these are paired shards?")

    # Counts and wall time
    total_audio = sum(_f(s, "total_audio_s") for s in summaries)
    max_wall    = max(_f(s, "total_wall_s") for s in summaries)
    total_tokens = sum(_i(s, "total_new_tokens") for s in summaries)
    n_samples   = sum(_i(s, "n_samples") for s in summaries)

    # WER: micro-average over reference words, NOT mean of per-shard WERs.
    sub = del_ = ins = ref_words = 0
    for rows in sample_sets:
        for r in rows:
            sub       += _i(r, "substitutions")
            del_      += _i(r, "deletions")
            ins       += _i(r, "insertions")
            ref_words += _i(r, "n_ref_words")
    wer = (sub + del_ + ins) / max(1, ref_words)

    # Latency percentiles over the concatenated per-sample stream.
    all_total = [_f(r, "total_seconds") for rows in sample_sets for r in rows
                 if _f(r, "total_seconds") > 0]
    all_ttft  = [_f(r, "ttft_seconds")  for rows in sample_sets for r in rows
                 if _f(r, "ttft_seconds") > 0]

    def _pct(xs, p):
        if not xs:
            return 0.0
        xs_sorted = sorted(xs)
        idx = max(0, min(len(xs_sorted) - 1, int(p * len(xs_sorted)) - 1))
        return xs_sorted[idx]

    p50_total = statistics.median(all_total) if all_total else 0.0
    p95_total = _pct(all_total, 0.95)
    p50_ttft  = statistics.median(all_ttft) if all_ttft else 0.0
    p95_ttft  = _pct(all_ttft, 0.95)

    # Effective batch size per generate() call:
    #   - mode=single  -> 1 (one sample per call)
    #   - mode=batch   -> BenchReport.batch_size (CLI --batch-size)
    #   - mode=offline -> "all" (the whole shard packed into a single call)
    base_mode = _strip_shard_tag(summaries[0].get("mode", ""))
    raw_bs    = summaries[0].get("batch_size", "")
    if base_mode == "single":
        batch_size_eff = "1"
    elif base_mode == "offline":
        batch_size_eff = "all"
    else:
        batch_size_eff = raw_bs or "?"

    return {
        "model":          summaries[0]["model"],
        "dtype":          summaries[0].get("dtype", ""),
        "dataset":        summaries[0]["dataset"],
        "base_mode":      base_mode,
        "batch_size":     batch_size_eff,
        "n_shards":       len(summaries),
        "n_samples":      n_samples,
        "total_audio_s":  total_audio,
        "max_wall_s":     max_wall,
        "box_xrt":        total_audio / max(1e-9, max_wall),
        "box_tok_per_s":  total_tokens / max(1e-9, max_wall),
        "wer":            wer,
        "ref_words":      ref_words,
        "substitutions":  sub,
        "deletions":      del_,
        "insertions":     ins,
        "p50_total_ms":   p50_total * 1000,
        "p95_total_ms":   p95_total * 1000,
        "p50_ttft_ms":    p50_ttft * 1000,
        "p95_ttft_ms":    p95_ttft * 1000,
        "per_shard_wall_s":   [_f(s, "total_wall_s") for s in summaries],
        "per_shard_xrt":      [_f(s, "throughput_xrt") for s in summaries],
        "per_shard_samples":  [_i(s, "n_samples") for s in summaries],
    }


def _print_box_report(r: dict) -> None:
    rows = [
        ("model",                 r["model"]),
        ("dtype",                 r.get("dtype", "")),
        ("dataset",               r["dataset"]),
        ("mode (pre-shard)",      r["base_mode"]),
        ("batch size",            str(r["batch_size"])),
        ("shards combined",       str(r["n_shards"])),
        ("samples (total)",       str(r["n_samples"])),
        ("audio (sum, s)",        f"{r['total_audio_s']:.1f}"),
        ("wall (max across, s)",  f"{r['max_wall_s']:.1f}"),
        ("box xRT",               f"{r['box_xrt']:.3f}"),
        ("box tok/s",             f"{r['box_tok_per_s']:.2f}"),
        ("box WER (micro-avg)",   f"{r['wer'] * 100:.2f}%"),
        ("substitutions",         str(r['substitutions'])),
        ("deletions",             str(r['deletions'])),
        ("insertions",            str(r['insertions'])),
        ("reference words",       str(r['ref_words'])),
        ("latency p50 / p95 (ms)", f"{r['p50_total_ms']:.1f} / {r['p95_total_ms']:.1f}"),
        ("TTFT p50 / p95 (ms)",    f"{r['p50_ttft_ms']:.1f} / {r['p95_ttft_ms']:.1f}"),
        ("per-shard wall (s)",     ", ".join(f"{x:.1f}" for x in r["per_shard_wall_s"])),
        ("per-shard xRT",          ", ".join(f"{x:.3f}" for x in r["per_shard_xrt"])),
        ("per-shard samples",      ", ".join(str(x) for x in r["per_shard_samples"])),
    ]
    width = max(len(k) for k, _ in rows)
    print()
    print("===== BOX-LEVEL REPORT (combined across shards) =====")
    for k, v in rows:
        print(f"  {k.ljust(width)}  {v}")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*",
                   help="Two-or-more per-shard combined CSVs. Omit and use --model/--precision "
                        "to auto-discover the latest matching shard set instead.")
    p.add_argument("--model", default=None,
                   help="Model name as passed to speech_benchmark (e.g. 'whisper-large-v3' "
                        "or 'openai/whisper-large-v3'). Used to auto-locate shard CSVs.")
    p.add_argument("--precision", default=None,
                   help="Precision tag in the filename (e.g. bf16, fp16, int8, w8a8).")
    p.add_argument("--backend", default="pytorch", choices=sorted(_BACKEND_DIRS),
                   help="Which output/{backend}/ directory to scan when auto-locating.")
    p.add_argument("--dataset", default=None,
                   help="Optional dataset filter (e.g. librispeech, tedlium).")
    p.add_argument("--mode", default=None, choices=("single", "batch", "offline"),
                   help="Optional mode filter.")
    p.add_argument("--stamp", default=None,
                   help="Optional exact timestamp filter (YYYYMMDD_HHMM). "
                        "Default: pick the latest matching set.")
    p.add_argument("--write-csv", default=None,
                   help="Optional path to also write the box-level summary as a 1-row CSV.")
    args = p.parse_args()

    if args.paths:
        paths = [Path(x) for x in args.paths]
    elif args.model and args.precision:
        paths = find_shard_set(
            backend=args.backend, model=args.model, precision=args.precision,
            dataset=args.dataset, mode=args.mode, stamp=args.stamp,
        )
        print(f"[auto] using {len(paths)} shard(s) from output/{args.backend}/:")
        for x in paths:
            print(f"  {x.name}")
    else:
        p.error("provide either positional shard paths or --model and --precision")

    missing = [x for x in paths if not x.exists()]
    if missing:
        print("Files not found:", *missing, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    if len(paths) < 2:
        print("Need at least 2 shard CSVs", file=sys.stderr)
        sys.exit(1)

    rep = aggregate(paths)
    _print_box_report(rep)

    if args.write_csv:
        out = Path(args.write_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Drop list-valued columns for the 1-row CSV.
        flat = {k: v for k, v in rep.items() if not isinstance(v, list)}
        write_header = not out.exists()
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat.keys()))
            if write_header:
                w.writeheader()
            w.writerow(flat)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
