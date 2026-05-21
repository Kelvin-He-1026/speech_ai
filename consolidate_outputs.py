"""Consolidate per-sample benchmark CSVs into one combined CSV.

Walks {output_dir}/<backend>/*.csv recursively (e.g. output/pytorch/,
output/whispercpp/, output/vllm/), parses each filename as
    {model}_{precision}_{dataset}_{mode_tag}_{YYYYMMDD}[_HHMM].csv
into metadata columns (backend, model, dataset, mode, date, time) and
appends each file's per-sample rows. The `# summary` block at the top
of each input CSV is skipped — only rows after `# per-sample` are kept.
"""
import argparse
import csv
import datetime as _dt
import re
from pathlib import Path

DATASETS = ("librispeech", "tedlium", "chime6")
# Trailing _YYYYMMDD with optional _HHMM (matches both old and new filenames).
DATE_RE = re.compile(r"_(\d{8})(?:_(\d{4}))?$")


def parse_filename(path: Path, output_root: Path):
    """Return a metadata dict for a per-sample CSV, or None if the name
    doesn't match the convention."""
    stem = path.stem
    m = DATE_RE.search(stem)
    if not m:
        return None
    date = m.group(1)
    time = m.group(2) or ""
    prefix = stem[: m.start()]

    # Strip trailing _{mode_tag} (e.g. _single, _batch8, _offline, _batch4_sorted)
    if "_" not in prefix:
        return None
    head, _, mode_tag = prefix.rpartition("_")
    # Re-attach _sorted suffix if it was part of the mode_tag
    if mode_tag == "sorted":
        head2, _, real_mode = head.rpartition("_")
        if real_mode:
            mode_tag = f"{real_mode}_sorted"
            head = head2

    # head should end with one of the known datasets
    matched_dataset = None
    matched_head = None
    for ds in DATASETS:
        suffix = f"_{ds}"
        if head.endswith(suffix):
            matched_dataset = ds
            matched_head = head[: -len(suffix)]
            break
    if matched_dataset is None:
        return None

    # matched_head is "{model}_{precision}" — keep combined; model names can
    # contain underscores so a precise split isn't reliable.
    model = matched_head

    # Backend = first directory below output_root (e.g. "pytorch", "vllm").
    try:
        rel = path.relative_to(output_root)
        backend = rel.parts[0] if len(rel.parts) > 1 else "(root)"
    except ValueError:
        backend = "(unknown)"

    return {
        "backend": backend,
        "model": model,
        "dataset": matched_dataset,
        "mode": mode_tag,
        "date": date,
        "time": time,
    }


def _read_per_sample(path: Path):
    """Yield (fieldnames, dict_row) pairs from the `# per-sample` block of a
    benchmark CSV. Skips the `# summary` block entirely. The fieldnames are
    the same for every row in a single file."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        in_persample = False
        fieldnames = None
        for row in reader:
            if not row:
                continue
            cell = row[0].strip()
            if cell == "# per-sample":
                in_persample = True
                fieldnames = None
                continue
            if cell.startswith("#"):
                in_persample = False
                continue
            if not in_persample:
                continue
            if fieldnames is None:
                fieldnames = row
            else:
                yield fieldnames, dict(zip(fieldnames, row))


def _discover_columns(files):
    """Walk all files once, capturing the union of per-sample column names
    in first-seen order so we get a stable header."""
    all_cols = []
    seen = set()
    for path, _meta in files:
        for fieldnames, _row in _read_per_sample(path):
            for col in fieldnames:
                if col not in seen:
                    seen.add(col)
                    all_cols.append(col)
            break  # fieldnames are stable per file; one peek is enough
    return all_cols


def consolidate(output_dir: Path, out_path: Path) -> int:
    files = []
    for f in sorted(output_dir.rglob("*.csv")):
        if f.name.startswith("combined"):
            continue
        meta = parse_filename(f, output_dir)
        if meta:
            files.append((f, meta))

    if not files:
        print(f"no per-sample CSVs found under {output_dir} "
              f"(searched all nested subdirectories)")
        return 0

    sample_cols = _discover_columns(files)
    meta_cols = ["backend", "model", "dataset", "mode", "date", "time"]
    header = [*meta_cols, *sample_cols]

    rows_written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=header,
                                extrasaction="ignore")
        writer.writeheader()
        for path, meta in files:
            rows_in_file = 0
            for _fieldnames, row in _read_per_sample(path):
                writer.writerow({**meta, **row})
                rows_in_file += 1
                rows_written += 1
            rel = path.relative_to(output_dir)
            stamp = meta["date"] + (f"_{meta['time']}" if meta["time"] else "")
            print(f"  + {rel}  (backend={meta['backend']}, "
                  f"model={meta['model']}, dataset={meta['dataset']}, "
                  f"mode={meta['mode']}, {stamp}, rows={rows_in_file})")

    print(f"\ncombined {len(files)} files, {rows_written} rows → {out_path}")
    return rows_written


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, default=here / "output",
                   help="Root directory containing per-backend subdirs "
                        "(default: speech_ai/output). Searched recursively.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (default: <output-dir>/combined_<today>.csv)")
    args = p.parse_args()

    out = args.out or args.output_dir / f"combined_{_dt.date.today():%Y%m%d}.csv"
    consolidate(args.output_dir, out)


if __name__ == "__main__":
    main()
