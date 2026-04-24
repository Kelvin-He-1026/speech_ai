import argparse
import csv
import datetime as _dt
import re
from pathlib import Path

DATASETS = ("librispeech", "tedlium", "chime6")
DATE_RE = re.compile(r"_(\d{8})$")


def parse_filename(path: Path):
    """Return (model, dataset, date) for a per-sample file, or None."""
    stem = path.stem
    m = DATE_RE.search(stem)
    if not m:
        return None
    date = m.group(1)
    prefix = stem[: m.start()]
    for ds in DATASETS:
        suffix = f"_{ds}"
        if prefix.endswith(suffix):
            return prefix[: -len(suffix)], ds, date
    return None


def consolidate(output_dir: Path, out_path: Path) -> int:
    files = []
    for f in sorted(output_dir.glob("*.csv")):
        if f.name.startswith("combined"):
            continue
        meta = parse_filename(f)
        if meta:
            files.append((f, meta))

    if not files:
        print(f"no per-sample CSVs found in {output_dir}")
        return 0

    all_cols: list[str] = []
    seen = set()
    for path, _ in files:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for col in reader.fieldnames or []:
                if col not in seen:
                    seen.add(col)
                    all_cols.append(col)

    header = ["model", "dataset", "date", *all_cols]

    rows_written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=header)
        writer.writeheader()
        for path, (model, dataset, date) in files:
            with open(path, newline="") as in_fh:
                reader = csv.DictReader(in_fh)
                for row in reader:
                    writer.writerow({"model": model, "dataset": dataset, "date": date, **row})
                    rows_written += 1
            print(f"  + {path.name}  (model={model}, dataset={dataset}, date={date})")

    print(f"\ncombined {len(files)} files, {rows_written} rows → {out_path}")
    return rows_written


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Consolidate per-sample benchmark CSVs into one combined CSV."
    )
    p.add_argument("--output-dir", type=Path, default=here / "output",
                   help="Directory containing per-sample CSVs (default: speech_ai/output)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (default: <output-dir>/combined_<today>.csv)")
    args = p.parse_args()

    out = args.out or args.output_dir / f"combined_{_dt.date.today():%Y%m%d}.csv"
    consolidate(args.output_dir, out)


if __name__ == "__main__":
    main()
