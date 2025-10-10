from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _union_columns(*dfs: Iterable[pd.DataFrame]) -> list[str]:
    """Return the sorted union of all columns across provided DataFrames."""
    cols: set[str] = set()
    for d in dfs:
        cols.update(map(str, d.columns))
    return sorted(cols)

def _pick_iclr_dir(root: Path) -> Path:
    for p in (root / "ICLR", root / "Data" / "ICLR"):
        if p.exists():
            return p
    # Fall back to root/ICLR to produce a helpful error path
    return root / "ICLR"

def get_parquet(
    project_root: str | Path = ".",
    outpath: str | Path | None = None,
    *,
    add_year: bool = True,
    dedupe_on: Optional[list[str]] = None,
) -> pd.DataFrame:
    root = Path(project_root).resolve()
    iclr_dir = _pick_iclr_dir(root)

    f2024 = iclr_dir / "iclr2024_accepted.parquet"
    f2025 = iclr_dir / "iclr2025_accepted.parquet"

    if not f2024.exists():
        raise FileNotFoundError(f"Missing file: {f2024}")
    if not f2025.exists():
        raise FileNotFoundError(f"Missing file: {f2025}")

    df24 = pd.read_parquet(f2024, engine="pyarrow")
    df25 = pd.read_parquet(f2025, engine="pyarrow")

    if add_year:
        df24 = df24.assign(year=2024)
        df25 = df25.assign(year=2025)

    all_cols = _union_columns(df24, df25)
    df24 = df24.reindex(columns=all_cols)
    df25 = df25.reindex(columns=all_cols)

    merged = pd.concat([df24, df25], ignore_index=True, sort=False)

    if dedupe_on:
        merged = merged.drop_duplicates(subset=dedupe_on, keep="first")

    if outpath is None:
        outpath = root / "Data" / "by_experiment" / "main_ICLR" / "data.parquet"

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    merged.to_parquet(outpath, index=False, engine="pyarrow")

    return merged


if __name__ == "__main__":
    import sys

    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    outpath = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    df = get_parquet(project_root, outpath)
    print(f"Merged {len(df)} rows. Written to: {outpath or (Path(project_root)/'Data'/'by_experiment'/'main_ICLR'/'data.parquet')}")
