"""Load / preprocess IMU trials, resample and window the data.
Includes CLI for converting .xlsx files to Parquet."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from scipy import interpolate
from tqdm import tqdm
import click

from .config import DATA_DIR, PARQUET_DIR, TARGET_HZ, WINDOW_SEC, STEP_SEC, LABELS

# ---------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------

def list_trials(base_dir: Path = DATA_DIR) -> Iterator[Tuple[str, str, Path]]:
    for subj_dir in sorted(base_dir.glob("sub*")):
        for activity_dir in subj_dir.iterdir():
            for fp in activity_dir.glob("*.xlsx"):
                yield subj_dir.name, activity_dir.name, fp

# ---------------------------------------------------------------------
# XLSX → parquet conversion
# ---------------------------------------------------------------------
_COL_CACHE: dict[str, str] = {}

def _clean_cols(cols: List[str]):
    if not _COL_CACHE:
        _COL_CACHE.update({c: (c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")) for c in cols})
    return [_COL_CACHE[c] for c in cols]

def xlsx_to_parquet(force: bool = False):
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    for subj, act, fp in tqdm(list(list_trials())):
        out_dir  = PARQUET_DIR / subj / act
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp   = out_dir / f"{fp.stem}.parquet"
        if out_fp.exists() and not force:
            continue
        df = pd.read_excel(fp, engine="openpyxl")
        df.columns = _clean_cols(df.columns)
        df["time"] = (df["time"] - df["time"].iloc[0]) / 1e6
        pq.write_table(pa.Table.from_pandas(df.astype(np.float32)), out_fp, compression="snappy")

# ---------------------------------------------------------------------
# Resampling & windowing
# ---------------------------------------------------------------------

def _resample(df: pd.DataFrame, hz: int = TARGET_HZ):
    t = df["time"].values
    new_t = np.linspace(0, t[-1], int(np.ceil(t[-1]*hz))+1, dtype=np.float32)
    res = {"time": new_t}
    for col in df.columns.difference(["time"]):
        res[col] = interpolate.interp1d(t, df[col], kind="linear", fill_value="extrapolate")(new_t).astype(np.float32)
    return pd.DataFrame(res)

def create_windows(df: pd.DataFrame, label: int):
    df = _resample(df)
    win = int(WINDOW_SEC*TARGET_HZ); step = int(STEP_SEC*TARGET_HZ)
    x   = df[df.columns.difference(["time"])].values
    for s in range(0, len(x)-win+1, step):
        yield x[s:s+win], label


def window_generator():
    for subj, act, fp in list_trials():
        pq_fp = PARQUET_DIR / subj / act / f"{fp.stem}.parquet"
        df = pq.read_table(pq_fp).to_pandas()
        for w, lbl in create_windows(df, LABELS[act]):
            yield subj, w, lbl

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
@click.command()
@click.option("--convert", is_flag=True, help="Convert XLSX to Parquet and exit.")
@click.option("--force",   is_flag=True, help="Overwrite existing Parquet files.")

def cli(convert, force):
    if convert:
        xlsx_to_parquet(force)
        print("✓ Conversion finished.")

if __name__ == "__main__":
    cli()