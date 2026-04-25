""" minimal pandas utilities needed by Conditional-Skew """
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

def read_csv_date_index(infile, date_col=0, date_min=None, date_max=None, nrows=None,
    ncol=None, print_fl: bool = False, columns: Iterable[str] | None = None,
    exclude_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Read a CSV file, parse one column as dates, and return a DataFrame indexed by date.
    Only the options used by this project are implemented here.
    """
    full_file_name = DATA_DIR / infile
    df = pd.read_csv(full_file_name, nrows=nrows)
    df.iloc[:, date_col] = pd.to_datetime(df.iloc[:, date_col]).dt.date
    df = df.set_index(df.columns.values[date_col])
    if df.index.isnull().any():
        raise ValueError(f"Error: The file {infile} contains missing index values.")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Error: The index in {infile} is not sorted ascending.")
    if date_min or date_max:
        df = df[date_min:date_max]
    if ncol and ncol < df.shape[1]:
        df = df.iloc[:, :ncol]
    if columns is not None:
        df = df[[col for col in columns]]
    if exclude_columns is not None:
        df = df[[col for col in df.columns if col not in exclude_columns]]
    df.index.rename("Date", inplace=True)
    if print_fl:
        print_first_last(df)
    return df

def print_first_last(df: pd.DataFrame, title=None, print_index=True, trailer=None,
    transpose=False, end=None) -> None:
    """Print shape and first and last rows of a DataFrame or Series."""
    if title is not None:
        if title == "":
            print(end="\n")
        else:
            print(title)
    if isinstance(df, pd.Series):
        print("#obs =", len(df))
        if len(df) > 0:
            print(df.iloc[[0, -1]].to_string(index=print_index))
        if trailer:
            print(trailer, end="")
        if end:
            print(end=end)
        return
    print("#sym, #obs =", df.shape[1], df.shape[0])
    if len(df) > 0:
        df_fl = df.iloc[[0, -1], :]
        if transpose:
            df_fl = df_fl.T
        print(df_fl.to_string(index=print_index))
    if trailer:
        print(trailer, end="")
    if end:
        print(end=end)
