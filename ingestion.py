"""CSV and Excel ingestion helpers for Reformulation Assurance v0.5."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd


@dataclass(frozen=True)
class WorkbookPreview:
    filename: str
    sheets: list[str]


def _read_bytes(source: bytes | bytearray | BinaryIO) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    data = source.read()
    try:
        source.seek(0)
    except (AttributeError, OSError):
        pass
    return data


def workbook_preview(source: bytes | bytearray | BinaryIO, filename: str) -> WorkbookPreview:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return WorkbookPreview(filename=filename, sheets=["CSV"])
    if suffix not in {".xlsx", ".xlsm"}:
        raise ValueError("supported file types are CSV, XLSX, and XLSM")
    data = _read_bytes(source)
    workbook = pd.ExcelFile(BytesIO(data), engine="openpyxl")
    return WorkbookPreview(filename=filename, sheets=list(workbook.sheet_names))


def load_table(
    source: bytes | bytearray | BinaryIO,
    filename: str,
    *,
    sheet_name: str | int | None = None,
) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    data = _read_bytes(source)
    if suffix == ".csv":
        frame = pd.read_csv(BytesIO(data))
    elif suffix in {".xlsx", ".xlsm"}:
        selected = 0 if sheet_name in {None, "CSV"} else sheet_name
        frame = pd.read_excel(BytesIO(data), sheet_name=selected, engine="openpyxl")
    else:
        raise ValueError("supported file types are CSV, XLSX, and XLSM")
    if frame.empty:
        raise ValueError("the selected table contains no rows")
    frame.columns = [str(column).strip() for column in frame.columns]
    if any(not column for column in frame.columns):
        raise ValueError("all columns must have a non-empty header")
    if len(set(frame.columns)) != len(frame.columns):
        duplicates = sorted({c for c in frame.columns if list(frame.columns).count(c) > 1})
        raise ValueError(f"duplicate column names are not supported: {', '.join(duplicates)}")
    return frame


def import_readiness_report(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(data)
    for column in data.columns:
        series = data[column]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_count = int(numeric.notna().sum())
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "rows": total,
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "numeric_fraction": numeric_count / max(total, 1),
            }
        )
    return pd.DataFrame(rows)
