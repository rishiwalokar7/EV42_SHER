"""
Simple CSV preprocessing utility.
Usage: python preprocess_csv.py path/to/file.csv [--out out.csv]

What it does:
- Detects common delimiters and loads with pandas
- Strips whitespace from string columns
- Drops exact duplicate rows
- Infers and converts numeric/date columns where possible
- Fills missing values: numeric->median, categorical->mode, boolean stays as-is
- Saves cleaned CSV with "_cleaned.csv" suffix (or --out path)
- Writes a small JSON report with column summary: original/dropped/missing counts/types

This script is intentionally conservative (non-destructive) and creates backups.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, Any

import pandas as pd

COMMON_DELIMS = [',', '\t', ';', '|']


def detect_delimiter(path: str) -> str:
    with open(path, 'rb') as fh:
        start = fh.read(4096)
    # try decoding as utf-8, fallback latin-1
    try:
        text = start.decode('utf-8')
    except Exception:
        try:
            text = start.decode('latin-1')
        except Exception:
            return ','
    scores = {}
    for d in COMMON_DELIMS:
        scores[d] = text.count(d)
    # choose delimiter with highest count
    best = max(scores, key=scores.get)
    return best


def summarize_df(df: pd.DataFrame) -> Dict[str, Any]:
    s = {}
    s['n_rows'] = len(df)
    s['n_cols'] = len(df.columns)
    s['columns'] = {}
    for c in df.columns:
        col = df[c]
        s['columns'][c] = {
            'dtype': str(col.dtype),
            'n_missing': int(col.isna().sum()),
            'n_unique': int(col.nunique(dropna=True))
        }
    return s


def preprocess(path: str, out: str | None = None, sample_rows: int | None = None) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    delim = detect_delimiter(path)
    read_kwargs = {'sep': delim}
    if sample_rows is not None:
        df = pd.read_csv(path, nrows=sample_rows, **read_kwargs)
        # if sample, re-read full later
        df_full = pd.read_csv(path, **read_kwargs)
        df = df_full
    else:
        df = pd.read_csv(path, **read_kwargs)

    report = {'input_path': path, 'detected_delimiter': delim}
    report['before'] = summarize_df(df)

    # Drop exact duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    report['dropped_duplicates'] = n_before - len(df)

    # Strip whitespace from object columns
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in obj_cols:
        # coerce to str where not null, then strip
        df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.strip())

    # Try to convert columns to numeric where possible
    converted_numeric = []
    for c in df.columns:
        if df[c].dtype == 'object':
            # try numeric
            coerced = pd.to_numeric(df[c].dropna().str.replace(',', ''), errors='coerce')
            # if a large portion converts to numeric, accept the conversion
            if len(coerced) > 0 and coerced.notna().sum() / len(coerced) > 0.9:
                df[c] = pd.to_numeric(df[c].str.replace(',', ''), errors='coerce')
                converted_numeric.append(c)

    report['converted_numeric'] = converted_numeric

    # Try to parse dates for object columns with date-like content
    converted_dates = []
    for c in df.columns:
        if df[c].dtype == 'object':
            try:
                parsed = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
                if parsed.notna().sum() / max(1, len(parsed.dropna())) > 0.5:
                    df[c] = parsed
                    converted_dates.append(c)
            except Exception:
                pass
    report['converted_dates'] = converted_dates

    # Fill missing values: numeric->median, object/category->mode
    imputed = {}
    for c in df.columns:
        if df[c].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[c].dtype):
            median = df[c].median()
            df[c] = df[c].fillna(median)
            imputed[c] = {'strategy': 'median', 'value': median}
        else:
            mode = None
            try:
                mode = df[c].mode(dropna=True)
                if not mode.empty:
                    mode = mode.iloc[0]
            except Exception:
                mode = None
            if mode is not None:
                df[c] = df[c].fillna(mode)
                imputed[c] = {'strategy': 'mode', 'value': mode}
            else:
                # leave as-is if cannot impute
                imputed[c] = {'strategy': 'leave', 'value': None}

    report['imputed'] = imputed

    report['after'] = summarize_df(df)

    # Output
    if out is None:
        base, ext = os.path.splitext(path)
        out = base + '_cleaned.csv'
    report_path = os.path.splitext(out)[0] + '_report.json'
    df.to_csv(out, index=False)
    with open(report_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, default=str)
    report['output_path'] = out
    report['report_path'] = report_path
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a CSV file (clean, impute, convert).')
    parser.add_argument('csv', help='Path to CSV file to preprocess')
    parser.add_argument('--out', '-o', help='Output cleaned CSV path (optional)')
    parser.add_argument('--sample', type=int, help='If set, reads sample rows first to detect types (not used now)')
    args = parser.parse_args()
    try:
        r = preprocess(args.csv, out=args.out, sample_rows=args.sample)
    except Exception as e:
        print('Error during preprocessing:', e, file=sys.stderr)
        sys.exit(2)
    print('Preprocessing complete')
    print('Cleaned CSV:', r['output_path'])
    print('Report JSON:', r['report_path'])
