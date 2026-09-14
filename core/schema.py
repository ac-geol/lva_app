"""Column naming: internal standard names, alias resolution for arbitrary client CSVs.

Pure pandas/numpy. No file I/O, no plotting.
"""
from __future__ import annotations

import pandas as pd

# Internal standard names. Every downstream module refers to columns by these.
COLS = {
    'hole': 'HoleID',
    'from': 'From_m',
    'to': 'To_m',
    'depth': 'Depth_m',
    'dip': 'Dip',
    'azm': 'Azimuth',
    'easting': 'Easting',
    'northing': 'Northing',
    'elev': 'Elevation',
    'length': 'Length_m',
    'property': 'Property',
    'prospect': 'Prospect',
    'holetype': 'HoleType',
}

ALIASES = {
    COLS['hole']:     ['HOLEID', 'HOLE-ID', 'HOLE_ID', 'BHID', 'DHID', 'DDHID'],
    COLS['from']:     ['FROM', 'FROM_M', 'FROMM', 'DEPTH_FROM', 'START', 'INTERVAL_FROM'],
    COLS['to']:       ['TO', 'TO_M', 'TOM', 'DEPTH_TO', 'END', 'INTERVAL_TO'],
    COLS['depth']:    ['DEPTH', 'DEPTH_M', 'MD', 'MEASUREDDEPTH', 'MEASURED_DEPTH'],
    COLS['dip']:      ['DIP', 'DIP_DEG', 'DIP_DEGREES'],
    COLS['azm']:      ['AZIMUTH', 'AZM', 'AZI', 'AZIMUTH_DEG', 'AZIMUT'],
    COLS['easting']:  ['EAST', 'EASTING', 'X', 'LOCATIONX', 'UTM_E', 'E'],
    COLS['northing']: ['NORTH', 'NORTHING', 'Y', 'LOCATIONY', 'UTM_N', 'N'],
    COLS['elev']:     ['ELEV', 'ELEVATION', 'RL', 'Z', 'LOCATIONZ'],
    COLS['length']:   ['LENGTH', 'LENGTH_M', 'TD', 'TOTALDEPTH', 'DEPTH_TOTAL'],
    COLS['property']: ['PROPERTY', 'PROP'],
    COLS['prospect']: ['PROSPECT', 'AREA'],
    COLS['holetype']: ['HOLETYPE', 'HOLE_TYPE', 'TYPE'],
}

# Which internal key each table needs, and which are merely nice to have.
REQUIRED = {
    'samples':  {'hole': True, 'from': True, 'to': True},
    # A single table of already-desurveyed points: what Leapfrog and Datamine
    # export. Carries its own coordinates, so no collar or survey table is
    # needed and no desurveying happens. Easting/Northing/Elevation resolve
    # from 'X'/'Y'/'Z' through the alias list above.
    'points':   {'hole': True, 'from': True, 'to': True,
                 'easting': True, 'northing': True, 'elevation': True},
    'collars':  {'hole': True, 'easting': True, 'northing': True,
                 'elevation': True, 'length': True},
    'surveys':  {'hole': True, 'depth': True, 'dip': True, 'azimuth': True},
}

# ('cfg key', internal std name) pairs considered when renaming any table.
_SPEC = [
    ('hole', COLS['hole']), ('from', COLS['from']), ('to', COLS['to']),
    ('depth', COLS['depth']), ('dip', COLS['dip']), ('azimuth', COLS['azm']),
    ('easting', COLS['easting']), ('northing', COLS['northing']),
    ('elevation', COLS['elev']), ('length', COLS['length']),
    ('property', COLS['property']), ('prospect', COLS['prospect']),
    ('holetype', COLS['holetype']),
]

# BOM, stray quotes and whitespace are routine in exported drillhole CSVs.
_STRIP = ' \t\r\n"\'﻿'


def clean_header(c) -> str:
    return str(c).strip(_STRIP).strip()


def standardize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_header(c) for c in df.columns]
    return df


def resolve_col(df: pd.DataFrame, configured_name, std_name: str, *,
                required: bool, auto_detect: bool = True):
    """Find the actual column in `df` that carries `std_name`.

    Explicit config wins; then the exact standard name; then the alias list;
    then a case-insensitive pass over both.
    """
    if configured_name is not None:
        candidates = (configured_name if isinstance(configured_name, (list, tuple))
                      else [configured_name])
        for c in candidates:
            if c in df.columns:
                return c

    if auto_detect:
        if std_name in df.columns:
            return std_name
        for c in ALIASES.get(std_name, []):
            if c in df.columns:
                return c
        lower_map = {str(c).lower(): c for c in df.columns}
        if std_name.lower() in lower_map:
            return lower_map[std_name.lower()]
        for c in ALIASES.get(std_name, []):
            if str(c).lower() in lower_map:
                return lower_map[str(c).lower()]

    if required:
        raise KeyError(
            f"Required column for {std_name!r} not found. Set it explicitly in "
            f"cfg['columns']. Available (first 30): {list(df.columns)[:30]}"
        )
    return None


def rename_to_internal(df: pd.DataFrame, mapping: dict | None, required: dict,
                       *, auto_detect: bool = True) -> pd.DataFrame:
    """Rename a table's key columns to the internal standard names."""
    df = standardize_headers(df)
    rename = {}
    for key, std_name in _SPEC:
        cfg_name = (mapping or {}).get(key)
        raw = resolve_col(df, cfg_name, std_name,
                          required=bool(required.get(key, False)),
                          auto_detect=auto_detect)
        if raw is not None and raw != std_name:
            rename[raw] = std_name
    return df.rename(columns=rename)


def describe_mapping(df: pd.DataFrame, mapping: dict | None, required: dict,
                     *, auto_detect: bool = True) -> pd.DataFrame:
    """What the resolver *would* do -- the preview an ingest UI needs."""
    rows = []
    clean = standardize_headers(df)
    for key, std_name in _SPEC:
        raw = resolve_col(clean, (mapping or {}).get(key), std_name,
                          required=False, auto_detect=auto_detect)
        rows.append({
            'internal': std_name,
            'source': raw,
            'required': bool(required.get(key, False)),
            'status': 'ok' if raw else ('MISSING' if required.get(key) else '-'),
        })
    return pd.DataFrame(rows)
