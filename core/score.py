"""Reduce several assay columns to one comparable mineralization score."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_mineralization_score(df: pd.DataFrame, score_columns,
                               log_transform: bool = True,
                               active_quantile: float = 0.85) -> pd.DataFrame:
    """Add min_score / score_rank / is_active.

    Each element is clipped at zero, optionally log1p'd to tame the skew of
    assay distributions, z-scored so elements of different units are
    commensurate, then averaged.
    """
    df = df.copy()
    parts, used = [], []

    for col in score_columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce').clip(lower=0)
        if log_transform:
            s = np.log1p(s)
        sd = s.std(skipna=True)
        if pd.isna(sd) or sd == 0:
            continue
        parts.append(((s - s.mean(skipna=True)) / sd).fillna(0.0))
        used.append(col)

    if not parts:
        raise ValueError(
            f"None of score_columns={list(score_columns)} were usable. "
            f"Available: {[c for c in df.columns][:40]}")

    df['min_score'] = np.vstack([p.values for p in parts]).T.mean(axis=1)
    df['score_rank'] = df['min_score'].rank(pct=True)
    df['is_active'] = df['score_rank'] >= active_quantile
    df.attrs['score_columns_used'] = used
    return df
