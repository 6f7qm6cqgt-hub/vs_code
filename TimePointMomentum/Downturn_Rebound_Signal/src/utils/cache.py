"""
src.utils.cache

缓存工具（终局版）：
- cache 只负责 IO
- 不改变 DataFrame 结构
"""

from __future__ import annotations

import os
import pandas as pd

from src.config import CACHE_DIR


def yyyymmdd(date_str: str) -> str:
    """将 'YYYY-MM-DD' 转换为 'YYYYMMDD'"""
    return date_str.replace("-", "")


def cache_path(filename: str) -> str:
    return os.path.join(CACHE_DIR, filename)


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _maybe_parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    尝试把常见日期列解析成 datetime，但【不 set_index】
    """
    if df is None or df.empty:
        return df

    for c in ["date", "trade_date", "datetime", "cal_date"]:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().mean() >= 0.5:
                df = df.copy()
                df[c] = dt.dt.normalize()
    return df


def load_csv_cache(filename: str) -> pd.DataFrame | None:
    fp = cache_path(filename)
    if not os.path.exists(fp):
        return None

    df = pd.read_csv(fp)
    return _maybe_parse_date_columns(df)


def save_csv_cache(df: pd.DataFrame, filename: str) -> None:
    _ensure_cache_dir()
    fp = cache_path(filename)

    if pd.api.types.is_datetime64_any_dtype(df.index):
        tmp = df.copy()
        tmp.index = pd.to_datetime(tmp.index, errors="coerce").normalize()
        tmp.to_csv(fp, index=True, index_label="date")
    else:
        df.to_csv(fp, index=False)


def invalidate_cache(match_substr: str) -> None:
    if not os.path.exists(CACHE_DIR):
        return

    for fn in os.listdir(CACHE_DIR):
        if match_substr in fn:
            try:
                os.remove(cache_path(fn))
            except FileNotFoundError:
                pass
