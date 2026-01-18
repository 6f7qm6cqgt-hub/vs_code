# -*- coding: utf-8 -*-
"""
src.data.industry_csv

中信行业分类指数（本地文件）读取模块

- 行业指数从本地 CSV  读取

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 工具函数
# =============================================================================

def _clean_text(x) -> str:
    return str(x).strip().replace("\t", "").replace("\u3000", "")
    # 从文本中去除空白


def _read_table(path: str) -> pd.DataFrame:
    """
    自动读取 csv / excel
    """
    p = str(path).lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".xls") or p.endswith(".xlsx"):
        return pd.read_excel(path)
    raise ValueError(f"不支持的文件类型: {path}")
    # 仅可识别csv/xlsx

# =============================================================================
# 交易日工具（供 grouping / event_study 使用）
# =============================================================================

@dataclass
class TradeDayTools:
    prev_trade_day: Callable[[pd.Timestamp], Optional[pd.Timestamp]]
    add_trade_days: Callable[[pd.Timestamp, int], Optional[pd.Timestamp]]


def make_trade_day_tools(dates: pd.DatetimeIndex) -> TradeDayTools:
    """
    根据交易日序列 dates 构造工具：
    - prev_trade_day
    - add_trade_days
    找到前一个/后一个交易日
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    pos = pd.Series(np.arange(len(dates)), index=dates)

    def prev_trade_day(dt) -> Optional[pd.Timestamp]:
        dt = pd.Timestamp(dt)
        if dt in pos.index:
            return dt
        i = pos.index.searchsorted(dt, side="right") - 1
        return None if i < 0 else pos.index[i]

    def add_trade_days(T, h: int) -> Optional[pd.Timestamp]:
        T = pd.Timestamp(T)
        if T not in pos.index:
            return None
        i = int(pos[T]) + int(h)
        if i < 0 or i >= len(dates):
            return None
        return dates[i]

    return TradeDayTools(prev_trade_day, add_trade_days)


# =============================================================================
# 行业指数读取
# =============================================================================

def load_industry_close_from_csv(
    path: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    读取行业指数收盘价，输出宽表：

    index   : DatetimeIndex（日频、唯一、排序）
    columns : 行业名称
    values  : 收盘价（float)

    """
    raw = _read_table(path) # raw为原始数据
    raw.columns = [_clean_text(c) for c in raw.columns]

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    cols_lower = {c.lower(): c for c in raw.columns}
    is_long = {"sec_type_name", "enddate", "closeprice"}.issubset(cols_lower)

    # ------------------------------------------------------------------
    # 长表
    # ------------------------------------------------------------------
    if is_long:
        df = raw.rename(
            columns={
                cols_lower["sec_type_name"]: "industry",
                cols_lower["enddate"]: "enddate",
                cols_lower["closeprice"]: "close",
            }
        ).copy()

        df["industry"] = df["industry"].astype(str).str.strip()

        # 优先 yyyymmdd
        df["date"] = pd.to_datetime(
            df["enddate"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        # fallback
        if df["date"].isna().any():
            df.loc[df["date"].isna(), "date"] = pd.to_datetime(
                df.loc[df["date"].isna(), "enddate"],
                errors="coerce",
            )

        df = df.dropna(subset=["date", "industry", "close"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        ind_close = (
            df.pivot(index="date", columns="industry", values="close")
            .sort_index()
        )

    # ------------------------------------------------------------------
    # 宽表
    # ------------------------------------------------------------------
    else:
        df = raw.copy()
        date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].isna().any():
            bad = df.loc[df[date_col].isna(), date_col].unique()[:5]
            raise ValueError(f"日期解析失败示例: {bad}")

        df = df.set_index(date_col).sort_index()
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        ind_close = df.apply(pd.to_numeric, errors="coerce")

    # ------------------------------------------------------------------
    # 统一清洗 index（非常关键）
    # ------------------------------------------------------------------
    ind_close.index = pd.to_datetime(ind_close.index, errors="coerce").normalize()
    ind_close = ind_close.sort_index()

    # 去重（解决你“第二次运行才炸”的根因）
    if ind_close.index.has_duplicates:
        ind_close = ind_close[~ind_close.index.duplicated(keep="last")]

    return ind_close


# =============================================================================
# 清洗 + 面板构建
# =============================================================================

def clean_industry_close(
    ind_close: pd.DataFrame,
    exclude_industries,
    drop_bench_col: str = "中证全指",
    valid_ratio_threshold: float = 0.90,
) -> pd.DataFrame:
    out = ind_close.copy()
    out = out.drop(columns=list(exclude_industries), errors="ignore")
    out = out.drop(columns=[drop_bench_col], errors="ignore")

    valid_ratio = out.notna().mean()
    out = out.loc[:, valid_ratio >= valid_ratio_threshold]
    return out.sort_index()


def build_industry_panel(
    csv_path: str,
    start: str,
    end: str,
    exclude_industries,
    valid_ratio_threshold: float = 0.90,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DatetimeIndex, TradeDayTools]:
    """
    【main.py 唯一需要的接口】

    返回：
    - ind_close : 行业收盘价
    - ret       : 日收益率
    - bench_ret : 行业等权基准
    - dates     : 交易日
    - tools     : TradeDayTools
    """
    ind_close = load_industry_close_from_csv(csv_path, start, end)
    ind_close = clean_industry_close(
        ind_close,
        exclude_industries=exclude_industries,
        valid_ratio_threshold=valid_ratio_threshold,
    )

    ret = ind_close.pct_change()
    bench_ret = ret.mean(axis=1)

    dates = ind_close.index
    tools = make_trade_day_tools(dates)

    return ind_close, ret, bench_ret, dates, tools
