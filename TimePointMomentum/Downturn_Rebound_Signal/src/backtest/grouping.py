# -*- coding: utf-8 -*-
"""
src.backtest.grouping

事件分组与事件结束日构造模块
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Protocol, Optional

import numpy as np
import pandas as pd


class TradeDayToolsLike(Protocol):
    def prev_trade_day(self, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
        ...

    def add_trade_days(self, dt: pd.Timestamp, h: int) -> Optional[pd.Timestamp]:
        ...


def _infer_sig_col(signals: pd.DataFrame) -> str:
    """
    防止sig_col未定义问题
    优先级：final_signal > signal > 第一个全是 0/1 的列
    """
    for c in ["final_signal", "signal", "sig", "rebound_signal"]:
        if c in signals.columns:
            return c

    # 兜底：找一个只包含 {0,1,NaN} 的列
    for c in signals.columns:
        x = signals[c].dropna().unique()
        if set(x).issubset({0, 1}):
            return c

    raise ValueError(f"无法推断信号列名，signals.columns={signals.columns.tolist()}")


def build_event_groups_and_end(
    signals: pd.DataFrame,
    ret: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tools: TradeDayToolsLike,
    holding_days: int = 20,
    n_groups: int = 5,
    sig_col: str | None = None,
) -> Tuple[pd.DatetimeIndex, Dict[pd.Timestamp, Dict[int, List[str]]], Dict[pd.Timestamp, pd.Timestamp], str]:
    """
    构建：
    - sig_list: 信号日列表（DatetimeIndex）
    - event_groups[entry_day][g] -> 行业列表（g=1..n_groups）
    - event_end[entry_day] -> exit_day
    - sig_col2: 实际使用的信号列名（返回给 main 继续用）
    """

    # 1) 信号列名
    sig_col2 = sig_col if sig_col is not None else _infer_sig_col(signals)

    # 2) 信号日列表（以 signals.index 为准，确保 datetime）
    sig_idx = pd.to_datetime(signals.index, errors="coerce")
    sig_mask = (signals[sig_col2] == 1).values
    sig_list = pd.DatetimeIndex(sig_idx[sig_mask]).dropna().sort_values()

    prev_trade_day = tools.prev_trade_day
    add_trade_days = tools.add_trade_days

    event_groups: Dict[pd.Timestamp, Dict[int, List[str]]] = {}
    event_end: Dict[pd.Timestamp, pd.Timestamp] = {}

    for T in sig_list:
        entry = prev_trade_day(T)
        if entry is None:
            continue
        if entry not in ret.index:
            continue

        exit_ = add_trade_days(entry, holding_days)
        if exit_ is None or exit_ not in ret.index:
            continue
        if exit_ <= entry:
            continue

        # 用 entry 当日截面收益做分组
        cross = ret.loc[entry].dropna()
        if cross.empty:
            continue

        ranked = cross.sort_values()
        codes = ranked.index.tolist()
        if len(codes) < n_groups:
            continue

        # 均分切片分组（稳定，不怕重复值）
        edges = np.linspace(0, len(codes), n_groups + 1).astype(int)
        groups: Dict[int, List[str]] = {}
        for g in range(1, n_groups + 1):
            groups[g] = codes[edges[g - 1] : edges[g]]

        event_groups[entry] = groups
        event_end[entry] = exit_

    return sig_list, event_groups, event_end, sig_col2
