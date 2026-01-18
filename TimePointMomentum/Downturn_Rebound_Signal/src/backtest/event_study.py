# -*- coding: utf-8 -*-
"""
src.backtest.event_study

事件研究模块
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Protocol

import numpy as np
import pandas as pd


class TradeDayToolsLike(Protocol):
    """仅约束 run_event_study 需要的接口：prev_trade_day。"""
    def prev_trade_day(self, dt: pd.Timestamp) -> pd.Timestamp | None:
        ...


def run_event_study(
    signals: pd.DataFrame,
    ind_close: pd.DataFrame,
    event_groups: Dict[pd.Timestamp, Dict[int, List[str]]],
    event_end: Dict[pd.Timestamp, pd.Timestamp],
    tools: TradeDayToolsLike,
    sig_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    - event_excess: 每个事件（索引=signal_date(entry_day)）的 G1..G5 超额收益
    - mean_excess : 对所有事件取均值后的 Series（G1..G5）

    参数
    - signals: 信号表（index 为信号日）
    - ind_close: 行业收盘价宽表（index=交易日，columns=行业）
    - event_groups/event_end: 来自 grouping 模块
    - tools: 交易日工具（需要 prev_trade_day）
    - sig_col: 实际使用的信号列名（final_signal 或 signal）
    """
    prev_trade_day = tools.prev_trade_day

    sig_list = pd.DatetimeIndex(signals.index[signals[sig_col] == 1]).sort_values()

    rows = []
    for T in sig_list:
        entry = prev_trade_day(T)
        if entry is None:
            continue

        exit_ = event_end.get(entry, None)
        if exit_ is None:
            continue
        if entry not in ind_close.index or exit_ not in ind_close.index:
            continue
        if exit_ <= entry:
            continue

        gross = (ind_close.loc[exit_] / ind_close.loc[entry] - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
        if gross.empty:
            continue

        bench_gross = float(gross.mean())

        g = event_groups.get(entry)
        if g is None:
            continue

        rec = {"signal_date": entry, "entry": entry, "exit": exit_}
        for k in [1, 2, 3, 4, 5]:
            members = [x for x in g[k] if x in gross.index]
            rec[f"G{k}"] = (float(gross.loc[members].mean()) - bench_gross) if len(members) else np.nan
        rows.append(rec)

    if len(rows) == 0:
        # 空结果时返回空表与全 NaN
        event_excess = pd.DataFrame(columns=["entry", "exit", "G1", "G2", "G3", "G4", "G5"])
        mean_excess = pd.Series({f"G{k}": np.nan for k in [1, 2, 3, 4, 5]})
        return event_excess, mean_excess

    event_excess = pd.DataFrame(rows).set_index("signal_date").sort_index()
    mean_excess = event_excess[[f"G{k}" for k in [1, 2, 3, 4, 5]]].mean()
    return event_excess, mean_excess
