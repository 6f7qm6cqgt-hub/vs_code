# -*- coding: utf-8 -*-
"""
src.backtest.long_short

日频多空净值模块
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def build_long_short_nav(
    ret: pd.DataFrame,
    bench_ret: pd.Series,
    dates: pd.DatetimeIndex,
    event_groups: Dict[pd.Timestamp, Dict[int, List[str]]],
    event_end: Dict[pd.Timestamp, pd.Timestamp],
) -> pd.Series:
    """
    构建日频多空净值序列（严格对齐 main.py 口径）。
    """
    ls_ret = bench_ret.copy()

    # 事件日倒序遍历
    event_items_desc = sorted(event_end.items(), key=lambda x: x[0], reverse=True)

    for i in range(1, len(dates)):
        d = dates[i]
        d_prev = dates[i - 1]

        for T, end_ in event_items_desc:
            if d_prev >= T and d_prev < end_:
                groups = event_groups.get(T)
                if groups is None:
                    continue

                rrow = ret.loc[d]
                long_names = [x for x in groups[5] if x in rrow.index]
                short_names = [x for x in groups[1] if x in rrow.index]

                # 某一侧为空则 break
                if len(long_names) == 0 or len(short_names) == 0:
                    break

                ls_ret.loc[d] = float(rrow[long_names].mean() - rrow[short_names].mean())
                break

    ls_nav = (1.0 + ls_ret.fillna(0.0)).cumprod()
    return ls_nav
