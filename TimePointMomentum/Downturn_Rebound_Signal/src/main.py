# -*- coding: utf-8 -*-
"""
src.main
--------------------------------------------------------------------

1. 当市场出现显著下跌并触发反弹信号时，
   行业之间是否存在可系统利用的相对表现差异？
2. 以“反弹阶段相对动量”为分组依据构建的多空组合，
   是否在事件窗口内显著跑赢行业等权基准？
3. 若将该事件型策略扩展为日频、可切仓的多空策略，
   其长期净值表现如何？

本脚本作为整个研究的唯一入口，通过 `python -m src.main` 运行。
"""
from __future__ import annotations
from src.config import (
    START,
    END,
    INDUSTRY_CSV,
    EXCLUDE_INDUSTRIES,
    HOLD_DAYS,
    PRE_N,
    ensure_cache_dir,
)
from src.data.csi_index import load_csi_all_ohlc
from src.signals.downturn_rebound import detect_downturn_rebound
from src.report.plots import (
    plot_signals_on_price,
    plot_signal_year_distribution,
    plot_group_excess_bar,
    plot_nav,
)
from src.backtest.long_short import build_long_short_nav
from src.backtest.event_study import run_event_study
from src.backtest.grouping import build_event_groups_and_end
from src.data.industry_csv import build_industry_panel

import pandas as pd
import numpy as np

def main() -> None:
    # ---------------------------------------------------------------------
    #  信号识别
    # ---------------------------------------------------------------------
    ensure_cache_dir()

    ohlc = load_csi_all_ohlc(start=START, end=END)
    signals = detect_downturn_rebound(ohlc)

    #  sig_col 选择逻辑
    sig_col = "final_signal" if ("final_signal" in signals.columns) else "signal"

    # 打印信号天数与年度分布
    sig_days = pd.DatetimeIndex(signals.index[signals[sig_col] == 1])
    print("【信号天数】", int((signals[sig_col] == 1).sum()))
    print("【信号年度分布】")
    print(pd.Series(sig_days.year).value_counts().sort_index())

    # 信号示意图 + 年度分布图
    close = ohlc["close"]
    plot_signals_on_price(close, signals, sig_col=sig_col, title="Downturn-Rebound Signals on CSI All Share")
    _ = plot_signal_year_distribution(signals, sig_col=sig_col, title="Signal Count by Year")

    # ---------------------------------------------------------------------
    #  行业 CSV 读取、清洗、收益率与交易日工具
    # ---------------------------------------------------------------------

    ind_close, ret, bench_ret, dates, tools = build_industry_panel(
        csv_path=INDUSTRY_CSV,
        start=START,
        end=END,
        exclude_industries=EXCLUDE_INDUSTRIES,
        valid_ratio_threshold=0.90,
    )

    prev_trade_day = tools.prev_trade_day
    add_trade_days = tools.add_trade_days

    # ---------------------------------------------------------------------
    #  构建事件分组 event_groups 与事件窗口 event_end（切仓规则）
    # ---------------------------------------------------------------------
    sig_list, event_groups, event_end, sig_col2 = build_event_groups_and_end(
        signals=signals,
        ret=ret,
        dates=dates,
        tools=tools,
    )

    # 一致性检查：sig_col2 应该与前面 sig_col 相同
    # assert sig_col2 == sig_col

    # ---------------------------------------------------------------------
    # 事件研究 mean_excess（gross=价格比，bench=横截面均值）
    # ---------------------------------------------------------------------
    event_excess, mean_excess = run_event_study(
        signals=signals,
        ind_close=ind_close,
        event_groups=event_groups,
        event_end=event_end,
        tools=tools,
        sig_col=sig_col,  # 注意：要用你前面动态选择的那个 sig_col
    )

    print("【分组次均超额收益（事件均值，含切仓）】")
    print(mean_excess)

    plot_group_excess_bar(
        mean_excess,
        title="Downturn-Rebound: Event Mean Excess Return (Switch-on-New-Signal)\n"
              "Momentum=Pure Industry (DailyMean-DailyMean), Bench=CrossSec Avg",
        ylabel="Mean excess return",
    )

    # ---------------------------------------------------------------------
    # 日频多空净值（事件期 G5-G1，非事件期 bench）
    # ---------------------------------------------------------------------
    ls_nav = build_long_short_nav(
        ret=ret,
        bench_ret=bench_ret,
        dates=dates,
        event_groups=event_groups,
        event_end=event_end,
    )

    print("【多空期末净值】", float(ls_nav.iloc[-1]))

    plot_nav(
        ls_nav,
        title="Downturn-Rebound: Long-Short NAV\nActive: G5-G1 | Inactive: EW-Industry Bench",
        ylabel="NAV",
    )


if __name__ == "__main__":
    main()