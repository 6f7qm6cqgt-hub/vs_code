"""
src.report.plots

绘图函数集合

- 输入由 main / backtest 模块传入
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def plot_signals_on_price(
    close: pd.Series,
    signals: pd.DataFrame,
    sig_col: str,
    title: str = "Downturn-Rebound Signals on Price",
) -> None:
    """
    收盘价曲线 + 信号点示意图
    """
    sig_days = pd.DatetimeIndex(signals.index[signals[sig_col] == 1])

    plt.figure(figsize=(12, 4))
    close.plot(label="Close")

    # 即便没有信号，也不要报错
    if len(sig_days) > 0:
        plt.scatter(sig_days, close.loc[sig_days], marker="^", s=50, label="Signal")

    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_signal_year_distribution(
    signals: pd.DataFrame,
    sig_col: str,
    title: str = "Signal Count by Year",
) -> pd.Series:
    """
    信号年度分布（支持“无信号”情况）
    """
    sig_days = pd.DatetimeIndex(signals.index[signals[sig_col] == 1])
    year_counts = pd.Series(sig_days.year).value_counts().sort_index()

    plt.figure(figsize=(8, 3))

    # ✅ 核心修复：无信号时不画 bar
    if year_counts.empty:
        plt.title(title)
        plt.text(0.5, 0.5, "No signals found", ha="center", va="center")
        plt.axis("off")
        plt.show()
        return year_counts

    year_counts.plot(kind="bar")
    plt.title(title)
    plt.ylabel("# Signals")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    return year_counts


def plot_group_excess_bar(
    mean_excess: pd.Series,
    title: str = "Event Study Excess Return by Group",
    ylabel: str = "Excess Return",
) -> None:
    """
    各组平均超额收益柱状图（支持空结果）
    """
    plt.figure(figsize=(8, 3))

    if mean_excess is None or mean_excess.empty:
        plt.title(title)
        plt.text(0.5, 0.5, "No event results", ha="center", va="center")
        plt.axis("off")
        plt.show()
        return

    mean_excess.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def plot_nav(
    nav: pd.Series,
    title: str = "Strategy NAV",
    ylabel: str = "NAV",
) -> None:
    """
    多空净值曲线（支持空序列）
    """
    plt.figure(figsize=(10, 4))

    if nav is None or nav.empty:
        plt.title(title)
        plt.text(0.5, 0.5, "No NAV data", ha="center", va="center")
        plt.axis("off")
        plt.show()
        return

    nav.plot()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.show()
