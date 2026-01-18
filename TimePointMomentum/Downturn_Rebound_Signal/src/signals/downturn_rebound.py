"""src.signals.downturn_rebound

下跌反弹信号识别模块。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_true_range_pct(df: pd.DataFrame) -> pd.Series:
    """
    True Range（百分比形式）：max{(H-L)/PrevC, |H-PrevC|/PrevC, |PrevC-L|/PrevC}
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    hl = (high - low) / prev_close
    hc = (high - prev_close).abs() / prev_close
    cl = (prev_close - low).abs() / prev_close

    return pd.concat([hl, hc, cl], axis=1).max(axis=1)


def compute_atr(df: pd.DataFrame, window: int = 60) -> pd.Series:
    tr = compute_true_range_pct(df)
    return tr.rolling(window=window, min_periods=window).mean()


def compute_dynamic_thresholds(
    atr: pd.Series,
    d0: float = -0.05,
    u0: float = 0.005,
    low_vol: float = 0.01,
    high_vol: float = 0.02,
) -> pd.DataFrame:
    """
    根据 ATR 水平对阈值进行缩放：
    - atr < low_vol  -> sqrt(atr/low_vol)
    - atr > high_vol -> sqrt(atr/high_vol)
    并 clip 到 [0.2, 5.0]
    """
    adj = np.ones(len(atr))
    adj = np.where(atr < low_vol, np.sqrt(atr / low_vol), adj)
    adj = np.where(atr > high_vol, np.sqrt(atr / high_vol), adj)
    adj = pd.Series(adj, index=atr.index).clip(0.2, 5.0)
    # adj为研报定义缩放因子

    out = pd.DataFrame(index=atr.index)
    out["D"] = d0 * adj
    out["U"] = u0 * adj
    return out


def detect_downturn_rebound(
    ohlc: pd.DataFrame,
    atr_window: int = 60,
    lookback_peak: int = 60,
    lookback_trough: int = 20,
    cooldown: int = 10,
    bounce_tol: float = 2.0,
    d0: float = -0.05,
    u0: float = 0.005,
    low_vol: float = 0.01,
    high_vol: float = 0.02,
) -> pd.DataFrame:
    """
    输入：
      ohlc: index=交易日，columns 包含 high/low/close

    输出：
      out: index=交易日，包含 signal/t_high/t_low/drawdown/.../D_t/U_t
    """

    # ------------------------------------------------------------------
    # 确保交易日 index 唯一
    # ------------------------------------------------------------------
    df = ohlc.copy()
    df.index = pd.to_datetime(df.index, errors="coerce").normalize()
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    # 必要列检查
    for c in ["high", "low", "close"]:
        if c not in df.columns:
            raise KeyError(f"ohlc 缺少必要列: {c}，当前列={df.columns.tolist()}")

    # 计算 ATR 与动态阈值
    atr = compute_atr(df, window=atr_window)
    thr = compute_dynamic_thresholds(atr, d0=d0, u0=u0, low_vol=low_vol, high_vol=high_vol)
    thr = thr.reindex(df.index)  # 对齐到 df.index（避免 index mismatch）

    out = pd.DataFrame(index=df.index)
    out["signal"] = 0
    out["t_high"] = pd.NaT
    out["t_low"] = pd.NaT
    out["drawdown"] = np.nan
    out["max_bounce_in_drop"] = np.nan
    out["rebound_from_low"] = np.nan
    out["D_t"] = thr["D"]
    out["U_t"] = thr["U"]

    dates = df.index
    close = df["close"]

    last_signal_pos = None
    start_i = max(atr_window, lookback_peak + lookback_trough)

    for i in range(start_i, len(df)):
        # cooldown：信号之间至少间隔 cooldown 个交易日
        if last_signal_pos is not None and i - last_signal_pos < cooldown:
            continue

        T = dates[i]

        # 取阈值（保证是标量）
        D_val = out.loc[T, "D_t"]
        U_val = out.loc[T, "U_t"]
        if isinstance(D_val, pd.Series):
            D_val = D_val.iloc[-1]
        if isinstance(U_val, pd.Series):
            U_val = U_val.iloc[-1]
        if pd.isna(D_val) or pd.isna(U_val):
            continue

        D_T = float(D_val)
        U_T = float(U_val)

        # 1) 近 lookback_trough 日内找 t_low
        low_window = close.iloc[i - lookback_trough : i + 1]
        t_low = low_window.idxmin()
        c_low = float(close.loc[t_low])

        # 2) t_low 前 lookback_peak 日内找 t_high
        pos_low = dates.get_loc(t_low)
        start_high = max(0, pos_low - lookback_peak)
        high_window = close.iloc[start_high : pos_low + 1]
        t_high = high_window.idxmax()
        c_high = float(close.loc[t_high])

        # 3) 下跌幅度
        drawdown = c_low / c_high - 1.0
        if drawdown > D_T:
            continue

        # 4) 下跌过程中的最大反弹：去除震荡下行
        seg = close.loc[t_high:t_low]
        if len(seg) < 2:
            continue

        running_min = seg.cummin()
        bounce = seg / running_min - 1.0
        if float(bounce.max()) > bounce_tol * U_T:
            continue

        # 5) 从 t_low 到 T 的反弹
        rebound = float(close.loc[T] / c_low - 1.0)
        if rebound < U_T:
            continue

        out.at[T, "signal"] = 1
        out.at[T, "t_high"] = t_high
        out.at[T, "t_low"] = t_low
        out.at[T, "drawdown"] = drawdown
        out.at[T, "max_bounce_in_drop"] = float(bounce.max())
        out.at[T, "rebound_from_low"] = rebound
        last_signal_pos = i

    return out
