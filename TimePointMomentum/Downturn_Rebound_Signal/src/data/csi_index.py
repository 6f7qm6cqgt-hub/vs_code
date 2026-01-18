# -*- coding: utf-8 -*-
"""
src.data.csi_index

中证全指（000985.CSI）日频 OHLC 获取模块
- 每次运行都直接调用 TuShare API；取消缓存———缓存数据结构出错
"""

from __future__ import annotations

import time
import pandas as pd
import tushare as ts

from src.config import START, END, TUSHARE_TOKEN
from src.utils.cache import yyyymmdd  # 只用日期格式化，不用 cache 读写

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


def fetch_index_daily_api(ts_code: str, start: str, end: str) -> pd.DataFrame:
    """
    直接调用 TuShare API 拉取指数日频数据
    """
    time.sleep(0.3)  # 轻微限速，避免触发频控
    df = pro.index_daily(
        ts_code=ts_code,
        start_date=yyyymmdd(start),
        end_date=yyyymmdd(end),
        fields="ts_code,trade_date,high,low,close",
    )

    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "high", "low", "close"])
    # 防止空表导致KeyError

    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    # 删除NaT行 保证升序 清理旧行号
    return df
    # 最终输出df为原始日线表


def load_csi_all_ohlc(start: str = START, end: str = END, ts_code: str = "000985.CSI") -> pd.DataFrame:
    """
    返回中证全指（默认 000985.CSI）的 OHLC DataFrame：
      index  : date(datetime)
      columns: high, low, close
    """
    raw = fetch_index_daily_api(ts_code=ts_code, start=start, end=end)

    ohlc = (
        raw[["trade_date", "high", "low", "close"]]
        .rename(columns={"trade_date": "date"})
        .set_index("date")
        .sort_index()
    )

    # 确保日频唯一（避免后续 normalize 后重复）
    if ohlc.index.has_duplicates:
        ohlc = ohlc[~ohlc.index.duplicated(keep="last")]

    return ohlc
