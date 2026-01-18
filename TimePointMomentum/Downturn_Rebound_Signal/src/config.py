"""
src.config

- 只放配置与常量，不写策略逻辑，避免引入结果差异。
- 默认值与原 Notebook 保持一致。
"""

from __future__ import annotations

import os

#  回测区间
START: str = "2013-01-01"
END: str = "2023-05-31"

#  缓存目录
CACHE_DIR: str = "data_cache"

#  TuShare Token
# 注意：需要自行填写。
TUSHARE_TOKEN: str = "".strip()

#  指数名称
CSI_INDEX_NAME: str = "中证全指"
CSI_MARKET: str = "CSI"

# -----------------------------------------------------------------------------
# 行业数据与回测参数
# -----------------------------------------------------------------------------

# 行业指数收盘价 CSV
# 注意：本地文件需要修改路径
INDUSTRY_CSV: str = r"C:\Users\36895\OneDrive\Desktop\TimePointMomentum\Downturn_Rebound_Signal\中信行业指数收盘价.csv"

# 事件持有期（交易日）
HOLD_DAYS: int = 20

# 动量计算：t_low 前的对照窗口长度
PRE_N: int = 20

# 剔除行业（研报要求)
EXCLUDE_INDUSTRIES = {"综合", "综合金融"}

# -----------------------------------------------------------------------------
# 3) 项目路径小工具（便于 PyCharm 运行）
# -----------------------------------------------------------------------------

def ensure_cache_dir() -> None:
    """确保缓存目录存在。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
