"""Spirit-only 快跑：只为 donk 操作画像，不等 597 图 pilot。

找到 donk 出现的全部 demo（93 图，覆盖 sh1ro/magixx/zont1x），
提取行为写 player_behavior_spirit.parquet / player_util_spirit.parquet。
"""
import pandas as pd

from cs2ml import config
from cs2ml.player_behavior import _build

pf = pd.read_parquet(config.DATA_DIR / "player_features.parquet")
donk = pf[pf["name"].astype(str).str.lower().str.strip() == "donk"]
demos = sorted(donk["demo_path"].unique())
print(f"Spirit demo 数：{len(demos)}")

_build(demos, out_beh="player_behavior_spirit.parquet",
       out_util="player_util_spirit.parquet",
       out_kills="player_kills_spirit.parquet")
