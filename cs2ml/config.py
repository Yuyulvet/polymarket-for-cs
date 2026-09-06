"""Central configuration: paths and constants."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "cs2.db"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- bo3.gg API ---
BO3_BASE = "https://api.bo3.gg"
BO3_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
BO3_REQUEST_INTERVAL_SEC = 0.35  # polite rate limit
BO3_MAX_RETRIES = 5
BO3_TIMEOUT_SEC = 30

DISCIPLINE_CS2 = 1  # bo3.gg discipline id for Counter-Strike

# --- Glicko-2 ---
GLICKO_TAU = 0.5
GLICKO_BASE_RATING = 1500.0
GLICKO_BASE_RD = 350.0
GLICKO_BASE_VOL = 0.06
# daily RD growth: RD grows ~100 points over ~180 idle days
GLICKO_DAILY_C = 7.5

# --- features / training ---
MIN_TEAM_HISTORY = 3          # prior matches needed before a team emits a feature row
FORM_WINDOWS = (5, 10, 20)
H2H_WINDOW_DAYS = 730
TEST_STEP_DAYS = 30           # walk-forward test window length
MIN_TRAIN_DAYS = 180          # minimum training history before first test fold
RECENCY_HALFLIFE_DAYS = 270   # recency decay: sample weight halves every N days before fold cutoff
