"""Central configuration for Binance futures pre-pump scanner."""

# Общие
SYMBOLS_BLACKLIST: list[str] = []
KLINE_INTERVAL = "1m"
HISTORY_MINUTES = 60 * 24

# Сеть
REQUEST_TIMEOUT = 15
REQUEST_RETRIES = 3
REQUEST_BACKOFF_FACTOR = 1.0

# Окна
ATR_WINDOW = 60
BBW_WINDOW = 60
BBW_PERCENTILE_WINDOW = 60 * 24
VOL_RATIO_60_WINDOW = 60
VOL_RATIO_30_WINDOW = 30
BODY_RATIO_MEAN_WINDOW = 30
REL_STRENGTH_WINDOW = 180

# Улучшенные pre-pump фильтры
PRE_BBW_MAX_PERCENTILE = 20
PRE_VOL60_MIN = 1.2
PRE_VOL60_MAX = 3.0
PRE_BODYR_WINDOW = 30
PRE_BODYR_MAX = 0.4
PRE_REL_STRENGTH_MIN = 0.0
PRE_BUYR_WINDOW = 30
PRE_BUYR_MIN = 0.6
PRE_FUNDING_ABS_MAX = 0.0005
PRE_OI_CHANGE_MIN = 0.05
PRE_PUMP_SCORE_MIN = 4

# Пороги входа (trigger)
VOL_RATIO_30_ENTRY = 3.0
BODY_RATIO_ENTRY = 0.6
ATR_MULTIPLIER_ENTRY = 2.0
BASE_RANGE_WINDOW = 60

# BTC-референс
BTC_SYMBOL = "BTCUSDT"

# Логирование/оповещения
LOG_SIGNALS_TO_CONSOLE = True
LOG_SIGNALS_TO_FILE = True
SIGNALS_LOG_PATH = "signals.log"
ENABLE_TELEGRAM_NOTIFICATIONS = False
DEBUG_LOG_CONDITIONS = True

# === Rule-based улучшения ===
MIN_QUOTE_VOLUME_1M = 50_000
MIN_QUOTE_VOLUME_60M = 1_000_000
SIGNAL_COOLDOWN_MINUTES = 30
CONFIRMATION_BARS = 1

# === Нейросеть ===
ENABLE_NN_ENTRY = True
NN_ENTRY_THRESHOLD = 0.9
LOG_NN_SIGNALS_TO_CONSOLE = True
LOG_NN_SIGNALS_TO_FILE = True
NN_MODEL_PATH = "models/pump_classifier.pt"
NN_WINDOW_SIZE = 60
NN_FEATURE_COLUMNS = [
    "close",
    "volume",
    "atr_60",
    "atr_30",
    "bbw",
    "bbw_percentile",
    "vol_ratio_60",
    "vol_ratio_30",
    "body_ratio",
    "buy_ratio",
    "rel_strength_180",
    "funding_rate",
    "oi_change_60",
]
NN_HIDDEN_SIZES = (128, 64)

# === Backtest ===
BACKTEST_INTERVAL = "1m"
BACKTEST_DAYS = 7
BACKTEST_SYMBOLS: list[str] = []
BACKTEST_OUTPUT_PATH = "backtest_signals.csv"

# === КЭШ СВЕЧЕЙ ===
USE_KLINE_CACHE = True
KLINE_CACHE_DIR = "data/klines"
KLINE_CACHE_FORMAT = "parquet"  # "csv" или "parquet"
KLINE_CACHE_INTERVALS = ["1m"]
FORCE_REFRESH_KLINE_CACHE = False

# Параметрический гридсерч
PARAM_GRID = [
    {
        "PRE_BBW_MAX_PERCENTILE": 15,
        "PRE_PUMP_SCORE_MIN": 4,
        "VOL_RATIO_30_ENTRY": 3.5,
        "BODY_RATIO_ENTRY": 0.6,
    },
    {
        "PRE_BBW_MAX_PERCENTILE": 20,
        "PRE_PUMP_SCORE_MIN": 5,
        "VOL_RATIO_30_ENTRY": 4.0,
        "BODY_RATIO_ENTRY": 0.7,
    },
]
BACKTEST_PARAM_GRID_OUTPUT = "backtest_param_grid_signals.csv"
