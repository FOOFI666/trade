"""Central configuration for Binance futures pre-pump scanner."""

# Общие
SYMBOLS_BLACKLIST: list[str] = []
KLINE_INTERVAL = "1m"
HISTORY_MINUTES = 60 * 24

# Окна
ATR_WINDOW = 60
BBW_WINDOW = 60
BBW_PERCENTILE_WINDOW = 60 * 24
VOL_RATIO_60_WINDOW = 60
VOL_RATIO_30_WINDOW = 30
BODY_RATIO_MEAN_WINDOW = 30
REL_STRENGTH_WINDOW = 180

# Пороги pre-pump
BBW_PERCENTILE_THRESHOLD = 10
VOL_RATIO_60_MIN = 1.2
VOL_RATIO_60_MAX = 3.0
BODY_RATIO_ACCUM_MAX = 0.4
REL_STRENGTH_MIN = 0.0
BUY_RATIO_MIN = 0.6
FUNDING_NEAR_ZERO = 0.0002
OI_CHANGE_60_MIN = 0.05

PRE_PUMP_SCORE_THRESHOLD = 3

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
NN_MODEL_PATH = "models/entry_nn.pt"
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
]
NN_ENTRY_THRESHOLD = 0.9
NN_HIDDEN_SIZES = (128, 64)
LOG_NN_SIGNALS_TO_FILE = True
LOG_NN_SIGNALS_TO_CONSOLE = True
