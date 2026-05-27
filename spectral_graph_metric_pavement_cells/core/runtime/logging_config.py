# spectral_graph_metric_pavement_cells/core/logging_config.py
import logging, os, sys
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"

RESET = "\033[0m"
DIM = "\033[2m"
FG_RED = "\033[31m"
FG_YELLOW = "\033[33m"
BG_RED = "\033[41m"
FG_WHITE = "\033[97m"

LEVEL_TO_COLOR = {
    logging.DEBUG: DIM,                  # subtle
    logging.INFO: "",                    # no color (white/normal)
    logging.WARNING: FG_YELLOW,          # yellow
    logging.ERROR: FG_RED,               # red
    logging.CRITICAL: BG_RED + FG_WHITE  # white on red background
}

def _supports_color(stream) -> bool:
    try:
        return hasattr(stream, "isatty") and stream.isatty()
    except Exception:
        return False

def _is_pycharm() -> bool:
    # PyCharm sets PYCHARM_HOSTED=1 for Run/Debug console
    return os.getenv("PYCHARM_HOSTED") == "1"

class LevelColorFormatter(logging.Formatter):
    """Colorize only LEVELNAME; message stays uncolored."""
    def __init__(self, fmt: str, datefmt: Optional[str], use_color: bool):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        if not self.use_color:
            return s
        color = LEVEL_TO_COLOR.get(record.levelno, "")
        if not color:
            return s
        return s.replace(record.levelname, f"{color}{record.levelname}{RESET}", 1)

class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level

def setup_logging(level: Optional[str] = None,
                  to_file: Optional[str] = None,
                  color: Optional[bool] = None) -> None:
    if getattr(setup_logging, "_configured", False):
        return

    lvl_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, lvl_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Decide color usage
    pycharm = _is_pycharm()
    env_color = os.getenv("LOG_COLOR")
    if color is not None:
        use_color = bool(color)
    elif env_color is not None:
        use_color = env_color not in ("0", "false", "False", "no", "NO")
    else:
        # FORCE color in PyCharm
        use_color = True if pycharm else (_supports_color(sys.stdout) or _supports_color(sys.stderr))

    formatter = LevelColorFormatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT, use_color=use_color)

    if pycharm:
        # Specific Logging setup for PyCharm
        h_all = logging.StreamHandler(stream=sys.stdout)
        h_all.setLevel(logging.DEBUG)
        h_all.setFormatter(formatter)
        root.addHandler(h_all)
    else:
        h_out = logging.StreamHandler(stream=sys.stdout)
        h_out.setLevel(logging.DEBUG)
        h_out.addFilter(MaxLevelFilter(logging.INFO))
        h_out.setFormatter(formatter)
        root.addHandler(h_out)

        h_err = logging.StreamHandler(stream=sys.stderr)
        h_err.setLevel(logging.WARNING)
        h_err.setFormatter(formatter)
        root.addHandler(h_err)

    if to_file:
        fh = logging.FileHandler(to_file, encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
        root.addHandler(fh)

    for noisy in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(max(numeric_level, logging.WARNING))

    setup_logging._configured = True

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
