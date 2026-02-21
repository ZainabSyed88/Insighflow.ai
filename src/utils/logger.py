"""Logging configuration and utilities."""

import logging
import logging.handlers
import time
from pathlib import Path
from src.config import settings

# Guard flag so handlers are only attached once per process, even across
# multiple Streamlit reruns that re-import/re-call setup_logging().
_logging_configured = False


def setup_logging():
    """Configure logging for the application (idempotent)."""
    global _logging_configured
    if _logging_configured:
        return logging.getLogger()

    # Create logs directory
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(exist_ok=True, parents=True)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, settings.log_level))

    # Remove any pre-existing handlers (safety net)
    logger.handlers.clear()

    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(getattr(logging, settings.log_level))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter — includes elapsed time placeholder via a custom filter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    _logging_configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(name)


class StepTimer:
    """Simple context-manager / helper to time and log individual steps.

    Usage:
        timer = StepTimer("TrendAgent")
        with timer.step("Mann-Kendall test"):
            ...   # timed block
        timer.summary()   # logs all step durations
    """

    def __init__(self, agent_name: str, logger: logging.Logger | None = None):
        self.agent_name = agent_name
        self._logger = logger or logging.getLogger(agent_name)
        self._steps: list[tuple[str, float]] = []
        self._step_start: float | None = None
        self._step_name: str = ""

    class _StepCtx:
        def __init__(self, timer: "StepTimer", name: str):
            self.timer = timer
            self.name = name

        def __enter__(self):
            self.timer._step_name = self.name
            self.timer._step_start = time.perf_counter()
            self.timer._logger.info(f"[{self.timer.agent_name}] ▸ {self.name} …")
            return self

        def __exit__(self, *exc):
            elapsed = time.perf_counter() - self.timer._step_start
            self.timer._steps.append((self.name, elapsed))
            self.timer._logger.info(
                f"[{self.timer.agent_name}] ✓ {self.name} done ({elapsed:.2f}s)"
            )
            return False

    def step(self, name: str) -> _StepCtx:
        return self._StepCtx(self, name)

    def summary(self) -> dict[str, float]:
        total = sum(d for _, d in self._steps)
        breakdown = {n: round(d, 2) for n, d in self._steps}
        self._logger.info(
            f"[{self.agent_name}] ── step summary: total={total:.2f}s  {breakdown}"
        )
        return breakdown


def audit(msg: str, **kwargs) -> None:
    """Audit log stub — currently aliases to logger.info.

    Designed to be swapped for a dedicated audit-log backend
    (e.g. append-only DB table, SIEM webhook) in production.
    """
    extra = f" | {kwargs}" if kwargs else ""
    logging.getLogger("audit").info(f"[AUDIT] {msg}{extra}")
