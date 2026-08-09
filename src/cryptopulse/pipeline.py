"""Command-line entry point: fetch, engineer features, and persist to SQLite.

Run locally, where you have normal internet access:

    pip install -e .
    python -m cryptopulse.pipeline
"""

from __future__ import annotations

import logging

from cryptopulse.config import settings
from cryptopulse.data.features import engineer_features
from cryptopulse.data.fetch import fetch_ohlcv
from cryptopulse.data.persistence import save_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run() -> None:
    """Fetch raw OHLCV data, engineer features/labels, and persist to SQLite."""
    raw = fetch_ohlcv(settings.ticker, settings.start_date, settings.end_date)
    logger.info("Fetched %d raw rows for %s", len(raw), settings.ticker)

    features = engineer_features(
        raw, short_window=settings.short_window, long_window=settings.long_window
    )
    dropped = len(raw) - len(features)
    logger.info(
        "Engineered %d feature rows (dropped %d to warm-up/label NaNs)", len(features), dropped
    )

    save_features(features, settings.db_path, settings.features_table)
    logger.info("Saved to %s (table=%s)", settings.db_path, settings.features_table)

    balance = features["target_up"].mean()
    logger.info("Class balance: %.1f%% up-days -- check this before training", balance * 100)


if __name__ == "__main__":
    run()
