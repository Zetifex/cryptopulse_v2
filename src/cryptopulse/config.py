"""Centeralized, typed configuration for CryptoPulse.

All previously-hardcoded constants (db path, window sizes, epochs, lr)
live here instead of being scattered across scripts. Override any value
via environment variables prefixed with CRYPTOPULSE_,e.g.
CRYPTOPULSE_TICKER=ETH-USD."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict  # corrected


class Settings(BaseSettings):
    """Runtime Configuration for CryptoPulse pipeline."""

    model_config = SettingsConfigDict(env_prefix="CRYPTOPULSE_")

    ticker: str = "BTC-USD"
    start_date: str = "2019-01-01"
    end_date: str | None = None  # None -> fetch through today

    short_window: int = 10
    long_window: int = 30
    sequence_length: int = 30

    train_frac: float = 0.70
    val_frac: float = 0.15  # remainder (0.15) is the test split

    db_path: str = "cryptopulse.db"
    features_table: str = "market_features"

    random_seed: int = 42

    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    grad_clip_norm: float = 1.0


settings = Settings()
