# 3. Use stationary ratio features instead of absolute moving averages

## Status
Accepted

## Context
The initial feature set used `short_ma` and `long_ma` directly -- absolute
BTC price levels. Diagnostic on real data (2019-2026) showed severe
distribution shift after standardizing on training statistics only:

| feature | train range | test range |
|---|---|---|
| short_ma | -1.34 to 2.45 | 1.91 to 5.40 |
| long_ma | -1.34 to 2.44 | 2.08 to 5.28 |

Test inputs reached 5.4 standard deviations above the training mean, with
roughly 85% of the test range beyond anything seen in training. Neural
networks extrapolate poorly outside their training distribution, so the
model would effectively be guessing on the test set.

`daily_return` and `volatility` showed no shift -- their test ranges sat
inside the training ranges. Price *levels* are non-stationary; price
*changes* are stationary.

## Decision
Replace the two absolute moving-average features with scale-invariant
ratios:

- `ma_ratio = short_ma / long_ma - 1.0`
- `price_to_long_ma = Close / long_ma - 1.0`

`short_ma` and `long_ma` remain in the persisted DataFrame for inspection
and plotting, but `FEATURE_COLUMNS` -- the model input contract -- excludes
them.

## Consequences
- Feature count stays at 4, so LSTM `input_size` is unchanged.
- Verified on a simulated series spanning $4k to $112k: 100% of test values
  fell outside the training range using the old features, 0% using the new
  ones.
- `ma_ratio` is positive exactly when the short MA exceeds the long MA,
  making it a direct encoding of the moving-average crossover signal that
  motivated the original project.
- The SQLite features table must be regenerated (`make pipeline`) before
  training; any previously saved database has the old schema.