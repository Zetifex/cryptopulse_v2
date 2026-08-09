# 2. Predict next-day direction (classification), not next-day price (regression)

## Status
Accepted

## Context
The original engine.py predicted raw `Close_Price` directly from a plain
MLP. Because crypto prices trend over long periods, a model can achieve
deceptively low MSE by learning to output "something near the recent
price" without learning anything that generalizes. This also didn't match
the project's own premise (a moving-average "crossover/pulse" signal),
which is inherently about direction, not price level.

## Decision
Reframe the problem as binary classification: predict whether tomorrow's
close will be higher than today's close (`target_up` in
`src/cryptopulse/data/features.py`). The label is computed with a forward
shift and explicitly nulled where undefined (see the code comment on the
`NaN > x` pandas behavior) rather than left to fall out of a raw
comparison.

## Consequences
- Loss function becomes `BCEWithLogitsLoss`, not `MSELoss`; the model's
  final layer outputs a single logit rather than a price value.
- Accuracy alone is not a sufficient metric -- class balance must be
  checked (`pipeline.py` logs it) and precision/recall/F1/AUC tracked
  alongside accuracy, in case "up days" are skewed by a long-run uptrend
  in the sampled date range.
- The FastAPI response contract (Milestone 4) will return a probability
  and a predicted class, not a predicted price.
- Train/validation/test split must remain strictly chronological; a
  random shuffle would leak future information into training regardless
  of this being a classification vs. regression problem.
