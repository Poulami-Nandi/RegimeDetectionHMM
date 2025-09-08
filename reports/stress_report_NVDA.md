# Stress Test Report — NVDA

**Purpose:** Simulate a stressed market by modifying daily returns, then check how the model’s
**regime probabilities** differ from baseline.

## Inputs
- **Ticker:** `NVDA`
- **Shock:** `-0.05` — Additive daily return shock (e.g., `-0.05` = -5% each day).
- **Volatility scale:** `1.4` — Multiplier on typical day-to-day move size.

## Output
- **Average Jensen–Shannon drift (`js_drift_avg`):** `0.380287`
  - 0 = identical distributions; 1 = completely different.
- **Threshold:** `0.25`
- **Alert:** `True`

## Notes
- This script **re-fits** an HMM on the shocked series, then compares its regime probabilities to the baseline.  
- For **fixed-model** stress (no re-fit), score the shocked features using the baseline model instead; that usually yields larger drift under severe shocks.
