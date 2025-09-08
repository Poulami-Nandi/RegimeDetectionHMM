# Stress Test Report — TSLA

**Purpose:** Simulate a stressed market by modifying daily returns, then check how the model’s
**regime probabilities** differ from baseline.

## Inputs
- **Ticker:** `TSLA`
- **Shock:** `-0.05` — Additive daily return shock (e.g., `-0.05` = -5% each day).
- **Volatility scale:** `1.5` — Multiplier on typical day-to-day move size.

## Output
- **Average Jensen–Shannon drift (`js_drift_avg`):** `0.148178`
  - 0 = identical distributions; 1 = completely different.
- **Threshold:** `0.25`
- **Alert:** `False`

## Notes
- This script **re-fits** an HMM on the shocked series, then compares its regime probabilities to the baseline.  
- For **fixed-model** stress (no re-fit), score the shocked features using the baseline model instead; that usually yields larger drift under severe shocks.
