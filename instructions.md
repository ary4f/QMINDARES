## QMINDARES — Tasks


### quick summary of whats done/changes I made to current code:

Data pipeline:

Enforced full calendar alignment (all assets share the same dates) with a fail-fast assert.

Computed per-asset log returns and dropped the first NaN per asset.

Added strict NaN/Inf checks and standardized column order to date, asset, close, ret.

Cast close/ret to float32 for consistency/perf.

Normalized date dtype and made summary printouts clearer.

Kept optional plotting behind a PLOT flag.

Trading env (merged): 

Unified both envs into one Gymnasium TradingEnv with softmax → long-only weights.

Added train/val/test slicing via start_idx/end_idx (no overlap bugs).

Added episode tracking and summary in info["episode"] (CAGR, Sharpe, max DD, avg turnover).

Optional CASH sleeve (include_cash=True) and slippage penalty (default 0).

Kept cost model as (cost_bps/1e4) * 0.5 * turnover.

Loader now pivots the long CSV to [T, A] returns with strict NaN checks.

Preserved action/obs spaces: action = logits in ℝ^A, obs = (window, A) of past returns.




### Alex, Edan, Jacob  Build RL Models
**Goal:** Each person ships one model on the same splits and saves test returns.

#### Alex - PPO
#### Edan - A2C
#### Jacob - TD3


**What to do**
- Use the same environment settings: window = 20, transaction cost = 10 bps, include cash = true.
- Train on the training split, tune on the validation split, and evaluate on the test split.

**Data splits (use exactly)**
- **Train:** `start_idx=0, end_idx=1500`  
- **Valid:** `start_idx=1500, end_idx=2000`  
- **Test:** `start_idx=2000, end_idx=None`  

**Outputs (names exactly)**
- `data/returns_<yourname>.csv` — daily returns on the **test** set (columns: `date, ret`).
- `reports/<yourname>_episode.json` — episode metrics from the env: `cagr, sharpe, max_drawdown, avg_turnover, steps, start_date, end_date`.
- `reports/<yourname>_notes.md` — 3 bullets (algorithm, key hyperparameters, one observation).

---

## Lukas & Yumna — Rolling Sharpe
**Goal:** Produce a rolling Sharpe series for each model/baseline returns CSV.

**What to do**
- Build a script that reads one or more `data/returns_*.csv` files (schema `date, ret`) and computes **20-day rolling Sharpe** for each series.
- **Require identical dates** across all input files; if dates don’t match, stop with a clear message.

**Outputs**
- `metrics/rolling_sharpe.py` — calculator script.
- `reports/rolling_sharpe_<name>.csv` — columns: `date, sharpe_20d`.

---

## Shaun — Baselines & Summary Table
**Goal:** Generate baseline returns and a single comparison table across all series.

**What to do**
- Implement three baselines on the **test** split:
  - Equal-Weight (long-only)
  - Buy-and-Hold SPY
  - Min-Variance (long-only)
- Save their test returns using the same schema as models (`date, ret`).
- Build a script that reads **all** `data/returns_*.csv` and produces a table with **CAGR, Sharpe(252), volatility, max drawdown**. 
- **Require identical dates** across all inputs before aggregating.

**Outputs**
- `metrics/baselines.py` — generates baseline returns.  
- `metrics/evaluate.py` — produces the comparison table.  
- `data/returns_EQW.csv`, `data/returns_BuyHoldSPY.csv`, `data/returns_MinVar.csv`  
- `reports/summary_table.csv` — one row per model/baseline.  


---

