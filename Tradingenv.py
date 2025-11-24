#!/usr/bin/env python3
"""
TradingEnv — merged from your two versions, minimal changes.

- Observation: last `window` days of per-asset returns, shape (window, A)
- Action: logits in R^A -> softmax -> long-only weights (sum to 1)
- Reward: w · r_t  −  (cost_bps/1e4)*0.5*Σ|w_t − w_{t-1}|  −  slippage_coef*Σ(Δw)^2
- Optional CASH sleeve (zero-return asset appended as last column)
- Tiny loader for data/prices_returns.csv and an EQW smoke-rollout

USAGE (after you've built data with your pipeline):
  python env/trading_env.py --csv data/prices_returns.csv --window 20 --cost_bps 10 --include_cash 1 --smoke_rollout
"""

import argparse
import os
from typing import List, Tuple, Optional
import gymnasium as gym

from gymnasium import spaces
import numpy as np
import pandas as pd


# ===== helpers =====
def softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax to map logits -> simplex weights (>=0, sum=1)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x - np.max(x)
    e = np.exp(x)
    w = e / (e.sum() + 1e-12)
    return w.astype(np.float32)


def load_returns_matrix(csv_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Read long table (date, asset, close, ret) and return:
      R: [T, A] float32 returns
      dates: list[str] length T (YYYY-MM-DD)
      assets: list[str] length A
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    # sanity: required columns
    for c in ("date", "asset", "ret"):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")

    piv = df.pivot(index="date", columns="asset", values="ret").sort_index()
    if piv.isna().any().any():
        raise ValueError("NaNs after pivot — calendar not aligned for all assets/dates.")

    R = piv.to_numpy(dtype=np.float32, copy=True)
    dates = [d.strftime("%Y-%m-%d") for d in piv.index]
    assets = list(map(str, piv.columns))
    return R, dates, assets


# ===== env =====
class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        R: np.ndarray,
        dates: List[str],
        assets: List[str],
        window: int = 20,
        cost_bps: float = 10.0,
        include_cash: bool = True,
        slippage_coef: float = 0.0,
        # --- NEW: simple slicing for train/val/test ---
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ):
        """
        R            : [T, A] daily returns (float32)
        dates        : list[str] length T
        assets       : list[str] length A
        window       : lookback days used in observation
        cost_bps     : transaction cost in basis points (e.g., 10 -> 0.10%)
        include_cash : append zero-return CASH asset if True
        slippage_coef: quadratic penalty on weight changes (default 0.0)
        start_idx/end_idx: slice the timeline for train/val/test
        """
        super().__init__()

        # --- slice for train/valid/test (MINIMAL ADDITION) ---
        T_total = R.shape[0]
        if end_idx is None:
            end_idx = T_total
        if not (0 <= start_idx < end_idx <= T_total):
            raise ValueError(f"Bad slice: start_idx={start_idx}, end_idx={end_idx}, T={T_total}")

        R = R[start_idx:end_idx].copy()
        dates = dates[start_idx:end_idx]

        assert R.ndim == 2, "R must be [T, A]"
        assert len(dates) == R.shape[0], "dates length must equal T"
        assert R.shape[0] > window, "Need T > window"
        if not np.isfinite(R).all():
            raise ValueError("R contains NaN/Inf.")

        self.R = R
        self.dates = list(dates)
        self.assets = list(assets)
        self.window = int(window)
        self.cost = float(cost_bps) / 1e4  # bps -> decimal
        self.slippage_coef = float(slippage_coef)

        # Optional CASH sleeve
        if include_cash:
            cash_col = np.zeros((self.R.shape[0], 1), dtype=np.float32)
            self.R = np.concatenate([self.R, cash_col], axis=1)
            self.assets = self.assets + ["CASH"]

        self.A = self.R.shape[1]
        self.t = self.window
        self.prev_w = np.ones(self.A, dtype=np.float32) / self.A

        # Gym spaces (same as your teammate's)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.A,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window, self.A), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.t = self.window
        self.prev_w = np.ones(self.A, dtype=np.float32) / self.A

        # --- NEW: episode trackers (MINIMAL ADDITION) ---
        self.ep_raw = []        # raw portfolio returns (w·r_t)
        self.ep_turnover = []   # Σ|Δw|
        self.ep_costs = []      # daily transaction cost
        self.eq = [1.0]         # equity curve (net of costs): *= (1 + reward)

        obs = self.R[self.t - self.window : self.t, :]
        return obs.astype(np.float32), {}

    def step(self, action: np.ndarray):
        # logits -> weights
        w = softmax(action)
        if w.shape[0] != self.A:
            raise ValueError(f"Action wrong shape {w.shape}, expected {(self.A,)}")

        # today's return & raw portfolio return
        r_t = self.R[self.t, :]
        reward_raw = float(np.dot(w, r_t))

        # turnover & costs (half-spread convention)
        delta = w - self.prev_w
        turnover = float(np.sum(np.abs(delta)))
        cost = self.cost * 0.5 * turnover

        # optional slippage (kept at 0.0 by default)
        slippage = self.slippage_coef * float(np.sum(delta * delta))

        reward = reward_raw - cost - slippage

        # --- NEW: track episode stats (MINIMAL ADDITION) ---
        self.ep_raw.append(reward_raw)
        self.ep_turnover.append(turnover)
        self.ep_costs.append(cost)
        self.eq.append(self.eq[-1] * (1.0 + reward))

        # advance
        self.prev_w = w
        self.t += 1
        terminated = self.t >= self.R.shape[0]
        truncated = False

        if not terminated:
            next_obs = self.R[self.t - self.window : self.t, :]
        else:
            # zeros is SB3-friendly; final obs is ignored anyway
            next_obs = np.zeros((self.window, self.A), dtype=np.float32)

        info = {
            "date": self.dates[self.t - 1] if (self.t - 1) < len(self.dates) else None,
            "weights": w,
            "raw_return": reward_raw,
            "turnover": turnover,
            "transaction_cost": cost,
            "slippage": slippage,
            "portfolio_ret": reward,
        }

        # --- NEW: summarize at episode end (MINIMAL ADDITION) ---
        if terminated and len(self.ep_raw) > 0:
            r = np.array(self.ep_raw, dtype=np.float64)
            eq = np.array(self.eq, dtype=np.float64)  # length = steps+1
            # max drawdown on equity (net of costs)
            dd_series = eq / np.maximum.accumulate(eq) - 1.0
            max_dd = float(dd_series.min())
            ann = 252.0
            cagr = float(eq[-1] ** (ann / max(1, len(r))) - 1.0)
            sharpe = float((r.mean() / (r.std(ddof=1) + 1e-12)) * np.sqrt(ann))
            info["episode"] = {
                "cagr": cagr,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "avg_turnover": float(np.mean(self.ep_turnover)),
                "avg_cost": float(np.mean(self.ep_costs)),
                "steps": int(len(r)),
                "start_date": self.dates[0] if self.dates else None,
                "end_date": self.dates[self.t - 1] if (self.t - 1) < len(self.dates) else None,
            }

        return next_obs.astype(np.float32), float(reward), terminated, truncated, info


# ===== utilities =====
def equal_weight_policy(n_assets: int) -> np.ndarray:
    """Logits that lead to ~equal weights after softmax."""
    return np.ones(n_assets, dtype=np.float32)


def rollout_eqw(env: TradingEnv, out_csv: str = "data/returns_EQW.csv") -> None:
    """Run equal-weight from start to finish and write (date, ret) CSV."""
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    obs, _ = env.reset(seed=42)
    rows, done = [], False
    while not done:
        action = equal_weight_policy(env.A)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append((info["date"], reward))
        done = terminated or truncated
    pd.DataFrame(rows, columns=["date", "ret"]).to_csv(out_csv, index=False)
    print(f"[EQW] saved -> {out_csv}")


# ===== CLI =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/prices_returns.csv")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--cost_bps", type=float, default=10.0)
    ap.add_argument("--include_cash", type=int, default=1)  # 1 true, 0 false
    ap.add_argument("--slippage_coef", type=float, default=0.0)
    ap.add_argument("--smoke_rollout", action="store_true")
    # NEW: expose slice args (optional)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=-1)
    args = ap.parse_args()

    R, dates, assets = load_returns_matrix(args.csv)

    end_idx = None if args.end_idx == -1 else args.end_idx
    env = TradingEnv(
        R, dates, assets,
        window=args.window,
        cost_bps=args.cost_bps,
        include_cash=bool(args.include_cash),
        slippage_coef=args.slippage_coef,
        start_idx=args.start_idx,
        end_idx=end_idx,
    )

    if args.smoke_rollout:
        rollout_eqw(env)
    else:
        print(
            f"Env ready: T={R.shape[0]}, A={env.A}, window={args.window}, "
            f"cost_bps={args.cost_bps}, include_cash={bool(args.include_cash)}"
        )


if __name__ == "__main__":
    main()
