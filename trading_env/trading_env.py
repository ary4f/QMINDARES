import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, R, dates, assets, window=20, cost_bps=10.0, include_cash=True):
        super().__init__()

        # Store input data
        self.R = R
        self.dates = dates
        self.assets = assets
        self.window = window
        self.include_cash = include_cash

        # Convert cost from bps to decimal (10 bps = 0.001)
        self.cost = cost_bps / 1e4

        # Add CASH as an asset
        if include_cash:
            cash_col = np.zeros((self.R.shape[0], 1))  # cash has 0 return
            self.R = np.concatenate([self.R, cash_col], axis=1)
            self.assets = self.assets + ["CASH"]

        # Number of assets
        self.A = self.R.shape[1]

        # Time index starts at window
        self.t = self.window

        # Previous weights start equal-weight
        self.prev_w = np.ones(self.A) / self.A

        # Define action space
        # Raw action vector (will be softmaxed)
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.A,),
            dtype=np.float32
        )

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window, self.A),
            dtype=np.float32
        )


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset index
        self.t = self.window

        # Reset weights to equal-weight
        self.prev_w = np.ones(self.A) / self.A

        # Observation = last `window` rows
        obs = self.R[self.t - self.window: self.t]

        return obs.astype(np.float32), {}


    def step(self, action):
        # Softmax to convert raw action → weights
        x = action - np.max(action)
        exp_x = np.exp(x)
        w = exp_x / np.sum(exp_x)   # new weights

        # Today's returns
        r_t = self.R[self.t]
        reward_raw = float(np.dot(w, r_t))

        # Transaction cost (half-spread model)
        turnover = np.sum(np.abs(w - self.prev_w))
        cost = self.cost * 0.5 * turnover

        # Slippage penalty (small)
        slippage = 0.01 * np.sum((w - self.prev_w)**2)

        # Final reward
        reward = reward_raw - cost - slippage

        # Update for next step
        self.prev_w = w
        self.t += 1

        # Termination logic
        terminated = self.t >= len(self.R)
        truncated = False  # we aren't using time limits yet

        # Next observation
        if not terminated:
            next_obs = self.R[self.t - self.window: self.t]
        else:
            next_obs = np.zeros((self.window, self.A))

        # Debug info
        info = {
            "date": self.dates[self.t - 1] if self.t - 1 < len(self.dates) else None,
            "weights": w,
            "raw_return": reward_raw,
            "turnover": turnover,
            "transaction_cost": cost,
            "slippage": slippage
        }

        return next_obs.astype(np.float32), float(reward), terminated, truncated, info
