import numpy as np

def softmax(x):
    """Stable softmax implementation (turns any action vector into valid portfolio weights: values >= 0 + sum to 1 + no shorting allowed (long-only) + uses a stable version to avoid numeric overflow)"""
    x = np.array(x)
    x = x - np.max(x)  # improves numerical stability since it subtracts each element in the np.array from the largest element in x
    exp_x = np.exp(x)  # exponentials are used to turn raw scores into metrics with relative importance
    return exp_x / np.sum(exp_x) # divides each exponential by the sum of all exponentials (this makes the output positive, adds to 1, and behaves like a probability distribution)


class TradingEnv:
    def __init__(self, R, dates, assets, window=20, cost_bps=10.0, include_cash=True):
        """
        R: numpy matrix of shape [T, A]
        dates: list of length T
        assets: list of length A
        window: number of past days to include in observation
        cost_bps: transaction cost in basis points (bps) -> 1 bps = 0.01% (which is 0.0001 as a decimal) so 10 bps = 0.1%, 50 = 0.5, 100 = 1%, etc
        """

        self.R = R
        self.dates = dates
        self.assets = assets
        self.window = window

        # Convert cost from bps (e.g., 10) to decimal (e.g., 0.001)
        self.cost = cost_bps / 1e4 # 1bps = 1/10,000

        # Number of assets
        self.A = R.shape[1]

        # Time index: starts at `window`
        # Because the first observation is the last `window` rows of R
        self.t = window

        # Start with equal weights
        self.prev_w = np.ones(self.A) / self.A

    def reset(self, seed=0):
        """
        Reset the environment to the starting point.
            
        Returns:
            obs: numpy array of shape (window, A)
                 The last `window` rows of R starting at day 0.
        """
        np.random.seed(seed)

        # Reset time index
        self.t = self.window # so if window = 20, we begin on day 20 because day 20's observations = rows 0 - 19

        # Reset previous weights to equal weight
        self.prev_w = np.ones(self.A) / self.A

        # Observation = past `window` days of returns
        obs = self.R[self.t - self.window : self.t, :] # eg: ig windows=2, t=2 -> rows 0 and 1 .... obs = R[t-window:t]

        return obs

