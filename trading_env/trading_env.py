import numpy as np
import csv   

def softmax(x):
    """Stable softmax implementation (turns any action vector into valid portfolio weights: values >= 0 + sum to 1 + no shorting allowed (long-only) + large positive actions get more weight + uses a stable version to avoid numeric overflow)"""

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
        window: number of past days to include in observation (last 'n' number of rows to take from the current timestamps available)
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
        obs = self.R[self.t - self.window : self.t, :] # eg: if windows=2, t=2 -> rows 0 and 1 .... obs = R[t-window:t]

        return obs
    
    def step(self, action):
        """
        Take one step (one trading day) in the environment.
        action: raw action vector of length A (the larger the action number, the more the agent wants to invest and so the weight for this should be bigger too after using softmax)
        Returns:
            next_obs: next window of returns (or None if done)
            reward: portfolio return after costs
            done: True when episode is finished
            info: extra debugging info
        """

        w = softmax(action) # converting raw action to valid portfolio weights using softmax 
        r_t = self.R[self.t] # accessing today's asset returns from the matrix R and accessing each row of return values (using self.t since t is time and each row is a different day timestamp)
        reward_raw = np.dot(w, r_t) # raw portfolio returns obtained using dot product -> (weights · returns_today) part of the final reward formula 
        turnover = np.sum(np.abs(w - self.prev_w)) # difference between the new and old weights -> sum(|w_t − w_{t-1}|) part of the final reward formula 
        cost = self.cost * 0.5 * turnover # half on buy, half on sell (also self.cost from the init implementation of the class already has cost_bps/1e4 implemented in it)

        reward = reward_raw - cost # reward final formula 

        self.prev_w = w # for the next step, the previous weight becomes the current weight 
        self.t += 1 # next day
        done = self.t >= len(self.R) # checking if we're now past the last available day

        if not done:
            next_obs = self.R[self.t - self.window: self.t, :]
        else:
            next_obs = None

        # extra debugging info
        info = {
            "date": self.dates[self.t - 1],
            "weights": w,
            "raw_return": reward_raw,
            "turnover": turnover,
            "transaction_cost": cost,
        }

        return next_obs, reward, done, info 
    
def equal_weight_policy(n_assets):
    """
    This function creates a raw action vector where every asset has the same preference (we return ones since softmax will convert it to valid portfolio weights anyways)
    eg: if we have n_assets = 4, then np.ones will return [1,1,1,1] and then softmax will return [0.25,0.25,0.25,0.25]
    """
    return np.ones(n_assets)

def random_policy(n_assets, seed=None):
    """
    This function generates random raw actions for testing the environment
    The difference with this and the equal_weight_policy function is that this returns values that are completely unbonded (negative, large positive -> literally anything) and then softmax takes all that and converts it
    eg: we might have [1.764, 0.400, 0.979, 2.240] as a randomly generated action vector, softmax(action) then takes this and converts it to something. 
    note: the larger the number is (positive), the more weight gets assigned since the agent signifies to put more emphasis on that asset type. so 2.240 will have more weight assigned than the rest, and 0.400 will have the least
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n_assets)

def rollout(env, policy_fn, out_csv="data/returns_EQW.csv"):
    """
    This function runs the environment from start to finish using a given policy function and saves (date, daily returns) into a CSV file
    """
    obs = env.reset()
    done = False # boolean to tell us when the episode is complete i.e. t >= len(R)
    results = [] # this will hold tuples i.e. (date, daily_portfolio_return) so like [ (d1, r1), (d2, r2), ... , (dn, rn)]
    
    while not done:
        action = policy_fn(env.A) # calls policy function. env.A is the number of assets so that it returns an action vector of length 'A'
        next_obs, reward, done, info = env.step(action) # this runs the full trading env step and calculates the portfolio return, trans cost and reward, then advances time and returns updates information 
        results.append((info["date"], reward))
        obs = next_obs

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "ret"])
        for date, ret in results:
            writer.writerow([date, ret])
    
    print(f"Rollout complete. Saved to {out_csv}")



