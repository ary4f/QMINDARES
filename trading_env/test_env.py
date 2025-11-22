import numpy as np
from trading_env.trading_env import TradingEnv

# Fake data for testing
R = np.random.randn(200, 3) * 0.01  # small returns
dates = list(range(200))
assets = ["AAPL", "MSFT", "GOOG"]

env = TradingEnv(R, dates, assets, window=20)

obs, info = env.reset()
print("Initial obs shape:", obs.shape)

for i in range(10):
    action = np.random.randn(env.A)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: reward={reward}, terminated={terminated}")
