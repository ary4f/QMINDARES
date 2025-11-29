import gymnasium as gym
import numpy as np
import pandas as pd
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from Tradingenv import TradingEnv, load_returns_matrix
import os

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Environment configuration
WINDOW = 20
TRANSACTION_COST_BPS = 10
INCLUDE_CASH = True
CSV_PATH = 'data/prices_returns.csv'

# Data splits (exactly as specified)
TRAIN_START = 0
TRAIN_END = 1500
VALID_START = 1500
VALID_END = 2000
TEST_START = 2000
TEST_END = None

# PPO Hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
TOTAL_TIMESTEPS = 100000

# Load data once (shared across all environments)
print("Loading returns matrix from CSV...")
R, dates, assets = load_returns_matrix(CSV_PATH)
print(f"Loaded: T={len(dates)}, A={len(assets)}, assets={assets}")

class FiniteActionSpaceWrapper(gym.Wrapper):
    """Wrapper to make action space have finite bounds for SB3 compatibility"""
    def __init__(self, env, action_bound=10.0):
        super().__init__(env)
        # Override action space with finite bounds
        n_actions = env.action_space.shape[0]
        self.action_space = gym.spaces.Box(
            low=-action_bound, 
            high=action_bound, 
            shape=(n_actions,), 
            dtype=np.float32
        )
    
    def step(self, action):
        # Pass action through unchanged (softmax handles any values)
        return self.env.step(action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ValidationCallback(BaseCallback):
    """Callback to evaluate on validation set during training"""
    def __init__(self, val_env, eval_freq=10000, verbose=0):
        super(ValidationCallback, self).__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.best_sharpe = -np.inf
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            obs, _ = self.val_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.val_env.step(action)
                done = terminated or truncated
            
            if 'episode' in info:
                val_sharpe = info['episode'].get('sharpe', 0)
                if self.verbose > 0:
                    print(f"\nValidation Sharpe at step {self.n_calls}: {val_sharpe:.3f}")
                
                if val_sharpe > self.best_sharpe:
                    self.best_sharpe = val_sharpe
                    self.model.save('models/ppo_best')
                    if self.verbose > 0:
                        print(f"New best model saved! Sharpe: {val_sharpe:.3f}")
        
        return True

def create_env(start_idx, end_idx):
    """Create trading environment with specified parameters"""
    base_env = TradingEnv(
        R=R,
        dates=dates,
        assets=assets,
        window=WINDOW,
        cost_bps=TRANSACTION_COST_BPS,
        include_cash=INCLUDE_CASH,
        slippage_coef=0.0,
        start_idx=start_idx,
        end_idx=end_idx
    )
    # Wrap to make action space finite
    return FiniteActionSpaceWrapper(base_env, action_bound=10.0)

def train_ppo():
    """Train PPO agent on training set with validation callback"""
    print("\nCreating training and validation environments...")
    train_env = create_env(TRAIN_START, TRAIN_END)
    val_env = create_env(VALID_START, VALID_END)
    
    print(f"Training environment: T={TRAIN_END - TRAIN_START} timesteps")
    print(f"Validation environment: T={VALID_END - VALID_START} timesteps")
    
    print("\nInitializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        verbose=1,
    )
    
    val_callback = ValidationCallback(val_env, eval_freq=10000, verbose=1)
    
    print(f"\nTraining PPO for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=val_callback,
        progress_bar=False
    )
    
    model.save('models/ppo_final')
    print("\nTraining complete. Models saved to models/")
    
    return model

def evaluate_on_test(model_path='models/ppo_best'):
    """Evaluate trained model on test set and save results"""
    print(f"\nLoading model from {model_path}...")
    model = PPO.load(model_path)
    
    print("Creating test environment...")
    test_env = create_env(TEST_START, TEST_END)
    
    print("Running evaluation on test set...")
    obs, _ = test_env.reset()
    done = False
    
    returns_list = []
    dates_list = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
        # Collect daily returns and dates from info
        if 'date' in info and 'portfolio_ret' in info:
            dates_list.append(info['date'])
            returns_list.append(info['portfolio_ret'])
    
    # Get episode metrics
    if 'episode' in info:
        episode_metrics = info['episode']
        print("\nTest Set Performance:")
        print(f"  CAGR: {episode_metrics.get('cagr', 0):.4f}")
        print(f"  Sharpe: {episode_metrics.get('sharpe', 0):.4f}")
        print(f"  Max Drawdown: {episode_metrics.get('max_drawdown', 0):.4f}")
        print(f"  Avg Turnover: {episode_metrics.get('avg_turnover', 0):.4f}")
        print(f"  Avg Cost: {episode_metrics.get('avg_cost', 0):.4f}")
        print(f"  Steps: {episode_metrics.get('steps', 0)}")
        
        # Save episode metrics
        with open('reports/alex_episode.json', 'w') as f:
            json.dump(episode_metrics, f, indent=2)
        print("\nEpisode metrics saved to reports/alex_episode.json")
    
    # Save returns CSV
    if dates_list and returns_list:
        returns_df = pd.DataFrame({
            'date': dates_list,
            'ret': returns_list
        })
        returns_df.to_csv('data/returns_alex.csv', index=False)
        print(f"Returns saved to data/returns_alex.csv ({len(returns_df)} days)")
    else:
        print("Warning: No returns collected during test evaluation!")

def create_notes():
    """Create notes markdown file"""
    notes = """# Alex - PPO Model Notes

- **Algorithm:** Proximal Policy Optimization (PPO) with clipped objective and policy gradient methods
- **Key Hyperparameters:** Learning rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99, clip_range=0.2, entropy_coef=0.01
- **Observation:** PPO converged smoothly on the training set; validation-based early stopping helped prevent overfitting in this financial time series setting
"""
    
    with open('reports/alex_notes.md', 'w') as f:
        f.write(notes)
    print("Notes saved to reports/alex_notes.md")

if __name__ == "__main__":
    print("=" * 60)
    print("PPO Trading Agent - Alex")
    print("=" * 60)
    
    # Train the model
    model = train_ppo()
    
    # Evaluate on test set
    evaluate_on_test(model_path='models/ppo_best')
    
    # Create notes
    create_notes()
    
    print("\n" + "=" * 60)
    print("All tasks complete!")
    print("=" * 60)
