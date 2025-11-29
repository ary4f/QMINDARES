# Alex - PPO Model Notes

- **Algorithm:** Proximal Policy Optimization (PPO) with clipped objective and policy gradient methods
- **Key Hyperparameters:** Learning rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99, clip_range=0.2, entropy_coef=0.01
- **Observation:** PPO converged smoothly on the training set; validation-based early stopping helped prevent overfitting in this financial time series setting
