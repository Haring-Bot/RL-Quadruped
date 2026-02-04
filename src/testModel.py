# test_model.py
import gymnasium as gym
import numpy as np
import pybullet as p
import time
from stable_baselines3 import PPO
from train import QuadrupedEnv  # Import your environment

# Load the trained model
model_path = "./models/quadruped_ppo_20260127_101350"  # Your model path (without .zip)
model = PPO.load(model_path)

print(f"Loaded model from {model_path}")

# Create environment with rendering
env = QuadrupedEnv(render_mode=True)
obs, _ = env.reset()

print("Running trained model...")
print("Press Ctrl+C to stop")

try:
    while True:
        # Predict action (deterministic=True for testing)
        action, _states = model.predict(obs, deterministic=True)
        
        # Execute action
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Slow down for visualization
        time.sleep(1/30.)  # 30 FPS
        
        # Reset if episode ends
        if terminated or truncated:
            print(f"Episode ended. Reward: {reward:.2f}")
            obs, _ = env.reset()
            time.sleep(1.0)
            
except KeyboardInterrupt:
    print("\nStopping...")
    env.close()