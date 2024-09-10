# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:56:07 2024

@author: Naina Said
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 17:14:37 2024

@author: wineuser
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:44:32 2024

@author: nas
"""

import gymnasium as gym
import time
from gymnasium import Env
from gymnasium.spaces import Box
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from aug_update import reverse_norm_voltages, fast_adjust_Simion, write_fly, fly_Simion, sort_simion_result, scaling_factors_real, eval_sf
import logging
from stable_baselines3.common.env_util import make_vec_env
#import wandb
from stable_baselines3.common.monitor import Monitor

# Initialize Weights & Biases with the correct entity
#import uuid

#api_key = "5b7cd5596d9337ad9137a4dacf4fbf1a92809677"

# Login to Weights & Biases
#wandb.login(key=api_key)
#run_id = str(uuid.uuid4())  # Generate a unique ID
#print(f"First instance run ID: {run_id}")
#wandb.init(project="results-with-restricited-actions", entity="nainasaid2015", name=f"run-{run_id}")# Define paths
grade_log_path = "grade_log.txt"
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

eval_interval = 5

# Custom Environment

class CusEnv(Env):
    
    def __init__(self):
        self.action_space = Box(low=0, high=1, shape=(19,), dtype=np.float32)
        self.observation_space = Box(low=0, high=1, shape=(19,), dtype=np.float32)
        self.episode_length = 25
        self.initial_setting = [0.18, 0.8, 0.8, 0.3, 0.11, 0.2, 0.52, 0.22, 0.12, 0.072, 0.405, 0.04, 0.02, 0.175, 0.048, 0.015, 0.047, 0.037, 0.1]
        self.initial_state = np.array(self.initial_setting)  # Convert to NumPy array
        self.state = np.copy(self.initial_state)
        self.negative_streak = 0  # Track consecutive negative grades
        
    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self.state = np.copy(self.initial_state)
        self.episode_length = 25
        self.negative_streak = 0
        return self.state, {}  # Return state and an empty dictionary
    
    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def render(self):
        pass

    def step(self, action):
        # Ensure the first and last values remain the same
        action[0] = self.initial_setting[0]
        action[-1] = self.initial_setting[-1]

        # Limit the range of adjustments for other parameters
        adjustment_factor = 0.1  # Adjust this factor to limit the deviation from initial values
        scaled_action = self.initial_state + adjustment_factor * (action - 0.5)

        # Ensure the scaled actions stay within reasonable bounds close to the initial state
        scaled_action = np.clip(scaled_action, self.initial_state * 0.95, self.initial_state * 1.05)

        # Apply the adjusted action to the environment
        fast_adjust_Simion(scaled_action)
        inputfilenames = write_fly(2000,spotsize=1, max_angle=10)
        fly_Simion(inputfilenames)
        results, ratiohits = sort_simion_result('SIMIONresults_temp.txt')
        sf = scaling_factors_real(results)
        grade = eval_sf(sf, ratiohits, showplot=False)
        elapsed_time = time.time() - start_time
        self.log_grade(grade, elapsed_time)
        #wandb.log({"grade": grade})

        if grade == -1:
            self.negative_streak += 1
        else:
            self.negative_streak = 0

        if self.negative_streak >= 5:
            self.state = np.copy(self.initial_state)  # Reset state to initial settings
            self.negative_streak = 0
            print("Resetting state due to negative streak.")
            print(f"Reset state to: {self.state}")

        reward = grade
        self.episode_length -= 1
        done = self.episode_length <= 0
        truncated = False  # No truncation logic in this environment
        return self.state, reward, done, truncated, {}  # Return state, reward, done, truncated, and info

    def log_grade(self, grade, elapsed_time):
        with open(grade_log_path, 'a') as log_file:
            log_file.write(f"Grade: {grade}, Time: {elapsed_time} seconds\n")

# Save and load functions
def save_checkpoint(model, save_path, step):
    model.save(save_path)
    with open(save_path + "_step.txt", "w") as f:
        f.write(str(step))

def load_checkpoint(model, load_path):
    model = PPO.load(load_path)
    with open(load_path + "_step.txt", "r") as f:
        step = int(f.read())
    return model, step

# Create monitored environment
def make_env():
    env = CusEnv()
    env = Monitor(env)
    return env

# Create vectorized environment with single environment
#vec_env = DummyVecEnv([make_env for _ in range(3)])

vec_env=make_env()
# Check the action space type
print("VecEnv action space type:", type(vec_env.action_space))

# Initialize PPO model
log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_path, n_steps=30)

# Load checkpoint if exists
checkpoint_file = os.path.join(checkpoint_dir, "ppo_model")
start_step = 0
if os.path.exists(checkpoint_file + ".zip"):
    model, start_step = load_checkpoint(model, checkpoint_file)

start_time = time.time()
logging.info("Starting model learning from step {}".format(start_step))

# Set training parameters
total_timesteps = 2000
checkpoint_interval = 5
eval_interval = 5

# Training loop with evaluation and logging
for step in range(start_step, total_timesteps, checkpoint_interval):
    print(f"Starting training step: {step + checkpoint_interval}")
    model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
    save_checkpoint(model, checkpoint_file, step + checkpoint_interval)
    logging.info("Saved checkpoint at step {}".format(step + checkpoint_interval))
    print(f"Saved checkpoint at step {step + checkpoint_interval}")

    if (step + checkpoint_interval) % eval_interval == 0:
        print(f"Evaluating at step: {step + checkpoint_interval}")
        mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10, render=False)
        logging.info(f"Evaluation at step {step + checkpoint_interval}: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        #wandb.log({
            #"mean_reward": mean_reward,
            #"std_reward": std_reward,
            #"timestep": step + checkpoint_interval
        #})
        #print(f"Logged to Wandb at step {step + checkpoint_interval}")

logging.info("Completed model learning")
#wandb.log({"model_learning": "completed"})

end_time = time.time()
elapsed_time_hours = (end_time - start_time) / 3600

with open('training_time.txt', 'w') as time_file:
    time_file.write(f"Total training time: {elapsed_time_hours} hours\n")

model.save("PPO_model")
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10, render=False)
logging.info(f"Final evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
#wandb.log({
    #"mean_reward": mean_reward,
    #"std_reward": std_reward,
    #"timestep": total_timesteps
#})

with open('run_completion.log', 'a') as log_file:
    log_file.write("Run completed successfully.\n")

vec_env.close()
#wandb.finish()