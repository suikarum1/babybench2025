
""" Example: Stable-baselines3 PPO with no reward
"""
import numpy as np
import os
import gymnasium as gym
import time
import argparse
import cv2
import mujoco
import yaml
from stable_baselines3 import PPO

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as env_utils

import sys
sys.path.append(".")
sys.path.append("..")
import babybench.utils as bb_utils


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_selftouch.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--train_for', default=10000, type=int,
                        help='Total timesteps of training')
    args = parser.parse_args()
    
    with open(args.config) as f:
            config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    env.reset()
    
    model = PPO("MultiInputPolicy", env, verbose=1)

    # train model
    model.learn(total_timesteps=args.train_for)
    model.save(os.path.join(config["save_dir"], "model"))

if __name__ == '__main__':
    main()
