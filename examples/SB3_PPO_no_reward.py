
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
import babybench.utils as bb_utils

def test(env, save_dir, test_for=1000, model=None, render_video=False):
    """ 

    """
    obs, _ = env.reset()
    images = []
    im_counter = 0

    for idx in range(test_for):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, _ = env.step(action)
        if render_video:
            images.append(bb_utils.evaluation_img(env))
        if done or trunc:
            time.sleep(1)
            obs, _ = env.reset()
            
            if render_video:
                bb_utils.evaluation_video(
                    images,
                    os.path.join(save_dir, f'episode_{im_counter}.avi'))
                images = []
                im_counter += 1
    env.reset()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--save_dir', default='results/PPO_no_reward', type=str,
                        help='Directory to save results')
    parser.add_argument('--render_video', default=True, type=bool,
                        help='Renders a video for each episode during the test run')
    parser.add_argument('--train_for', default=100000, type=int,
                        help='Total timesteps of training')
    args = parser.parse_args()
    
    bb_utils.make_dir(args.save_dir)

    with open(args.config) as f:
            config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    env.reset()
    
    model = PPO("MultiInputPolicy", env,
                tensorboard_log=os.path.join(args.save_dir, "tensorboard_logs", config['environment'], 'PPO_no_reward'),
                verbose=1)

    # train model
    model.learn(total_timesteps=args.train_for)
    model.save(os.path.join(args.save_dir, "model"))

    # test
    test(env, args.save_dir, model=model, test_for=1000, render_video=args.render_video)


if __name__ == '__main__':
    main()
