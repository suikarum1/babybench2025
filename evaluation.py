"""

"""

import numpy as np
import os
import gymnasium as gym
import time
import argparse
import cv2
import mujoco
import yaml

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as env_utils
import utils as bb_utils


def visualize(env, test_for, save_dir, render_video=False):
    obs, _ = env.reset()
    images = []
    im_counter = 0

    for idx in range(test_for):

        action = env.action_space.sample()

        # ------------------------------
        #
        # TODO REPLACE WITH YOUR TRAINED POLICY HERE:
        # action = policy(obs)
        #
        # ------------------------------

        obs, rew, done, trunc, info = env.step(action)
        
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

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--test_for', default=1000, type=int,
                        help='Total timesteps of testing')
    parser.add_argument('--save_dir', default='results/', type=str,
                        help='Directory to save results')
    parser.add_argument('--render_video', default=True, type=bool,
                        help='Renders a video for each episode during the test run')
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    env.reset()

    visualize(env, args.test_for, args.save_dir, render_video=args.render_video)
    


if __name__ == '__main__':
    main()
