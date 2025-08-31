import numpy as np
import os
import gymnasium as gym
import time
import argparse
import mujoco
import yaml

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as env_utils
import babybench.utils as bb_utils
import babybench.eval as bb_eval

from stable_baselines3 import PPO

from scipy.ndimage import convolve

def simple_saliency(rgb_img):
    gray_img = bb_utils.to_grayscale(rgb_img)
    # Define a simple Laplacian kernel
    laplacian_kernel = np.array([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]])
    # Apply the kernel using convolution
    edges = convolve(gray_img, laplacian_kernel, mode='reflect')
    # Compute energy as sum of squared edge intensities (normalized)
    energy = np.sqrt(np.sum(edges**2)) / (gray_img.shape[0] * gray_img.shape[1])
    return energy


class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def compute_intrinsic_reward(self, obs):
        return simple_saliency(obs['eye_left']) + simple_saliency(obs['eye_right'])

    def step(self, action):
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        intrinsic_reward = self.compute_intrinsic_reward(obs)
        total_reward = intrinsic_reward + extrinsic_reward # extrinsic reward is always 0  
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='examples/config_test_installation.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--render', default=True,  type=bool,
                        help='Renders a video for each episode during the evaluation.')
    parser.add_argument('--duration', default=1000, type=int,
                        help='Total timesteps per evaluation episode')
    parser.add_argument('--episodes', default=10, type=int,
                        help='Number of evaluation episode')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = bb_utils.make_env(config, training=False)
    wrapped_env = Wrapper(env)
    wrapped_env.reset()

    model = PPO.load("results/hand_regard/model.zip", env=wrapped_env, device="auto")  # keep your path

    evaluation = bb_eval.EVALS[config['behavior']](
        env=wrapped_env,
        duration=args.duration,
        render=args.render,
        save_dir=config['save_dir'],
    )

    evaluation.eval_logs()

    for ep_idx in range(args.episodes):
        print(f'Running evaluation episode {ep_idx+1}/{args.episodes}')
        obs, _ = wrapped_env.reset()
        evaluation.reset()

        for t_idx in range(args.duration):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, info = wrapped_env.step(action)
            evaluation.eval_step(info)

            
        evaluation.end(episode=ep_idx)


if __name__ == '__main__':
    main()
