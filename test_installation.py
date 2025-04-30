"""
Installation test for BabyBench. Checks required libraries, MIMo, simulations, and rendering
"""

def main():
    
    print("Starting BabyBench installation test")

    print("Checking libraries... ", end='')
    try:
        import numpy as np
        import os
        import gymnasium as gym
        import time
        import argparse
        #import cv2
        import mujoco
        import yaml
    except ImportError as e:
        print("Please check that the necessary libraries are installed")
        print(e)
        exit(1)
    print("OK")

    print("Checking MIMo... ", end='')
    try:
        import mimoEnv
        from mimoEnv.envs.mimo_env import MIMoEnv
        import mimoEnv.utils as env_utils
        import babybench.utils as bb_utils
    except ImportError as e:
        print("Please make sure you have MIMo installed")
        print(e)
        exit(1)
    print("OK")

    print("Checking config... ", end='')
    try:
        with open('config.yml') as f:
            config = yaml.safe_load(f)
            for param in ['save_dir','behavior','actuation_model',
                          'max_episode_steps','frame_skip','render_size','save_logs_every']:
                temp = config[param]
    except Exception as e:
        print(f"Please make sure the necessary parameters are defined in the config. Missing: {param}")
        print(e)
        exit(1)
    print("OK")

    print("Checking environment initialization... ", end='')
    try:
        env = bb_utils.make_env(config)
        env.reset()
    except Exception as e:
        print("Error creating environment")
        print(e)
        exit(1)
    print("OK")

    print("Checking environment simulation... ", end='')
    try:
        _ = env.step(env.action_space.sample())
    except Exception as e:
        print("Error taking step in environment")
        print(e)
        exit(1)
    print("OK")

    print("Checking image rendering... ", end='')
    try:
        images = []
        for _ in range(100):
            _ = env.step(env.action_space.sample())
            images.append(bb_utils.render(env))
    except Exception as e:
        print("Error rendering environment")
        print(e)
        exit(1)
    print("OK")

    print("Checking video rendering... ", end='')
    try:
        bb_utils.evaluation_video(images, save_name='test_installation.avi', resolution=(480,480))
    except Exception as e:
        print("Error creating video")
        print(e)
        exit(1)
    print("OK")

    print("Installation test successful!")

if __name__ == '__main__':
    main()