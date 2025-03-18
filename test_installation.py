"""

"""
def main():
    
    print("Checking libraries... ", end='')
    try:
        import numpy as np
        import os
        import gymnasium as gym
        import time
        import argparse
        import cv2
        import mujoco
        import yaml
    except ImportError as e:
        print("Please check that the necessary libraries are installed")
        print(e)
        exit(1)

    try:
        import mimoEnv
        from mimoEnv.envs.mimo_env import MIMoEnv
        import mimoEnv.utils as env_utils
        import utils as bb_utils
    except ImportError as e:
        print("Please make sure you have MIMo installed")
        print(e)
        exit(1)

        
    try:
        with open('config.yml') as f:
            config = yaml.safe_load(f)
            for param in ['environment','actuation_model','self_touch','hand_regard','mirror']:
                temp = config[param]
    except Exception as e:
        print(f"Please make sure the necessary parameters are defined in the config. Missing: {param}")
        print(e)
        exit(1)
    
    print("OK")
    print("Checking environment... ", end='')
    
    try:
        env = bb_utils.make_env(config)
        env.reset()
    except Exception as e:
        print("Error creating environment")
        print(e)
        exit(1)


    images = []
    try:
        _ = env.step(env.action_space.sample())
    except Exception as e:
        print("Error taking step in environment")
        print(e)
        exit(1)

    print("OK")
    print("Checking rendering... ", end='')

    try:
        images.append(bb_utils.evaluation_img(env))
    except Exception as e:
        print("Error rendering environment")
        print(e)
        exit(1)

    try:
        bb_utils.evaluation_video(images, save_name='test.avi')
        os.remove("test.avi") 
    except Exception as e:
        print("Error creating video")
        print(e)
        exit(1)
    
    print("OK")
    print("Installation test successful!")

if __name__ == '__main__':
    main()