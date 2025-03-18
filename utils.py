import os
import numpy as np
import gymnasium as gym
import mujoco
import cv2
import yaml


ENVS = {
    'crib': 'BabyBench-Crib-v0',
    'mirror': 'BabyBench-Crib-v0',
}

def make_env(config=None):
    env = gym.make(
        ENVS[config['environment']],
        actuation_model=config['actuation_model'],
        max_episode_steps=config['max_episode_steps'],
    )
    return env

def to_grayscale(x):
    return 0.2989*x[:,:,0] + 0.5870*x[:,:,1] + 0.1140*x[:,:,2]

def evaluation_img(env):
    img_corner = env.mujoco_renderer.render(render_mode="rgb_array", camera_name="corner")
    img_top = env.mujoco_renderer.render(render_mode="rgb_array", camera_name="top")
    img_left_eye = env.mujoco_renderer.render(render_mode="rgb_array", camera_name="eye_left")
    img_right_eye = env.mujoco_renderer.render(render_mode="rgb_array", camera_name="eye_right")
    img = np.zeros((480,720,3))
    img[:,:480,:] = img_corner
    img[240:,480:,:] = img_top[::2,::2,:]
    img[:240,480:,0] = to_grayscale(img_left_eye[::2,::2,:])
    img[:240,480:,1] = to_grayscale(img_right_eye[::2,::2,:])
    img[:240,480:,2] = to_grayscale(img_right_eye[::2,::2,:])
    return img.astype(np.uint8)

def evaluation_video(images, save_name=None):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_name, fourcc, 60, (720, 480))
    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()
    video.release()