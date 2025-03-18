import os
import numpy as np
import gymnasium as gym
import mujoco
import cv2
import yaml

ENVS = {
    'base': 'BabyBench-Base-v0',
    'crib': 'BabyBench-Crib-v0',
    'mirror': 'BabyBench-Crib-v0',
}

def make_env(config=None):
    env = gym.make(
        ENVS[config['environment']],
        actuation_model=config['actuation_model'],
        max_episode_steps=config['max_episode_steps'],
        behavior=config['behavior'],
        width=config['render_width'],
        height=config['render_height'],
        proprio_velocity=config['proprio_velocity'],
        proprio_torque=config['proprio_torque'],
        proprio_limits=config['proprio_limits'],
        proprio_actuation=config['proprio_actuation'],
        vestibular_active=config['vestibular_active'],
        vision_active=config['vision_active'],
        vision_resolution=config['vision_resolution'],
        touch_active=config['touch_active'],
        touch_scale=config['touch_scale'],
        touch_function=config['touch_function'],
        touch_response_function=config['touch_response_function'],
        touch_left_toes=config['touch_left_toes'],
        touch_right_toes=config['touch_right_toes'],
        touch_left_foot=config['touch_left_foot'],
        touch_right_foot=config['touch_right_foot'],
        touch_left_lower_leg=config['touch_left_lower_leg'],
        touch_right_lower_leg=config['touch_right_lower_leg'],
        touch_left_upper_leg=config['touch_left_upper_leg'],
        touch_right_upper_leg=config['touch_right_upper_leg'],
        touch_hip=config['touch_hip'],
        touch_lower_body=config['touch_lower_body'],
        touch_upper_body=config['touch_upper_body'],
        touch_head=config['touch_head'],
        touch_left_eye=config['touch_left_eye'],
        touch_right_eye=config['touch_right_eye'],
        touch_left_upper_arm=config['touch_left_upper_arm'],
        touch_right_upper_arm=config['touch_right_upper_arm'],
        touch_left_lower_arm=config['touch_left_lower_arm'],
        touch_right_lower_arm=config['touch_right_lower_arm'],
        touch_left_hand=config['touch_left_hand'],
        touch_right_hand=config['touch_right_hand'],
        touch_left_fingers=config['touch_left_fingers'],
        touch_right_fingers=config['touch_right_fingers'],
    )
    return env

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

def to_grayscale(x):
    return 0.2989*x[:,:,0] + 0.5870*x[:,:,1] + 0.1140*x[:,:,2]

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)