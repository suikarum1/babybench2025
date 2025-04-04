import os
import numpy as np
import gymnasium as gym
import mujoco
import cv2
import yaml

ENVS = {
    'self_touch': 'BabyBench-SelfTouch',
    'hand_regard': 'BabyBench-HandRegard',
    'mirror': 'BabyBench-Mirror',
}

def make_env(config=None, training=True):
    make_save_dirs(config['save_dir'])
    env = gym.make(
        ENVS[config['behavior']],
        actuation_model=config['actuation_model'],
        max_episode_steps=config['max_episode_steps'],
        frame_skip=config['frame_skip'],
        width=config['render_size'],
        height=config['render_size'],
        config=config,
        training=training,
    )
    return env

def render(env, camera="corner"):
    img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name=camera)
    return img.astype(np.uint8)

def evaluation_img(env):
    img_corner = render(env, "corner")
    img_top = render(env, "top")
    img_left_eye = render(env, "eye_left")
    img_right_eye = render(env, "eye_right")
    img = np.zeros((480,720,3))
    img[:,:480,:] = img_corner
    img[240:,480:,:] = img_top[::2,::2,:]
    img[:240,480:,0] = to_grayscale(img_left_eye[::2,::2,:])
    img[:240,480:,1] = to_grayscale(img_right_eye[::2,::2,:])
    img[:240,480:,2] = to_grayscale(img_right_eye[::2,::2,:])
    return img.astype(np.uint8)

def evaluation_video(images, save_name=None, frame_rate=60, resolution=((720,480))):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_name, fourcc, frame_rate, resolution)
    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()
    video.release()

def to_grayscale(x):
    return 0.2989*x[:,:,0] + 0.5870*x[:,:,1] + 0.1140*x[:,:,2]

def make_save_dirs(save_dir):
    make_dir(save_dir)
    dirs = ['logs','trajectories','videos']
    for dir_name in dirs:
        make_dir(f'{save_dir}/{dir_name}')

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)