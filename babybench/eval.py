import numpy as np
import os
import pickle
import babybench.utils as bb_utils

class Eval():

    def __init__(self, env, duration, render, save_dir):
        self._env = env
        self._duration = duration
        self._render = render
        self._save_dir = save_dir
        self._track_bodies = []
        self._images = []
        
    def _init_track(self):
        self._trajectories = {}
        for body in self._track_bodies:
            self._trajectories[body] = {}
    
    def track(self, info):
        for body in self._track_bodies:
            self._trajectories[body][info['steps']] = {
                'pos' : self._env.data.body(body).xpos,
                'mat' : self._env.data.body(body).xmat,
            }

    def render_image(self):
        self._images.append(bb_utils.evaluation_img(self._env))

    def step(self, info):
        self.track(info)
        if self._render:
            self.render_image()

    def end(self, episode=0):
        # Store trajectories for submission
        with open(f'{self._save_dir}/trajectories/episode_{episode}.pkl', 'wb') as f:
            pickle.dump(self._trajectories, f, -1)
        # Store videos for submission
        if self._render:
            bb_utils.evaluation_video(self._images, f'{self._save_dir}/videos/episode_{episode}.avi')

class EvalSelfTouch(Eval):

    def __init__(self, **kwargs):
        super(EvalSelfTouch, self).__init__(**kwargs)

        self._track_bodies = ['right_hand','left_hand']
        self._init_track()


EVALS = {
    'self_touch' : EvalSelfTouch,
}