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
        
    def eval_logs(self):
        try:
            with open(f'{self._save_dir}/logs/training.pkl', 'rb') as f:
                logs = pickle.load(f)
        except:
            raise TypeError(f'Training logs not found -- make sure to use the correct save_dir in the config')
        return self._eval_logs(logs)

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

    def eval_step(self, info):
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

    def _eval_logs(self, logs):
        # Count total unique touches for left and right hands
        # during last 1000 episodes logged
        n_episodes = min(1001, len(logs))
        right_touches = np.array([])
        left_touches = np.array([])
        for ep in range(1,n_episodes):
            right_touches = np.unique(np.concatenate(
                (right_touches, logs[-ep]['right_hand_touches']),0))
            left_touches = np.unique(np.concatenate(
                (left_touches, logs[-ep]['left_hand_touches']),0))
        # maximum of 34 geoms touched per hand
        score = (len(right_touches) + len(left_touches)) / (34*2)
        return score

class EvalHandRegard(Eval):

    def __init__(self, **kwargs):
        super(EvalHandRegard, self).__init__(**kwargs)

        self._track_bodies = ['head','left_eye','right_eye','right_hand','left_hand']
        self._init_track()

    def _eval_logs(self, logs):
        # Average time looking at hands
        # during last 1000 episodes logged
        n_episodes = min(1001, len(logs))
        hand_in_view = 0
        steps = 0
        for ep in range(1,n_episodes):
            hand_in_view += logs[-ep]['right_eye_right_hand']
            hand_in_view += logs[-ep]['left_eye_right_hand']
            hand_in_view += logs[-ep]['right_eye_left_hand']
            hand_in_view += logs[-ep]['left_eye_left_hand']
            steps += logs[-ep]['steps']
        score = hand_in_view / (4 * steps)
        return score

EVALS = {
    'self_touch' : EvalSelfTouch,
    'hand_regard' : EvalHandRegard,
}