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
from stable_baselines3 import SAC


class Wrapper(gym.Wrapper):
    """
    Intrinsic reward = prediction disagreement (variance) across an ensemble
    of tiny forward models f_e: [phi(s_t); a_t] -> phi(s_{t+1}).

    - Minimal numpy-only ensemble (no torch), online SGD after each step
    - Uses a random projection phi() to compress 'touch' into d features
    - Reward at step t is computed *before* model update, using (phi_t, a_t)
    - Then we update each model on the observed target phi_{t+1}
    """
    def __init__(self, env, *, ensemble=5, d=64, lr=1e-2, l2=1e-4, boot_p=0.5, eps=1e-6):
    # def __init__(self, env, *, ensemble=5, d=64, lr=1e-3, l2=1e-3, boot_p=0.5, eps=1e-6):
        super().__init__(env)

        # --- feature extraction from obs['touch'] ---
        self.eps = eps
        obs_space = self.env.observation_space
        assert hasattr(obs_space, "spaces") and "touch" in obs_space.spaces, "need MultiInput obs with 'touch'"
        self.M = int(np.prod(obs_space.spaces["touch"].shape)) #68589 with 0.5 touch_scale

        rng = np.random.RandomState(1234)
        # random Gaussian projection for touch -> d
        self.P = rng.normal(scale=1.0/np.sqrt(self.M), size=(self.M, d)).astype(np.float32)

        # --- ensemble of linear models y = W x + b ; x = [phi_t, a_t] ---
        self.E = ensemble
        self.d = d
        act_space = self.env.action_space
        assert len(act_space.shape) == 1, "expects 1D continuous action space" #shape is (30,)
        self.ad = act_space.shape[0]

        self.lr = lr
        self.l2 = l2
        self.boot_p = boot_p

        # action scaling to [-1, 1]
        self.a_low  = np.array(act_space.low,  dtype=np.float32)
        self.a_high = np.array(act_space.high, dtype=np.float32)
        self.a_span = np.where(np.isfinite(self.a_high - self.a_low),
                               (self.a_high - self.a_low), 2.0)
        self.a_mid  = np.where(np.isfinite(self.a_high + self.a_low),
                               0.5*(self.a_high + self.a_low), 0.0)

        # parameters: list of (W, b)
        self.models = []
        for e in range(self.E):
            W = rng.normal(scale=0.05, size=(d, d + self.ad)).astype(np.float32) #compressed sensor space + action space, might need normalization
            b = np.zeros(d, dtype=np.float32) # why not the same size as W?
            self.models.append([W, b])

        # roll-over buffers
        self.prev_phi = None          # phi(s_t) from last reset/step
        self._x_prev = None           # [phi_t, a_t_scaled] used for last reward/update
        self._pre_r = 0.0             # disagreement computed for the last action

    # ---- small helpers ----
    def _phi(self, obs):
        touch = np.asarray(obs["touch"], dtype=np.float32).reshape(-1)
        return touch @ self.P  # (d,)
    def _scale_action(self, a):
        a = np.asarray(a, dtype=np.float32)
        # map to approx [-1,1] even if bounds are infinite/symmetric
        if np.all(np.isfinite(self.a_low)) and np.all(np.isfinite(self.a_high)):
            return (a - self.a_mid) / (0.5 * self.a_span + 1e-8)
        return np.clip(a, -1.0, 1.0)

    def _ensemble_pred(self, x):
        # stack predictions from each model: (E, d)
        preds = []
        for (W, b) in self.models:
            preds.append(W @ x + b)
        return np.stack(preds, axis=0)

    def _disagreement(self, preds):
        # variance across ensemble (average over feature dims)
        return float(np.mean(np.var(preds, axis=0)))  # scalar

    def _sgd_update(self, x, y):
        # one online SGD step per model; bootstrap updates to keep diversity
        for (W, b) in self.models:
            if np.random.rand() > self.boot_p:
                continue
            y_hat = W @ x + b               # (d,)
            err = (y_hat - y)               # (d,)
            # L2 on weights
            W *= (1.0 - self.lr * self.l2)
            # SGD: W -= lr * err[:, None] * x[None, :]
            W -= self.lr * np.outer(err, x)
            b -= self.lr * err

    # ---- RL plumbing ----
    def compute_intrinsic_reward(self, obs):
        """
        Called *after* env.step(). We:
          1) use the precomputed disagreement reward (set in step() before stepping),
          2) update models with target phi(s_{t+1}),
          3) roll phi.
        """
        # if we haven't formed an input yet (first step after reset), reward is zero
        r = float(self._pre_r)

        # update on the observed target
        phi_next = self._phi(obs)  # phi(s_{t+1})
        if self._x_prev is not None:
            self._sgd_update(self._x_prev, phi_next)

        # roll
        self.prev_phi = phi_next
        self._x_prev = None
        self._pre_r = 0.0

        return r

    def step(self, action):
        # ---- compute disagreement reward for current (phi_t, a_t) BEFORE stepping ----
        if self.prev_phi is not None:
            a_scaled = self._scale_action(action)
            x = np.concatenate([self.prev_phi, a_scaled]).astype(np.float32)  # (d+ad,)
            preds = self._ensemble_pred(x)                                    # (E,d)
            self._pre_r = self._disagreement(preds)                           # scalar
            self._x_prev = x
        else:
            self._pre_r = 0.0
            self._x_prev = None

        # ---- step env ----
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        # ---- compute intrinsic (also trains models & rolls phi) ----
        intrinsic_reward = self.compute_intrinsic_reward(obs)

        total_reward = float(intrinsic_reward) + float(extrinsic_reward)
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # gymnasium returns (obs, info)
        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}
        self.prev_phi = self._phi(obs)
        self._x_prev = None
        self._pre_r = 0.0
        return (obs, info) if isinstance(out, tuple) else obs


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

    model = PPO.load("results/self_touch/model.zip", env=wrapped_env, device="auto")  # keep your path

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
