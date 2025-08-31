
"""
Self-supervised exploration as Disagreement for hand regard.
"""
import numpy as np
import os
import gymnasium as gym
import time
import argparse
import mujoco
import yaml
from stable_baselines3 import PPO
import sys
sys.path.append(".")
sys.path.append("..")
import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as env_utils
import babybench.utils as bb_utils

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
    """
    Intrinsic reward = ensemble prediction disagreement for next *visual* feature.

    - Uses both eyes (eye_left/eye_right). Each eye image is downsampled in-place
      (stride sampling), converted to grayscale, per-frame standardized, then
      projected to low-d features with a fixed random matrix.
    - Optionally includes proprio (e.g., 'qpos') if present in obs; also projected.
    - Tiny linear ensemble trained online with SGD; reward = variance across
      ensemble predictions for phi_{t+1} given [phi_t; a_t].
    - Reward computed BEFORE env.step on (phi_t, a_t); models updated AFTER with phi_{t+1}.
    - Includes light clipping/normalization for stability.
    """

    def __init__(
        self,
        env,
        *,
        ensemble=5,
        d=96,        # total feature dim (split across eyes, e.g., 64+32)
        lr=1e-3,
        l2=1e-3,
        boot_p=0.5,
        stride=8,    # image downsample stride
        eps=1e-6
    ):
        super().__init__(env)

        self.eps = eps
        self.stride = int(stride)
        self.E = int(ensemble)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.boot_p = float(boot_p)

        # ---- check obs space and sizes ----
        obs_space = self.env.observation_space
        assert hasattr(obs_space, "spaces"), "Expect Dict obs space"
        assert "eye_left" in obs_space.spaces and "eye_right" in obs_space.spaces, \
            "Expect 'eye_left' and 'eye_right' image observations"

        # Get image shapes from spaces (H, W, C) or (H, W)
        def _hwc(shape):
            if len(shape) == 3:  # H, W, C
                return shape[0], shape[1], shape[2]
            elif len(shape) == 2:  # H, W (grayscale)
                return shape[0], shape[1], 1
            else:
                raise RuntimeError(f"Unexpected image shape: {shape}")

        Hl, Wl, Cl = _hwc(obs_space.spaces["eye_left"].shape)
        Hr, Wr, Cr = _hwc(obs_space.spaces["eye_right"].shape)

        # Downsampled sizes (stride sampling)
        self.Nl = (Hl // self.stride) * (Wl // self.stride)   # left pixels after downsample (grayscale)
        self.Nr = (Hr // self.stride) * (Wr // self.stride)   # right pixels after downsample

        # Optional proprio block (e.g., joint positions)
        self.use_proprio = "qpos" in obs_space.spaces
        self.Nq = int(np.prod(obs_space.spaces["qpos"].shape)) if self.use_proprio else 0

        # ---- feature dims per modality ----
        # Split total d across eyes (and a small slice for proprio if available)
        if self.use_proprio:
            dq = max(16, min(32, d // 6))     # small slice for proprio
            d_eyes = d - dq
        else:
            dq = 0
            d_eyes = d
        self.dL = max(16, (2 * d_eyes) // 3)  # left eye gets ~2/3
        self.dR = d_eyes - self.dL            # right eye gets remainder
        self.dq = dq
        self.d = self.dL + self.dR + self.dq  # final feature dim

        rng = np.random.RandomState(1234)
        # Random projections for each modality (scaled for variance ~1)
        self.P_left  = rng.normal(scale=1.0 / np.sqrt(max(1, self.Nl)), size=(self.Nl, self.dL)).astype(np.float32)
        self.P_right = rng.normal(scale=1.0 / np.sqrt(max(1, self.Nr)), size=(self.Nr, self.dR)).astype(np.float32)
        if self.use_proprio and self.Nq > 0 and self.dq > 0:
            self.P_q = rng.normal(scale=1.0 / np.sqrt(max(1, self.Nq)), size=(self.Nq, self.dq)).astype(np.float32)
        else:
            self.P_q = None

        # ---- action space ----
        act_space = self.env.action_space
        assert len(act_space.shape) == 1, "expects 1D continuous action space"
        self.ad = int(act_space.shape[0])

        # ---- stability guards ----
        self.pred_clip = 50.0
        self.err_clip  = 10.0
        self.x_clip    = 10.0
        self.r_clip    = 10.0

        # ---- ensemble models: y = W x + b, x = [phi_t, a_t] ----
        self.models = []
        for _ in range(self.E):
            W = rng.normal(scale=0.05, size=(self.d, self.d + self.ad)).astype(np.float32)
            b = np.zeros(self.d, dtype=np.float32)
            self.models.append([W, b])

        # ---- roll-over buffers ----
        self.prev_phi = None
        self._x_prev = None
        self._pre_r = 0.0

    # ----------------- helpers -----------------

    @staticmethod
    def _to_gray01(img):
        """Return grayscale float32 in [0,1] from HxWxC uint8/float arrays."""
        arr = np.asarray(img)
        if arr.ndim == 3:  # H, W, C
            if arr.dtype != np.float32 and arr.dtype != np.float64:
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
            gray = arr.mean(axis=2)  # simple luminance
        elif arr.ndim == 2:
            gray = arr.astype(np.float32)
            if gray.max() > 1.0 + 1e-6:  # probably uint8
                gray /= 255.0
        else:
            raise RuntimeError(f"Unexpected image ndim={arr.ndim}")
        return np.clip(gray, 0.0, 1.0)

    def _eye_feat(self, eye_img, P, N_expected):
        """Downsample by stride, standardize per-frame, then project with P."""
        gray = self._to_gray01(eye_img)
        # stride-sample
        small = gray[::self.stride, ::self.stride].reshape(-1)
        if small.size != N_expected:
            # guard if shapes shift; resize by cropping to expected length
            N = min(N_expected, small.size)
            tmp = np.zeros(N_expected, dtype=np.float32)
            tmp[:N] = small[:N].astype(np.float32)
            small = tmp
        # per-frame standardization to stabilize scale across lighting/content
        mu = float(small.mean())
        sd = float(small.std())
        small = (small - mu) / (sd + 1e-6)
        # project
        return (small @ P).astype(np.float32)

    def _phi(self, obs):
        # Build visual feature from both eyes
        left = self._eye_feat(obs["eye_left"],  self.P_left,  self.Nl)
        right = self._eye_feat(obs["eye_right"], self.P_right, self.Nr)
        parts = [left, right]
        # Optional proprio (e.g., qpos)
        if self.use_proprio:
            q = np.asarray(obs["qpos"], np.float32).reshape(-1)
            if self.P_q is not None:
                q_proj = (np.nan_to_num(q) @ self.P_q).astype(np.float32)
                parts.append(q_proj)
        phi = np.concatenate(parts).astype(np.float32)  # (d,)
        # sanitize/clip
        return np.clip(np.nan_to_num(phi), -self.pred_clip, self.pred_clip)

    def _scale_action(self, a):
        # Env already uses [-1, 1]; just clip and sanitize
        a = np.asarray(a, np.float32)
        return np.clip(np.nan_to_num(a), -1.0, 1.0)

    def _ensemble_pred(self, x):
        x = np.clip(np.nan_to_num(x), -self.x_clip, self.x_clip)
        preds = []
        for (W, b) in self.models:
            y = W @ x + b
            preds.append(np.clip(np.nan_to_num(y), -self.pred_clip, self.pred_clip))
        return np.stack(preds, axis=0)  # (E, d)

    @staticmethod
    def _disagreement(preds):
        # mean variance across ensemble along feature dims → scalar
        return float(np.mean(np.var(preds, axis=0)))

    def _sgd_update(self, x, y):
        x = np.clip(np.nan_to_num(x), -self.x_clip, self.x_clip)
        y = np.clip(np.nan_to_num(y), -self.pred_clip, self.pred_clip)
        for (W, b) in self.models:
            if np.random.rand() > self.boot_p:
                continue
            y_hat = W @ x + b
            err = np.clip(np.nan_to_num(y_hat - y), -self.err_clip, self.err_clip)
            # L2 on weights
            W *= (1.0 - self.lr * self.l2)
            # SGD step
            W -= self.lr * np.outer(err, x)
            b -= self.lr * err

    # --------------- RL plumbing ----------------

    def compute_intrinsic_reward(self, obs):
        """
        Called AFTER env.step(). Return reward computed in step() for (phi_t, a_t),
        then update models on phi_{t+1} and roll state.
        """
        r = float(np.clip(np.nan_to_num(self._pre_r, nan=0.0, posinf=self.r_clip), 0.0, self.r_clip))

        # update on observed target
        phi_next = self._phi(obs)
        if self._x_prev is not None:
            self._sgd_update(self._x_prev, phi_next)

        # roll
        self.prev_phi = phi_next
        self._x_prev = None
        self._pre_r = 0.0

        return r

    def step(self, action):
        # Compute disagreement reward BEFORE stepping (uses phi_t, a_t)
        if self.prev_phi is not None:
            a_scaled = self._scale_action(action)
            x = np.concatenate([self.prev_phi, a_scaled]).astype(np.float32)  # (d + ad,)
            preds = self._ensemble_pred(x)                                    # (E, d)
            r = self._disagreement(preds)
            self._pre_r = float(np.clip(np.nan_to_num(r, nan=0.0, posinf=self.r_clip), 0.0, self.r_clip))
            self._x_prev = x
        else:
            self._pre_r = 0.0
            self._x_prev = None

        # Step environment
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        # Compute intrinsic (also updates models and rolls phi)
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
    parser.add_argument('--config', default='examples/config_handregard.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--train_for', default=10000, type=int,
                        help='Total timesteps of training')
    args = parser.parse_args()
    
    with open(args.config) as f:
            config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    wrapped_env = Wrapper(env)
    wrapped_env.reset()

    model = PPO("MultiInputPolicy", wrapped_env, verbose=1)
    model.learn(total_timesteps=args.train_for)
    model.save(os.path.join(config["save_dir"], "model"))

if __name__ == '__main__':
    main()
