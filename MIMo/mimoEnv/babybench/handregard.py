""" This module contains a simple reaching experiment in which MIMo tries to stand up.

The scene consists of MIMo and some railings representing a crib. MIMo starts sitting on the ground with his hands
on the railings. The task is to stand up.
MIMos feet and hands are welded to the ground and railings, respectively. He can move all joints in his arms, legs and
torso. His head is fixed.
Sensory input consists of proprioceptive and vestibular inputs, using the default configurations for both.

MIMo initial position is determined by slightly randomizing all joint positions from a standing position and then
letting the simulation settle. This leads to MIMo sagging into a slightly random crouching or sitting position each
episode. All episodes have a fixed length, there are no goal or failure states.

Reward shaping is employed, such that MIMo is penalised for using muscle inputs and large inputs in particular.
Additionally, he is rewarded each step for the current height of his head.

The class with the environment is :class:`~mimoEnv.envs.standup.MIMoStandupEnv` while the path to the scene XML is
defined in :data:`STANDUP_XML`.
"""
import os
import numpy as np
import mujoco

from mimoEnv.babybench.base import BabyBenchEnv
from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VESTIBULAR_PARAMS
from mimoActuation.actuation import SpringDamperModel
import mimoEnv.utils as env_utils
import mimoEnv.babybench.utils as bb_utils

SCENE_XML = os.path.join(SCENE_DIRECTORY, "babybench_cubes.xml")
""" Path to the stand up scene.

:meta hide-value:
"""

class BabyBenchHandRegardEnv(BabyBenchEnv):
    """ MIMo stands up using crib railings as an aid.

    Attributes and parameters are the same as in the base class, but the default arguments are adapted for the scenario.
    Specifically we have :attr:`.done_active` and :attr:`.goals_in_observation` as ``False`` and touch and vision
    sensors disabled.

    Even though we define a success condition in :meth:`~mimoEnv.envs.standup.MIMoStandupEnv._is_success`, it is
    disabled since :attr:`.done_active` is set to ``False``. The purpose of this is to enable extra information for
    the logging features of stable baselines.

    Attributes:
        init_crouch_position (numpy.ndarray): The initial position.
    """
    def __init__(self,
                 model_path=SCENE_XML,
                 frame_skip=1,
                 **kwargs):

        super().__init__(model_path=model_path,
                         frame_skip=frame_skip,
                         **kwargs)
        
        # initialize info functions
        self._info_init()

    def _info(self):
        # Get positions of hands and eyes
        right_hand_pos = self.data.body('right_hand').xpos
        left_hand_pos = self.data.body('left_hand').xpos
        right_eye_pos = self.data.body('right_eye').xpos
        left_eye_pos = self.data.body('left_eye').xpos
        # Compute vectors from eyes to hands
        right_eye_to_right_hand = right_hand_pos - right_eye_pos
        right_eye_to_left_hand = left_hand_pos - right_eye_pos
        left_eye_to_right_hand = right_hand_pos - left_eye_pos
        left_eye_to_left_hand = left_hand_pos - left_eye_pos
        # Get rotation matrices of eyes
        right_eye_rot = self.data.body('right_eye').xmat.reshape(3, 3)
        left_eye_rot = self.data.body('left_eye').xmat.reshape(3, 3)
        # Camera forward direction is the negative z-axis in the eye's local frame
        right_eye_forward = -right_eye_rot[:, 2]
        left_eye_forward = -left_eye_rot[:, 2]
        # Compute angles between eye forward directions and eyes-to-hands vectors
        right_eye_right_hand_angle = self._angle_between_vectors(right_eye_forward, right_eye_to_right_hand)
        right_eye_left_hand_angle = self._angle_between_vectors(right_eye_forward, right_eye_to_left_hand)
        left_eye_right_hand_angle = self._angle_between_vectors(left_eye_forward, left_eye_to_right_hand)
        left_eye_left_hand_angle = self._angle_between_vectors(left_eye_forward, left_eye_to_left_hand)
        # Field of view of eye cameras
        fovy = 30.0
        # Add to hand regard times
        self._hand_regard_right_eye_right_hand += int(right_eye_right_hand_angle < fovy/2)
        self._hand_regard_right_eye_left_hand += int(right_eye_left_hand_angle < fovy/2)
        self._hand_regard_left_eye_right_hand += int(left_eye_right_hand_angle < fovy/2)
        self._hand_regard_left_eye_left_hand += int(left_eye_left_hand_angle < fovy/2)
        info = {'right_eye_right_hand': self._hand_regard_right_eye_right_hand,
                'right_eye_left_hand': self._hand_regard_right_eye_left_hand,
                'left_eye_right_hand': self._hand_regard_left_eye_right_hand,
                'left_eye_left_hand': self._hand_regard_left_eye_left_hand}
        return info

    def _info_init(self):
        self._hand_regard_right_eye_right_hand = 0
        self._hand_regard_right_eye_left_hand = 0
        self._hand_regard_left_eye_right_hand = 0
        self._hand_regard_left_eye_left_hand = 0

    def _randomize_reset(self):
        # perform 10 steps with random actions
        for _ in range(10):
            self._set_action(self.action_space.sample())
            self._single_mujoco_step()
    