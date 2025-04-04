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

SCENE_XML = os.path.join(SCENE_DIRECTORY, "babybench_crib.xml")
""" Path to the stand up scene.

:meta hide-value:
"""

class BabyBenchSelfTouchEnv(BabyBenchEnv):
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

        # initialize behavior-relevant variables
        self.right_hand_geoms = env_utils.get_geoms_for_body(self.model, env_utils.get_body_id(self.model, body_name="right_hand"))
        self.left_hand_geoms = env_utils.get_geoms_for_body(self.model, env_utils.get_body_id(self.model, body_name="left_hand"))
        self.mimo_bodies = env_utils.get_child_bodies(self.model, env_utils.get_body_id(self.model, body_name="hip"))
        self.mimo_geoms = np.concatenate([np.array(env_utils.get_geoms_for_body(self.model, body_id)) for body_id in self.mimo_bodies])
        
        # initialize info functions
        self._info_init()

    def _info(self):
        # Get all contacts
        contacts = self.data.contact
        for idx in range(len(contacts.geom1)):
            # Check if hand in contact
            if contacts.geom1[idx] in self.right_hand_geoms:
                other_geom = contacts.geom2[idx]
                # Check if other geom is in mimo's body and is new contact
                if (other_geom in self.mimo_geoms) and (other_geom not in self._self_touch_right_hand):
                    # Add to list of contacts
                    self._self_touch_right_hand.append(int(other_geom))
            elif contacts.geom2[idx] in self.right_hand_geoms:
                other_geom = contacts.geom1[idx]
                if (other_geom in self.mimo_geoms) and (other_geom not in self._self_touch_right_hand):
                    self._self_touch_right_hand.append(int(other_geom))
            elif contacts.geom1[idx] in self.left_hand_geoms:
                other_geom = contacts.geom2[idx]
                if (other_geom in self.mimo_geoms) and (other_geom not in self._self_touch_left_hand):
                    self._self_touch_left_hand.append(int(other_geom))
            elif contacts.geom2[idx] in self.left_hand_geoms:
                other_geom = contacts.geom1[idx]
                if (other_geom in self.mimo_geoms) and (other_geom not in self._self_touch_left_hand):
                    self._self_touch_left_hand.append(int(other_geom))
        info = {
            'right_hand_touches' : self._self_touch_right_hand,
            'left_hand_touches'  : self._self_touch_left_hand,
        }
        return info

    def _info_init(self):
        self._self_touch_right_hand = []
        self._self_touch_left_hand = []

    def _randomize_reset(self):
        # perform 10 steps with random actions
        for _ in range(10):
            self._set_action(self.action_space.sample())
            self._single_mujoco_step()
    