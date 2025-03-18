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

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, \
                                  DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VESTIBULAR_PARAMS, \
                                  DEFAULT_TOUCH_PARAMS, DEFAULT_TOUCH_PARAMS_V2
from mimoActuation.actuation import SpringDamperModel, PositionalModel
from mimoActuation.muscle import MuscleModel
import mimoEnv.utils as env_utils

BASE_XML = os.path.join(SCENE_DIRECTORY, "babybench_base.xml")
""" Path to the stand up scene.

:meta hide-value:
"""

VISION_PARAMS = {
    "eye_left": {"width": 64, "height": 64},
    "eye_right": {"width": 64, "height": 64},
}
""" Default vision parameters.

:meta hide-value:
"""

ACTUATION_MODELS = {
    'spring_damper': SpringDamperModel,
    'positional': PositionalModel,
    'muscle': MuscleModel,
}

DEFAULT_SIZE = 480
""" Default window size for gym rendering functions.

:meta hide-value:
"""

class BabyBenchEnv(MIMoEnv):
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
                 model_path=BASE_XML,
                 frame_skip=1,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 vision_params=VISION_PARAMS,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 touch_params=DEFAULT_TOUCH_PARAMS,
                 actuation_model='spring_damper',
                 width=DEFAULT_SIZE,
                 height=DEFAULT_SIZE,
                 **kwargs):

        if kwargs['vision_active'] is not None:
            if kwargs['vision_active'] is False:
                vision_params = None
            elif kwargs['vision_resolution'] is not None:
                vision_params = {
                    "eye_left": {"width": kwargs['vision_resolution'], "height": kwargs['vision_resolution']},
                    "eye_right": {"width": kwargs['vision_resolution'], "height": kwargs['vision_resolution']},
                }
        if kwargs['vestibular_active'] is not None:
            if kwargs['vestibular_active'] is False:
                vestibular_params = None

        if kwargs['touch_active'] is not None:
            if (kwargs['touch_active'] is False) or (kwargs['touch_scale']==0):
                touch_params = None
            else:
                if kwargs['touch_scale'] is not None:
                    for body in touch_params["scales"]:
                        touch_params["scales"][body] = DEFAULT_TOUCH_PARAMS["scales"][body]*kwargs['touch_scale']
                if kwargs['touch_function'] is not None:
                    touch_params["touch_function"] = kwargs['touch_function']
                if kwargs['touch_response_function'] is not None:
                    touch_params["response_function"] = kwargs['touch_response_function']

        self.behavior = kwargs['behavior']

        super().__init__(model_path=model_path,
                         frame_skip=frame_skip,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=ACTUATION_MODELS[actuation_model],
                         goals_in_observation=False,
                         done_active=False,
                         width=width,
                         height=height,)

        self.right_hand_geoms = env_utils.get_geoms_for_body(self.model, env_utils.get_body_id(self.model, body_name="right_hand"))
        self.left_hand_geoms = env_utils.get_geoms_for_body(self.model, env_utils.get_body_id(self.model, body_name="left_hand"))
        self.mimo_bodies = env_utils.get_child_bodies(self.model, env_utils.get_body_id(self.model, body_name="hip"))
        self.mimo_geoms = np.concatenate([np.array(env_utils.get_geoms_for_body(self.model, body_id)) for body_id in self.mimo_bodies])
        
        # initialize 
        self.set_state(self.init_qpos, self.init_qvel)
        # perform 100 steps with no action to stabilize initial position
        self._set_action(np.zeros(self.action_space.shape))
        for _ in range(100):
            self._single_mujoco_step()
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        self.steps = 0

        # initialize info functions
        self._info_init()
                         
    def compute_reward(self, **kwargs):
        """ Dummy function for intrinsically motivated learning.
        Always returns 0.
        """
        return 0

    def reset_model(self):
        """ Resets the simulation.

        Return the simulation to the XML state, then slightly randomize all joint positions. Afterwards we let the
        simulation settle for a fixed number of steps. This leads to MIMo settling into a slightly random sitting or
        crouching position.

        Returns:
            Dict: Observations after reset.
        """

        # initialize state
        self.set_state(self.init_qpos, self.init_qvel)

        # perform 10 steps with random actions for randomization
        for _ in range(10):
            self._set_action(self.action_space.sample())
            self._single_mujoco_step()

        # reset info functions
        self._info_init()
        
        self.steps = 0
        return self._get_obs()

    def _step_callback(self):
        pass

    def step(self, action):
        """ Run one timestep of the environment's dynamics.

        This function takes a simulation step with the given control inputs, collects the observations, computes the
        reward and finally determines if we are done with this episode or not. :meth:`._get_obs` collects the
        observations, :meth:`.compute_reward` calculates the reward.`:meth:`._is_done` is called to determine if we
        have reached a terminal state and :meth:`._step_callback` can be used for extra functions each step, such as
        incrementing a step counter. Both the 'terminated' and 'truncated' return values are determined by
        `:meth:`._is_done`.

        Args:
            action (np.ndarray): An action provided by the agent

        Returns:
            observation (object): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a `terminal state` (success or failure as defined under the MDP of the task) is
                reached. In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and
                logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that
                are hidden from observations, or individual reward terms that are combined to produce the total reward.
        """
        self.do_simulation(action, self.frame_skip)
        self._step_callback()
        obs = self._get_obs()
        self._obs_callback()

        info = self._info()
        
        terminated, truncated = self._is_done(None, self.goal, info)
        reward = self.compute_reward()
        self.steps += 1
        return obs, reward, terminated, truncated, info
        
    def _info(self):
        info = {
            'steps' : self.steps,
        }
        if self.behavior == 'self_touch':
            info['self_touch'] = self._info_self_touch(),
        elif self.behavior == 'hand_regard':
            info['hand_regard'] = self._info_hand_regard(),
        return info

    def _info_init(self):
        if self.behavior == 'self_touch':
            self._self_touch_right_hand = []
            self._self_touch_left_hand = []
        elif self.behavior == 'hand_regard':
            self._hand_regard_right_eye_right_hand = 0
            self._hand_regard_right_eye_left_hand = 0
            self._hand_regard_left_eye_right_hand = 0
            self._hand_regard_left_eye_left_hand = 0

    def _info_self_touch(self):
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
        return {'right_hand_touches': self._self_touch_right_hand,
                'left_hand_touches': self._self_touch_left_hand}

    def _info_hand_regard(self):
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
        return {'right_eye_right_hand': self._hand_regard_right_eye_right_hand,
                'right_eye_left_hand': self._hand_regard_right_eye_left_hand,
                'left_eye_right_hand': self._hand_regard_left_eye_right_hand,
                'left_eye_left_hand': self._hand_regard_left_eye_left_hand}

    def _angle_between_vectors(self, v1, v2):
        v1_unit = v1 / np.linalg.norm(v1)
        v2_unit = v2 / np.linalg.norm(v2)
        dot_product = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
        angle = np.arccos(dot_product)
        return np.degrees(angle)

    def _angle_between_vector_and_rotation(self, vector, rotation_matrix):
        # Normalize the vector
        vector = vector / np.linalg.norm(vector)
        # The forward direction is the third column of the rotation matrix
        forward_direction = rotation_matrix[:, 2]
        # Compute the dot product
        dot_product = np.dot(vector, forward_direction)
        # Clamp the dot product to [-1, 1] to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        # Compute the angle in radians
        angle = np.arccos(dot_product)
        # Convert to degrees
        angle_degrees = np.degrees(angle)
        return angle_degrees

    def is_success(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns ``False``.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``False``
        """
        return False
    
    def is_failure(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns ``False``.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``False``
        """
        return False

    def is_truncated(self):
        """ Dummy function. Always returns ``False``.

        Returns:
            bool: ``False``.
        """
        return False

    def sample_goal(self):
        """ Dummy function. Always returns ``False``.

        Returns:
            bool: ``False``.
        """
        return False

    def get_achieved_goal(self):
        """ Dummy function. Always returns ``False``.

        Returns:
            bool: ``False``.
        """
        return False