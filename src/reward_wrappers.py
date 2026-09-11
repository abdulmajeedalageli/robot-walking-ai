"""
Reward wrapper for the MuJoCo Humanoid env.

Adds two extra reward terms on top of the default reward
(forward_velocity - ctrl_cost - contact_cost + healthy_bonus):

1. Arm-swing symmetry - rewards the arms swinging opposite to each other,
   like they do in a normal human walk. Penalizes the squared sum of the
   left+right shoulder angular velocities, which is smallest when the two
   arms move in opposite directions.

2. Straight knee at stance - penalizes bent knees while a foot is planted
   on the ground. Without this the policy tends to shuffle/crouch instead
   of walking with extended legs.


"""

import numpy as np
import gymnasium as gym
import mujoco


class NaturalGaitRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        arm_swing_weight: float = 0.05,
        knee_weight: float = 0.03,
        shoulder_joint_names=("right_shoulder1", "left_shoulder1"),
        knee_joint_names=("right_knee", "left_knee"),
        foot_body_names=("right_foot", "left_foot"),
        straight_knee_angle: float = -0.05,
        verbose: bool = False,
    ):
        super().__init__(env)
        self.arm_swing_weight = arm_swing_weight
        self.knee_weight = knee_weight
        self.right_shoulder_name, self.left_shoulder_name = shoulder_joint_names
        self.right_knee_name, self.left_knee_name = knee_joint_names
        self.right_foot_name, self.left_foot_name = foot_body_names
        self.straight_knee_angle = straight_knee_angle
        self.verbose = verbose

        self._model = None
        self._data = None
        self._ids_resolved = False

    def _resolve_ids(self):
        # grab model/data refs once the underlying env actually exists
        self._model = self.env.unwrapped.model
        self._data = self.env.unwrapped.data

        def jid(name):
            return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)

        def bid(name):
            return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)

        self.right_shoulder_jid = jid(self.right_shoulder_name)
        self.left_shoulder_jid = jid(self.left_shoulder_name)
        self.right_knee_jid = jid(self.right_knee_name)
        self.left_knee_jid = jid(self.left_knee_name)
        self.right_foot_bid = bid(self.right_foot_name)
        self.left_foot_bid = bid(self.left_foot_name)

        # bail early if any name doesn't exist in this model, instead of
        # failing later with a confusing index error
        missing = [
            n for n, i in [
                (self.right_shoulder_name, self.right_shoulder_jid),
                (self.left_shoulder_name, self.left_shoulder_jid),
                (self.right_knee_name, self.right_knee_jid),
                (self.left_knee_name, self.left_knee_jid),
                (self.right_foot_name, self.right_foot_bid),
                (self.left_foot_name, self.left_foot_bid),
            ] if i == -1
        ]
        if missing:
            raise ValueError(
                f"Couldn't find these names in the MuJoCo model: {missing}. "
                "Run debug_print_names() to see the real joint/body names "
                "and pass the correct ones to the wrapper."
            )
        self._ids_resolved = True

        if self.verbose:
            print(
                f"[NaturalGaitRewardWrapper] resolved -> "
                f"shoulders=({self.right_shoulder_jid},{self.left_shoulder_jid}) "
                f"knees=({self.right_knee_jid},{self.left_knee_jid}) "
                f"feet=({self.right_foot_bid},{self.left_foot_bid})"
            )

    def debug_print_names(self):
        # quick way to dump every joint/body name if defaults don't match
        model = self.env.unwrapped.model
        print("Joint names:")
        for i in range(model.njnt):
            print(" ", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
        print("Body names:")
        for i in range(model.nbody):
            print(" ", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))

    def _foot_in_contact(self, body_id):
        data = self._data
        model = self._model
        for c in range(data.ncon):
            contact = data.contact[c]
            geom1_body = model.geom_bodyid[contact.geom1]
            geom2_body = model.geom_bodyid[contact.geom2]
            if geom1_body == body_id or geom2_body == body_id:
                return True
        return False

    def _extra_reward(self):
        model, data = self._model, self._data

        # anti-phase arm swing: sum of shoulder velocities should hover
        # around zero if the arms are swinging opposite each other
        r_shoulder_qveladr = model.jnt_dofadr[self.right_shoulder_jid]
        l_shoulder_qveladr = model.jnt_dofadr[self.left_shoulder_jid]
        r_shoulder_vel = data.qvel[r_shoulder_qveladr]
        l_shoulder_vel = data.qvel[l_shoulder_qveladr]
        arm_swing_penalty = (r_shoulder_vel + l_shoulder_vel) ** 2
        arm_swing_reward = -self.arm_swing_weight * arm_swing_penalty

        # only penalize knee bend while that foot is actually on the ground
        r_knee_qposadr = model.jnt_qposadr[self.right_knee_jid]
        l_knee_qposadr = model.jnt_qposadr[self.left_knee_jid]
        r_knee_angle = data.qpos[r_knee_qposadr]
        l_knee_angle = data.qpos[l_knee_qposadr]

        knee_penalty = 0.0
        if self._foot_in_contact(self.right_foot_bid):
            knee_penalty += max(0.0, self.straight_knee_angle - r_knee_angle) ** 2
        if self._foot_in_contact(self.left_foot_bid):
            knee_penalty += max(0.0, self.straight_knee_angle - l_knee_angle) ** 2
        knee_reward = -self.knee_weight * knee_penalty

        return arm_swing_reward + knee_reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not self._ids_resolved:
            self._resolve_ids()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self._ids_resolved:
            self._resolve_ids()
        extra = self._extra_reward()
        info["natural_gait_bonus"] = extra
        return obs, reward + extra, terminated, truncated, info