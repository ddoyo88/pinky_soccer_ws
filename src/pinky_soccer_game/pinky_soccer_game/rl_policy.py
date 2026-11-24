"""Run a trained RL striker policy inside the ROS striker - numpy only (exported by pinky_soccer_rl.export).

MLPPolicy    the network: VecNormalize observation scaling -> Tanh MLP -> mean action clipped to [-1, 1]
RLPlayer     builds the SAME 17 observations as pinky_soccer_rl/env.py from what the striker has (AMCL pose,
             camera ball estimate with memory, lidar opponent) and turns the action into (v, w).

The policy was trained as YELLOW attacking +X.  Everything is first rotated into the "attack frame" (the
attacked goal at +X), so the same policy plays blue (attacks -X) as well.
"""
import math

import numpy as np


class MLPPolicy:
    def __init__(self, path):
        d = np.load(path)
        n = int(d['n_layers'])
        self.layers = [(d[f'w{i}'], d[f'b{i}']) for i in range(n)]
        self.mean, self.var = d['obs_mean'], d['obs_var']
        self.clip, self.eps = float(d['clip_obs']), float(d['epsilon'])
        self.use_opponent = bool(d['use_opponent'])
        self.v_max, self.w_max, self.dt = float(d['v_max']), float(d['w_max']), float(d['dt'])
        self.obs_size = int(d['obs_size'])

    def __call__(self, obs):
        x = np.clip((np.asarray(obs, dtype=float) - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return np.clip(x, -1.0, 1.0)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class RLPlayer:
    def __init__(self, path, field_length, field_width, attack_yaw):
        self.pol = MLPPolicy(path)
        self.L, self.W = field_length, field_width
        self.c, self.s = math.cos(-attack_yaw), math.sin(-attack_yaw)     # map -> attack frame
        self.dyaw = -attack_yaw
        self.goal = np.array([self.L / 2, 0.0])
        self.reset()

    def reset(self):
        self.prev_action = np.zeros(2)
        self.prev_pose = None
        self.meas = np.zeros(2)

    def _frame(self, x, y):
        return self.c * x - self.s * y, self.s * x + self.c * y

    @staticmethod
    def _to_robot(p, pose):
        x, y, yaw = pose
        dx, dy = p[0] - x, p[1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * dx + s * dy, -s * dx + c * dy])

    def act(self, pose, ball, seen, since_seen, opponent=None):
        """pose (x, y, yaw) map; ball: last estimate (map) or None; seen: the camera sees it now;
        since_seen [s]; opponent (x, y) map or None.  Returns (v, w, observation)."""
        fx, fy = self._frame(pose[0], pose[1])
        p = (fx, fy, _wrap(pose[2] + self.dyaw))
        if self.prev_pose is not None:                   # measured speed like the environment (pose change)
            dx, dy = p[0] - self.prev_pose[0], p[1] - self.prev_pose[1]
            sign = 1.0 if math.cos(math.atan2(dy, dx) - self.prev_pose[2]) >= 0 else -1.0
            self.meas = np.array([math.hypot(dx, dy) / self.pol.dt * sign,
                                  _wrap(p[2] - self.prev_pose[2]) / self.pol.dt])
        self.prev_pose = p
        # never seen yet: the training environment gives the field centre (with the 'ball known' flag 0)
        b = self._to_robot(self._frame(*ball) if ball is not None else (0.0, 0.0), p)
        g = self._to_robot(self.goal, p)
        o = self._to_robot(self._frame(*opponent), p) if (opponent is not None and self.pol.use_opponent) \
            else np.zeros(2)
        obs = np.array([b[0], b[1], 1.0 if seen else 0.0, min(5.0, since_seen) / 5.0,
                        p[0] / (self.L / 2), p[1] / (self.W / 2), math.cos(p[2]), math.sin(p[2]),
                        g[0] / self.L, g[1] / self.L, self.meas[0] / self.pol.v_max, self.meas[1] / self.pol.w_max,
                        self.prev_action[0], self.prev_action[1], o[0], o[1], 1.0 if ball is not None else 0.0])
        a = self.pol(obs)
        self.prev_action = a
        return float(a[0] * self.pol.v_max), float(a[1] * self.pol.w_max), obs
