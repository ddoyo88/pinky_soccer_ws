"""Run a trained RL striker policy inside the ROS striker - numpy only (exported by pinky_soccer_rl.export).

MLPPolicy    the network: VecNormalize observation scaling -> Tanh MLP -> mean action clipped to [-1, 1]
RLPlayer     builds the SAME observations as pinky_soccer_rl/env.py from what the striker has (AMCL pose,
             camera ball estimate with memory, lidar opponent; 19-input policies also the ball velocity from
             successive camera estimates) and turns the action into (v, w).

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


def has_priority(me, opp, ball, tie_first, margin=0.03, had=None, keep=0.06):
    """Right of way at the ball (no communication needed: each robot sees the ball with its camera and the other
    robot with its lidar).  The robot clearly closer to the ball (by `margin`) goes first; a near tie goes to the
    robot with tie_first (real match: yellow; training: random per episode).
    Hysteresis (had = the previous decision): a robot that has the right of way keeps it until the other one is
    `keep` closer, one without it gets it only when `keep` closer itself - no flipping every frame when both are
    about as far away (that made both hesitate)."""
    if ball is None or opp is None:
        return True
    d_me = math.hypot(ball[0] - me[0], ball[1] - me[1])
    d_opp = math.hypot(ball[0] - opp[0], ball[1] - opp[1])
    if had is not None:
        if had:
            return not (d_opp < d_me - keep)
        return d_me < d_opp - keep
    if abs(d_me - d_opp) < margin:
        return bool(tie_first)
    return d_me < d_opp


def robot_guard(v, opp_rel, priority=None, corridor=0.16):
    """No collisions with the other robot (shared by the RL training environment and the ROS striker).
    opp_rel: other robot centre in this robot's frame (x forward, y left) or None.  Driving towards it (it lies
    in the corridor ahead when v > 0, behind when v < 0): full speed until 0.12 m before the stop distance,
    linearly less, zero at the stop distance (centre to centre).  Turning on the spot is not limited.
    Stop distance: 0.20 m (no right-of-way information) / 0.17 m with the right of way at the ball (~2 cm
    between the bumpers: it may go in) / 0.24 m without it (it gives way) - so two robots next to each other
    never both wait for the other."""
    if opp_rel is None or abs(v) < 1e-6:
        return v
    ox, oy = float(opp_rel[0]), float(opp_rel[1])
    ahead = ox if v > 0 else -ox
    if ahead <= 0.0 or abs(oy) > corridor:
        return v
    stop = 0.20 if priority is None else (0.17 if priority else 0.24)
    d = math.hypot(ox, oy)
    return v * max(0.0, min(1.0, (d - stop) / 0.12))


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
        self.prev_ball, self.ball_vel = None, np.zeros(2)

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
        # ball velocity (attack frame) like the environment: smoothed difference of successive sightings
        if seen and ball is not None:
            fb = np.array(self._frame(*ball))
            if self.prev_ball is not None:
                self.ball_vel = 0.5 * self.ball_vel + 0.5 * np.clip((fb - self.prev_ball) / self.pol.dt, -1.5, 1.5)
            self.prev_ball = fb
        else:
            self.prev_ball = None
            self.ball_vel = 0.5 * self.ball_vel
        c, s = math.cos(p[2]), math.sin(p[2])
        bv = (c * self.ball_vel[0] + s * self.ball_vel[1], -s * self.ball_vel[0] + c * self.ball_vel[1])
        obs = np.array([b[0], b[1], 1.0 if seen else 0.0, min(5.0, since_seen) / 5.0,
                        p[0] / (self.L / 2), p[1] / (self.W / 2), math.cos(p[2]), math.sin(p[2]),
                        g[0] / self.L, g[1] / self.L, self.meas[0] / self.pol.v_max, self.meas[1] / self.pol.w_max,
                        self.prev_action[0], self.prev_action[1], o[0], o[1], 1.0 if ball is not None else 0.0,
                        bv[0], bv[1]])[:self.pol.obs_size]
        a = self.pol(obs)
        self.prev_action = a
        return float(a[0] * self.pol.v_max), float(a[1] * self.pol.w_max), obs
