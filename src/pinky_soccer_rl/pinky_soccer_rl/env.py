"""Gymnasium environment: one Pinky striker, the golf ball, the training field - MuJoCo only (no ROS, no camera
rendering), so it runs far faster than real time.

Same physics as the match simulation (pinky_soccer_sim/mujoco_soccer.py): the official pinky_mujoco robot with
bumper, velocity-controlled wheels with the Dynamixel acceleration limit, rolling golf ball, field from
rcj_field.  The agent plays YELLOW and attacks +X.

Observation (what the real robot has, not the simulator's ground truth), in the robot's "attack frame":
    ball      in the robot frame (x, y) from the "camera": only inside the camera's view (hfov 59 deg, 0.10 -
              1.6 m), with noise; otherwise the last seen position (field centre before the first sighting)
    seen      1 if the camera sees the ball now, time since it was last seen (s / 5, clipped)
    robot     field pose x / (L/2), y / (W/2), cos(yaw), sin(yaw)   (AMCL on the real robot, with noise)
    goal      attacked goal centre in the robot frame (/ L)
    speed     measured v, w (/ max);   previous action
    opponent  (stage >= 2) other robot in the robot frame (lidar), 0 if none
    known     1 once the ball has been seen
    ball vel  (stage >= 3, observation size 19) ball velocity in the robot frame from successive camera
              estimates (smoothed), m/s
Action: [v, w] in [-1, 1] -> v * v_max (0.30 m/s), w * w_max (4 rad/s), held for 0.1 s (10 Hz like the striker).
The striker's guards are applied to the action, exactly as in the match: no motion that would cross the white
boundary line, and no driving into the other robot (slow from 0.32 m centre distance, stop at 0.20 m).
Right of way at the ball: the robot clearly closer to the ball goes first (rl_policy.has_priority) - its
guard lets it in to 0.17 m, the other one stops at 0.24 m; a near tie is decided at random per episode.
The other robot is an obstacle: a collision costs -5, every step in contact -0.2, coming closer than 0.25 m
(centre distance) up to -0.05 per step, being stopped by the guard -0.05; a robot tilted over 30 deg (flipped)
ends the episode (-10).

Stages (curriculum):
    1  no opponent (parked outside the field)
    2  scripted opponent (60 % of the episodes) that pushes the ball towards our goal, speed set by train.py
    3  defence: ball velocity input, defensive positioning reward, opponent = scripted pusher or a frozen copy of
       an earlier policy (self-play, .npz loaded with load_opponent)
"""
import math
import os
import tempfile

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from pinky_soccer_sim.field import resolve_dims

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pinky_soccer_game'))
from pinky_soccer_game.rl_policy import has_priority, robot_guard_vw  # noqa: E402  (the striker uses the same)

PARK = {'blue': (0.0, 3.0)}         # no opponent: the other robot far outside the field (re-placed every step)
OBS_BASE, OBS_VEL = 17, 19


def _yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


_MODEL_CACHE = {}


def load_model(style='hardboard', visual=False):
    """Build (once per process) the same scene as the match simulation.  visual=False (training): no painted
    lines, no visual meshes, coarse collision hulls - same contact shapes, several times faster and it scales
    over CPU cores."""
    key = (style, visual)
    if key not in _MODEL_CACHE:
        from pinky_soccer_sim.mujoco_soccer import build_scene
        L = resolve_dims(style)[0]
        out = os.path.join(tempfile.gettempdir(), f'pinky_rl_scene_{os.getpid()}_{int(visual)}')
        model, _ = build_scene(out, style, 20.0, {'yellow': (-L / 4, 0.0, 0.0), 'blue': (L / 4, 0.0, math.pi)},
                               markings=visual, hull_verts=0 if visual else 64, visuals=visual)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


class _Senses:
    """One robot's view of the game: camera ball memory, ball-velocity estimate, own speed, last action."""

    def __init__(self):
        self.ball_mem, self.since_seen = None, 5.0
        self.prev_meas, self.ball_vel = None, np.zeros(2)
        self.meas_vw, self.prev_action = np.zeros(2), np.zeros(2)


class PinkySoccerEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    def __init__(self, stage=1, style='hardboard', episode_s=30.0, randomize=True, render_mode=None,
                 line_guard=True):
        visual = render_mode == 'human'
        super().__init__()
        self.stage, self.randomize, self.render_mode = stage, randomize, render_mode
        self.line_guard = line_guard
        self.opp_speed = 0.20                                    # m/s; train.py raises it step by step
        self.opp_policy, self.opp_policy_path = None, ''
        self.L, self.W, self.G, self.strip = resolve_dims(style)
        self.model = load_model(style, visual)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.dt = 0.1
        self.substeps = round(self.dt / m.opt.timestep)
        self.max_steps = int(episode_s / self.dt)
        self.v_max, self.w_max = 0.30, 4.0
        self.wheel_max, self.wheel_accel = 14.0, 74.9 * m.opt.timestep
        self.bots = {}
        for team in ('yellow', 'blue'):
            p = f'{team}/'
            free = m.joint(f'{p}floating_base')
            self.bots[team] = {
                'ctrl': [m.actuator(f'{p}l_motor').id, m.actuator(f'{p}r_motor').id],
                'q': free.qposadr[0], 'v': free.dofadr[0],
                'radius': float(m.geom(f'{p}l_wheel_collision_0').size[0]),
                'track': float(abs(m.body(f'{p}l_wheel').pos[1] + m.geom(f'{p}l_wheel_collision_0').pos[1]
                                   - m.body(f'{p}r_wheel').pos[1] - m.geom(f'{p}r_wheel_collision_0').pos[1]))}
        body_team = lambda b: next((t for t in ('yellow', 'blue') if m.body(b).name.startswith(t + '/')), None)  # noqa
        self.team_geoms = {t: {g for g in range(m.ngeom) if body_team(m.geom_bodyid[g]) == t} for t in ('yellow', 'blue')}
        bj = m.joint('ball_joint')
        self.ball_q, self.ball_v = bj.qposadr[0], bj.dofadr[0]
        self.ball_geom = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.jnt_bodyid[bj.id]]
        self.ball_friction0 = m.geom_friction[self.ball_geom[0]].copy()
        self.obs_size = OBS_VEL if stage >= 3 else OBS_BASE
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.goal = np.array([self.L / 2, 0.0])
        self.own_goal = np.array([-self.L / 2, 0.0])
        self.viewer = None

    # ------------------------------------------------------------------ state helpers
    def _pose(self, team):
        q = self.data.qpos[self.bots[team]['q']:self.bots[team]['q'] + 7]
        return float(q[0]), float(q[1]), _yaw(q[3:7])

    def _ball(self):
        return self.data.qpos[self.ball_q:self.ball_q + 2].copy()

    def _place(self, team, x, y, yaw):
        b = self.bots[team]
        self.data.qpos[b['q']:b['q'] + 7] = [x, y, 0.03, *_quat(yaw)]
        self.data.qvel[b['v']:b['v'] + 6] = 0.0

    def _place_ball(self, x, y):
        self.data.qpos[self.ball_q:self.ball_q + 7] = [x, y, 0.02135 + 0.003, 1, 0, 0, 0]
        self.data.qvel[self.ball_v:self.ball_v + 6] = 0.0

    def _wheels(self, team, v, w):
        b = self.bots[team]
        r = b['radius'] * self.wheel_scale.get(team, 1.0)       # wrong wheel radius on purpose (calibration)
        left, right = (v - w * b['track'] / 2) / r, (v + w * b['track'] / 2) / r
        return np.clip([left, right], -self.wheel_max, self.wheel_max)

    @staticmethod
    def _to_robot(p, pose):
        x, y, yaw = pose
        dx, dy = p[0] - x, p[1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * dx + s * dy, -s * dx + c * dy])

    @staticmethod
    def _frame(team, p):
        """Field point -> the team's attack frame (attacked goal at +X).  Blue attacks -X: rotate by 180 deg."""
        return np.asarray(p, dtype=float) if team == 'yellow' else -np.asarray(p, dtype=float)

    def _frame_pose(self, team, pose):
        return pose if team == 'yellow' else (-pose[0], -pose[1], _wrap(pose[2] + math.pi))

    # ------------------------------------------------------------------ opponent (stage >= 2)
    def set_opp_speed(self, v):
        self.opp_speed = float(v)

    def load_opponent(self, path):
        """Self-play: a frozen policy (.npz from pinky_soccer_rl.export) drives blue in stage-3 episodes."""
        if path and path != self.opp_policy_path and os.path.exists(path):
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pinky_soccer_game'))
            from pinky_soccer_game.rl_policy import MLPPolicy
            self.opp_policy, self.opp_policy_path = MLPPolicy(path), path
        return self.opp_policy_path

    def _scripted_cmd(self):
        """Scripted opponent: drive to a point behind the ball (seen from OUR goal) and push it that way."""
        x, y, yaw = self._pose('blue')
        b = self._ball()
        u = self.own_goal - b
        u = u / (np.linalg.norm(u) + 1e-9)
        target = b - 0.10 * u
        to = target - np.array([x, y])
        if float(np.linalg.norm(to)) < 0.05:
            to = u
        err = _wrap(math.atan2(to[1], to[0]) - yaw)
        v = self.opp_speed * max(0.0, math.cos(err)) * (1.0 if abs(err) < 0.9 else 0.0)
        return v, max(-3.0, min(3.0, 3.0 * err))

    def _policy_cmd(self):
        """Self-play opponent: the frozen policy sees the game from blue's side (same senses as yellow)."""
        obs = self._team_obs('blue', self.senses['blue'], noisy=False)[:self.opp_policy.obs_size]
        a = self.opp_policy(obs)
        self.senses['blue'].prev_action = a
        return float(a[0] * self.v_max), float(a[1] * self.w_max)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        mujoco.mj_resetData(self.model, self.data)
        L2, W2 = self.L / 2, self.W / 2
        if self.randomize:
            fr = self.ball_friction0.copy()
            fr[2] = rng.uniform(0.0002, 0.0005)                 # ball rolling resistance
            for g in self.ball_geom:
                self.model.geom_friction[g] = fr
            self.wheel_scale = {'yellow': rng.uniform(0.97, 1.03), 'blue': 1.0}
            self.obs_noise = rng.uniform(0.005, 0.02)
            self.delay = int(rng.integers(0, 2))                # 0-1 control periods of latency
        else:
            self.wheel_scale, self.obs_noise, self.delay = {}, 0.0, 0
        kind = rng.random()
        if kind < 0.25:                                         # match start: own start pose, ball on the spot
            self._place('yellow', -(L2 - 0.245), 0.0, 0.0)
            ball = np.array([0.0, 0.0])
        else:                                                   # anywhere
            ball = np.array([rng.uniform(-L2 + 0.12, L2 - 0.12), rng.uniform(-W2 + 0.10, W2 - 0.10)])
            while True:
                rp = np.array([rng.uniform(-L2 + 0.12, L2 - 0.12), rng.uniform(-W2 + 0.10, W2 - 0.10)])
                if np.linalg.norm(rp - ball) > 0.25:
                    break
            self._place('yellow', rp[0], rp[1], rng.uniform(-math.pi, math.pi))
        self._place_ball(*ball)
        # who plays blue: stage 2 - scripted pusher in 60 % of the episodes; stage 3 - 75 %, half of them the
        # frozen self-play policy (when loaded).  options={'opponent': False | 'scripted' | 'policy'} forces it.
        p_active = 0.6 if self.stage == 2 else 0.75
        self.opp_kind = None
        if self.stage >= 2 and rng.random() < p_active:
            self.opp_kind = 'policy' if (self.stage >= 3 and self.opp_policy is not None and rng.random() < 0.5) \
                else 'scripted'
        if options and 'opponent' in options:
            o = options['opponent']
            self.opp_kind = None if (not o or self.stage < 2) else ('scripted' if o is True else o)
            if self.opp_kind == 'policy' and self.opp_policy is None:
                self.opp_kind = 'scripted'
        self.opp_active = self.opp_kind is not None
        if self.opp_active:
            self._place('blue', L2 - 0.245, rng.uniform(-0.2, 0.2), math.pi)
        else:
            self._place('blue', *PARK['blue'], 0.0)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.guarded = False
        self.queue = [np.zeros(2)] * self.delay
        self.ctrl_state = {t: np.zeros(2) for t in self.bots}
        self.senses = {'yellow': _Senses(), 'blue': _Senses()}
        self.prev_d_ball_goal = float(np.linalg.norm(ball - self.goal))
        self.prev_d_robot_ball = float(np.linalg.norm(np.array(self._pose('yellow')[:2]) - ball))
        self.prev_xy = np.array(self._pose('yellow')[:2])
        self.last_touch_t, self.touching = -1.0, False
        self.robot_contact, self.robot_guarded = False, False
        self.idle_steps = 0
        self.priority, self.blue_priority = None, None             # first decision without hysteresis
        self.tie_yellow = bool(rng.random() < 0.5)               # who goes first on a near tie at the ball
        if self.opp_kind == 'policy':
            self._team_obs('blue', self.senses['blue'], noisy=False)      # first camera look for blue
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.queue.append(action)
        a = self.queue.pop(0)
        v, w = a[0] * self.v_max, a[1] * self.w_max
        self.guarded = self.robot_guarded = False
        if self.opp_active:                                     # same no-collision guard as the striker
            self.priority = has_priority(self._pose('yellow'), self._pose('blue'), self._ball(), self.tie_yellow,
                                         had=self.priority)
            v_g, w_g = robot_guard_vw(v, w, self._to_robot(self._pose('blue')[:2], self._pose('yellow')),
                                      self.priority)
            self.robot_guarded = abs(v_g) < abs(v) - 1e-6 or abs(w_g) < abs(w) - 1e-6
            v, w = v_g, w_g
        if self.line_guard and self._over_line_ahead('yellow', v):
            v, self.guarded = 0.0, True                         # same hard stop as the striker's line guard
        targets = {'yellow': self._wheels('yellow', v, w)}
        if self.opp_active:
            bv, bw = self._scripted_cmd() if self.opp_kind == 'scripted' else self._policy_cmd()
            blue_first = has_priority(self._pose('blue'), self._pose('yellow'), self._ball(), not self.tie_yellow,
                                      had=self.blue_priority)
            self.blue_priority = blue_first
            bv, bw = robot_guard_vw(bv, bw, self._to_robot(self._pose('yellow')[:2], self._pose('blue')), blue_first)
            if self.line_guard and self._over_line_ahead('blue', bv):
                bv = 0.0
            targets['blue'] = self._wheels('blue', bv, bw)
        poses0 = {t: self._pose(t) for t in targets}
        for _ in range(self.substeps):
            for team, tgt in targets.items():
                st = self.ctrl_state[team]
                st += np.clip(tgt - st, -self.wheel_accel, self.wheel_accel)
                self.data.ctrl[self.bots[team]['ctrl']] = st
            mujoco.mj_step(self.model, self.data)
        if not self.opp_active:
            self._place('blue', *PARK['blue'], 0.0)
        for team, p0 in poses0.items():                         # measured v, w like the real odometry
            p1 = self._pose(team)
            sign = 1 if math.cos(math.atan2(p1[1] - p0[1], p1[0] - p0[0]) - p0[2]) >= 0 else -1
            self.senses[team].meas_vw = np.array([math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / self.dt * sign,
                                                  _wrap(p1[2] - p0[2]) / self.dt])
        self.meas_vw = self.senses['yellow'].meas_vw
        self.steps += 1
        self.prev_action = self.senses['yellow'].prev_action
        reward, terminated, info = self._reward(action)
        self.senses['yellow'].prev_action = action
        truncated = self.steps >= self.max_steps
        if terminated or truncated:
            info['opp_active'] = int(self.opp_active)
            info['opp_kind'] = self.opp_kind or 'none'
        if truncated and not terminated:
            reward -= 1.0                                        # time up without a goal
            info['time_up'] = 1
        if self.render_mode == 'human':
            self.render()
        if self.opp_kind == 'policy':
            self._team_obs('blue', self.senses['blue'], noisy=False)     # blue's camera this step
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ observation / reward
    def _team_obs(self, team, sn, noisy=True):
        """The observation of `team` (its attack frame), updating its camera memory / ball-velocity estimate."""
        rng = self.np_random
        other = 'blue' if team == 'yellow' else 'yellow'
        pose = self._pose(team)
        if noisy and self.randomize:
            seen_pose = (pose[0] + rng.normal(0, 0.02), pose[1] + rng.normal(0, 0.02), pose[2] + rng.normal(0, 0.03))
        else:
            seen_pose = pose
        ball = self._ball()
        rel = self._to_robot(ball, pose)
        d, bearing = float(np.linalg.norm(rel)), math.atan2(rel[1], rel[0])
        seen = 0.10 < d < 1.6 and abs(bearing) < math.radians(29.5)
        opp_on = self.opp_active
        if seen and opp_on:                                      # the other robot may hide the ball
            o = self._to_robot(np.array(self._pose(other)[:2]), pose)
            od = float(np.linalg.norm(o))
            if od < d and abs(_wrap(math.atan2(o[1], o[0]) - bearing)) < math.atan2(0.07, max(od, 0.05)):
                seen = False
        if seen:
            meas = ball + rng.normal(0, self.obs_noise * (1 + d), 2) if (noisy and self.randomize) else ball
            if sn.prev_meas is not None and sn.since_seen < 0.15:
                sn.ball_vel = 0.5 * sn.ball_vel + 0.5 * np.clip((meas - sn.prev_meas) / self.dt, -1.5, 1.5)
            sn.prev_meas = meas
            sn.ball_mem, sn.since_seen = meas, 0.0
        else:
            sn.since_seen = min(5.0, sn.since_seen + self.dt)
            sn.ball_vel = 0.5 * sn.ball_vel                      # unseen: the estimate fades out
        fp = self._frame_pose(team, seen_pose)                   # everything in the team's attack frame
        mem = self._frame(team, sn.ball_mem) if sn.ball_mem is not None else np.zeros(2)
        b = self._to_robot(mem, fp)
        g = self._to_robot(self.goal, fp)
        o = self._to_robot(self._frame(team, self._pose(other)[:2]), fp) if opp_on else np.zeros(2)
        c, s = math.cos(fp[2]), math.sin(fp[2])
        vf = self._frame(team, sn.ball_vel)
        bv = np.array([c * vf[0] + s * vf[1], -s * vf[0] + c * vf[1]])       # ball velocity in the robot frame
        obs = np.array([b[0], b[1], 1.0 if seen else 0.0, sn.since_seen / 5.0,
                        fp[0] / (self.L / 2), fp[1] / (self.W / 2), math.cos(fp[2]), math.sin(fp[2]),
                        g[0] / self.L, g[1] / self.L, sn.meas_vw[0] / self.v_max, sn.meas_vw[1] / self.w_max,
                        sn.prev_action[0], sn.prev_action[1], o[0], o[1],
                        1.0 if sn.ball_mem is not None else 0.0, bv[0], bv[1]], dtype=np.float32)
        return obs

    def _obs(self):
        return self._team_obs('yellow', self.senses['yellow'])[:self.obs_size]

    def _over_line_ahead(self, team, v):
        """The striker's line guard (striker._line_guard), from the known field lines instead of the camera:
        the strip the robot sweeps next (front 0.05 / rear 0.08 m + 0.03 m + braking 0.25 s, half width 0.075 m)
        reaches past the white boundary line (the goal mouths excepted) -> no motion in that direction."""
        if abs(v) < 1e-3:
            return False
        x, y, yaw = self._pose(team)
        L2, W2, G2 = self.L / 2, self.W / 2, self.G / 2
        reach = (0.05 if v > 0 else -0.08) + math.copysign(0.03 + 0.25 * abs(v), v)
        c, s = math.cos(yaw), math.sin(yaw)
        for along in np.linspace(0.0, reach, 4):
            for side in (-0.075, 0.0, 0.075):
                px, py = x + c * along - s * side, y + s * along + c * side
                if abs(py) > W2 or (abs(px) > L2 and abs(py) > G2 + 0.03):
                    return True
        return False

    def _tilted(self, team, limit_deg=30.0):
        w, x, y, z = self.data.qpos[self.bots[team]['q'] + 3:self.bots[team]['q'] + 7]
        return 1.0 - 2.0 * (x * x + y * y) < math.cos(math.radians(limit_deg))      # body z axis vs up

    def _target(self):
        """Where the robot should be: the spot 12 cm behind the ball on the goal -> ball line (where a push
        starts) - or, stage 3, defending: ball in our half and the opponent closer to it -> 15 cm in front of
        the ball on the ball -> own goal line (between the ball and our goal)."""
        b = self._ball()
        x, y, _ = self._pose('yellow')
        if self.stage >= 3 and self.opp_active and b[0] < 0.0:
            ox, oy, _ = self._pose('blue')
            if math.hypot(ox - b[0], oy - b[1]) < math.hypot(x - b[0], y - b[1]):
                u = self.own_goal - b
                return b + 0.15 * u / (np.linalg.norm(u) + 1e-9), True
        u = self.goal - b
        # stage 3: 18 cm behind (room to get past the ball, turn, THEN push straight); before: 12 cm
        back = 0.18 if self.stage >= 3 else 0.12
        return b - back * u / (np.linalg.norm(u) + 1e-9), False

    def _reward(self, action):
        L2, W2, G2 = self.L / 2, self.W / 2, self.G / 2
        ball = self._ball()
        x, y, _ = self._pose('yellow')
        info = {}
        r = -0.005                                               # time
        d_bg = float(np.linalg.norm(ball - self.goal))
        r += 5.0 * (self.prev_d_ball_goal - d_bg)                # ball towards the goal
        self.prev_d_ball_goal = d_bg
        # robot to the right spot (potential-based: the same target for the old and the new robot position)
        target, defending = self._target()
        xy = np.array([x, y])
        r += 1.0 * (float(np.linalg.norm(self.prev_xy - target)) - float(np.linalg.norm(xy - target)))
        self.prev_xy = xy
        if defending:
            info['defending'] = 1
        d_rb = float(np.linalg.norm(xy - ball))
        r += 0.5 * (self.prev_d_robot_ball - d_rb)               # robot towards the ball
        self.prev_d_robot_ball = d_rb
        r -= 0.02 * float(np.sum((action - self.prev_action) ** 2))    # smooth driving
        # contacts this step: ball touch, pushing the other robot
        ball_g, mine, theirs = self.ball_geom[0], self.team_geoms['yellow'], self.team_geoms['blue']
        touch = push = False
        for i in range(self.data.ncon):
            g1, g2 = self.data.contact[i].geom1, self.data.contact[i].geom2
            if (g1 == ball_g and g2 in mine) or (g2 == ball_g and g1 in mine):
                touch = True
            elif (g1 in mine and g2 in theirs) or (g2 in mine and g1 in theirs):
                push = True
        t = self.steps * self.dt
        if touch and not self.touching and t - self.last_touch_t > 1.0:
            r += 0.1                                             # a new touch of the ball
            self.last_touch_t = t
            info['touch'] = 1
        self.touching = touch
        if touch and self.stage >= 3:
            # dribble like the real robot can: no steering the ball by spinning against it (the simulated bumper
            # cradles the ball; a real ball rolls off) - line up first, then push straight
            w_abs = abs(self.meas_vw[1])
            if w_abs > 1.2:
                r -= 0.05 * min(1.0, (w_abs - 1.2) / 1.5)
                info['spin_touch'] = 1
            u = self.goal - ball
            if abs(_wrap(math.atan2(u[1], u[0]) - self._pose('yellow')[2])) < math.radians(20):
                r += 0.02                                        # pushing in the right direction
        if self.guarded:
            info['guard'] = 1                                    # (the stuck penalty below applies too)
        # the other robot is an obstacle: never touch it, keep away from it
        if push and not self.robot_contact:
            r -= 5.0                                             # a collision (new contact)
            info['collision'] = 1
        if push:
            r -= 0.2                                             # every step in contact
            info['contact'] = 1
        self.robot_contact = push
        if self.opp_active:
            ox, oy, _ = self._pose('blue')
            d = math.hypot(ox - x, oy - y)
            if not self.priority and d < 0.24:                  # giving way: keep the distance
                r -= 0.03 * min(1.0, (0.24 - d) / 0.04)
                info['close'] = 1
        if self.robot_guarded:
            r -= 0.02                                            # had to be stopped by the robot guard
            info['robot_guard'] = 1
        # game flow: never stand around (both robots waiting for each other), never lose the ball for long
        if abs(self.meas_vw[0]) < 0.03 and abs(self.meas_vw[1]) < 0.3:
            self.idle_steps += 1
        else:
            self.idle_steps = 0
        if self.idle_steps > 10:                                 # > 1 s nearly still
            r -= 0.02
            info['idle'] = 1
        if self.senses['yellow'].since_seen > 2.0:               # ball not seen for > 2 s: go and look for it
            r -= 0.01
            info['ball_lost'] = 1
        # told to drive but not moving: stuck on a wall / goal post
        if abs(action[0]) > 0.3 and abs(self.meas_vw[0]) < 0.02 and abs(self.meas_vw[1]) < 0.2:
            r -= 0.03
            info['stuck'] = 1
        # never over the white boundary line (the real robot has a line guard)
        over = max(abs(y) + 0.065 - W2, (abs(x) + 0.065 - L2) if abs(y) > G2 + 0.05 else -1.0)
        if over > 0:
            r -= 0.1
            info['over_line'] = 1
        terminated = False
        if self._tilted('yellow') or (self.opp_active and self._tilted('blue')):
            r -= 10.0                                            # a robot tipped over: never
            return r, True, {**info, 'flipped': 1}
        if ball[0] > L2 + 0.02 and abs(ball[1]) < G2:
            r += 10.0
            terminated, info['goal'] = True, 1
        elif ball[0] < -L2 - 0.02 and abs(ball[1]) < G2:
            r -= 10.0
            terminated, info['own_goal'] = True, 1
        elif abs(ball[0]) > L2 + 0.02 or abs(ball[1]) > W2 + 0.02:
            # out: like the referee, the ball goes back to the centre and play goes on.  (Ending the episode
            # here was a loophole: kicking the ball out was cheaper than playing on until the time ran out.)
            r -= 2.0
            info['out'] = 1
            self._place_ball(0.0, 0.0)
            mujoco.mj_forward(self.model, self.data)
            self.prev_d_ball_goal = float(np.linalg.norm(-self.goal))
            self.prev_d_robot_ball = float(np.linalg.norm(xy))   # no reward jump from the move
            for sn in self.senses.values():
                sn.prev_meas, sn.ball_vel = None, np.zeros(2)
        return r, terminated, info

    def render(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
