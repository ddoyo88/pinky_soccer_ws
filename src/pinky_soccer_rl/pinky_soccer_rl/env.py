"""Gymnasium environment: one Pinky striker, the golf ball, the training field - MuJoCo only (no ROS, no camera
rendering), so it runs far faster than real time.

Same physics as the match simulation (pinky_soccer_sim/mujoco_soccer.py): the official pinky_mujoco robot with
bumper, velocity-controlled wheels with the Dynamixel acceleration limit, rolling golf ball, field from
rcj_field.  The agent plays YELLOW and attacks +X.

Observation (what the real robot has, not the simulator's ground truth):
    ball      in the robot frame (x, y) from the "camera": only inside the camera's view (hfov 59 deg, 0.10 -
              1.6 m), with noise; otherwise the last seen position carried along by the robot's own motion
    seen      1 if the camera sees the ball now, time since it was last seen (s, clipped)
    robot     field pose x / (L/2), y / (W/2), cos(yaw), sin(yaw)   (AMCL on the real robot, with noise)
    goal      attacked goal centre in the robot frame
    speed     measured v, w;   previous action
    opponent  (stage >= 2) other robot in the robot frame (lidar), 0 if none
Action: [v, w] in [-1, 1] -> v * v_max (0.30 m/s), w * w_max (4 rad/s), held for 0.1 s (10 Hz like the striker).

Stages (curriculum):
    1  no opponent (parked outside the field)
    2  scripted opponent that chases the ball and pushes it towards our goal (simple, slower than us)
"""
import math
import os
import tempfile

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from pinky_soccer_sim.field import resolve_dims

PARK = {'blue': (0.0, 3.0)}         # stage 1: the other robot far outside the field (no floor: re-placed every step)


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


class PinkySoccerEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    def __init__(self, stage=1, style='hardboard', episode_s=30.0, randomize=True, render_mode=None,
                 line_guard=True):
        visual = render_mode == 'human'
        super().__init__()
        self.stage, self.randomize, self.render_mode = stage, randomize, render_mode
        self.line_guard = line_guard
        self.opp_speed = 0.20                                    # m/s; train.py raises it step by step
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
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32)
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

    def _to_robot(self, p, pose):
        x, y, yaw = pose
        dx, dy = p[0] - x, p[1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * dx + s * dy, -s * dx + c * dy])

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
        # stage 2: the opponent plays in 60 % of the episodes (the rest: alone, so stage-1 skills stay)
        self.opp_active = self.stage >= 2 and rng.random() < 0.6
        if options and 'opponent' in options:                   # evaluation: with / without on purpose
            self.opp_active = self.stage >= 2 and bool(options['opponent'])
        if self.opp_active:
            self._place('blue', L2 - 0.245, rng.uniform(-0.2, 0.2), math.pi)
        else:
            self._place('blue', *PARK['blue'], 0.0)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.prev_action = np.zeros(2)
        self.guarded = False
        self.queue = [np.zeros(2)] * self.delay
        self.meas_vw = np.zeros(2)
        self.ctrl_state = {t: np.zeros(2) for t in self.bots}
        self.ball_mem, self.since_seen = None, 5.0
        self.prev_d_ball_goal = float(np.linalg.norm(ball - self.goal))
        self.prev_d_robot_ball = float(np.linalg.norm(np.array(self._pose('yellow')[:2]) - ball))
        self.last_touch_t, self.touching = -1.0, False
        self.prev_d_behind = self._d_behind()
        return self._obs(), {}

    def _opponent_cmd(self):
        """Stage 2 opponent: drive to a point behind the ball (seen from OUR goal) and push it that way."""
        x, y, yaw = self._pose('blue')
        b = self._ball()
        u = self.own_goal - b
        u = u / (np.linalg.norm(u) + 1e-9)
        target = b - 0.10 * u
        to = target - np.array([x, y])
        dist = float(np.linalg.norm(to))
        if dist < 0.05:
            to = u
        err = _wrap(math.atan2(to[1], to[0]) - yaw)
        v = self.opp_speed * max(0.0, math.cos(err)) * (1.0 if abs(err) < 0.9 else 0.0)
        return v, max(-3.0, min(3.0, 3.0 * err))

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.queue.append(action)
        a = self.queue.pop(0)
        v, w = a[0] * self.v_max, a[1] * self.w_max
        self.guarded = False
        if self.line_guard and self._over_line_ahead(v):
            v, self.guarded = 0.0, True                         # same hard stop as the striker's line guard
        targets = {'yellow': self._wheels('yellow', v, w)}
        if self.opp_active:
            targets['blue'] = self._wheels('blue', *self._opponent_cmd())
        pose0 = self._pose('yellow')
        for _ in range(self.substeps):
            for team, tgt in targets.items():
                st = self.ctrl_state[team]
                st += np.clip(tgt - st, -self.wheel_accel, self.wheel_accel)
                self.data.ctrl[self.bots[team]['ctrl']] = st
            mujoco.mj_step(self.model, self.data)
        if not self.opp_active:
            self._place('blue', *PARK['blue'], 0.0)
        pose1 = self._pose('yellow')
        self.meas_vw = np.array([math.hypot(pose1[0] - pose0[0], pose1[1] - pose0[1]) / self.dt *
                                 (1 if math.cos(math.atan2(pose1[1] - pose0[1], pose1[0] - pose0[0]) - pose0[2]) >= 0
                                  else -1),
                                 _wrap(pose1[2] - pose0[2]) / self.dt])
        self.steps += 1
        reward, terminated, info = self._reward(action)
        self.prev_action = action
        truncated = self.steps >= self.max_steps
        if terminated or truncated:
            info['opp_active'] = int(self.opp_active)
        if truncated and not terminated:
            reward -= 1.0                                        # time up without a goal
            info['time_up'] = 1
        if self.render_mode == 'human':
            self.render()
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ observation / reward
    def _obs(self):
        rng = self.np_random
        pose = self._pose('yellow')
        noisy = (pose[0] + rng.normal(0, 0.02), pose[1] + rng.normal(0, 0.02), pose[2] + rng.normal(0, 0.03)) \
            if self.randomize else pose
        ball = self._ball()
        rel = self._to_robot(ball, pose)
        d, bearing = float(np.linalg.norm(rel)), math.atan2(rel[1], rel[0])
        seen = 0.10 < d < 1.6 and abs(bearing) < math.radians(29.5)
        if seen and self.opp_active:                            # the other robot may hide the ball
            o = self._to_robot(np.array(self._pose('blue')[:2]), pose)
            od = float(np.linalg.norm(o))
            if od < d and abs(_wrap(math.atan2(o[1], o[0]) - bearing)) < math.atan2(0.07, max(od, 0.05)):
                seen = False
        if seen:
            meas = ball + rng.normal(0, self.obs_noise * (1 + d), 2) if self.randomize else ball
            self.ball_mem, self.since_seen = meas, 0.0
        else:
            self.since_seen = min(5.0, self.since_seen + self.dt)
        mem = self.ball_mem if self.ball_mem is not None else np.array([0.0, 0.0])
        b = self._to_robot(mem, noisy)
        g = self._to_robot(self.goal, noisy)
        if self.opp_active:
            o = self._to_robot(np.array(self._pose('blue')[:2]), noisy)
        else:
            o = np.zeros(2)
        obs = np.array([b[0], b[1], 1.0 if seen else 0.0, self.since_seen / 5.0,
                        noisy[0] / (self.L / 2), noisy[1] / (self.W / 2), math.cos(noisy[2]), math.sin(noisy[2]),
                        g[0] / self.L, g[1] / self.L, self.meas_vw[0] / self.v_max, self.meas_vw[1] / self.w_max,
                        self.prev_action[0], self.prev_action[1], o[0], o[1],
                        1.0 if self.ball_mem is not None else 0.0], dtype=np.float32)
        return obs

    def set_opp_speed(self, v):
        self.opp_speed = float(v)

    def _over_line_ahead(self, v):
        """The striker's line guard (striker._line_guard), from the known field lines instead of the camera:
        the strip the robot sweeps next (front 0.05 / rear 0.08 m + 0.03 m + braking 0.25 s, half width 0.075 m)
        reaches past the white boundary line (the goal mouths excepted) -> no motion in that direction."""
        if abs(v) < 1e-3:
            return False
        x, y, yaw = self._pose('yellow')
        L2, W2, G2 = self.L / 2, self.W / 2, self.G / 2
        reach = (0.05 if v > 0 else -0.08) + math.copysign(0.03 + 0.25 * abs(v), v)
        c, s = math.cos(yaw), math.sin(yaw)
        for along in np.linspace(0.0, reach, 4):
            for side in (-0.075, 0.0, 0.075):
                px, py = x + c * along - s * side, y + s * along + c * side
                if abs(py) > W2 or (abs(px) > L2 and abs(py) > G2 + 0.03):
                    return True
        return False

    def _d_behind(self):
        """Distance from the robot to the spot 12 cm behind the ball on the goal -> ball line (where a push
        starts).  Potential-based shaping: guides without changing the best play."""
        b = self._ball()
        u = self.goal - b
        u = u / (np.linalg.norm(u) + 1e-9)
        x, y, _ = self._pose('yellow')
        return float(np.linalg.norm(np.array([x, y]) - (b - 0.12 * u)))

    def _reward(self, action):
        L2, W2, G2 = self.L / 2, self.W / 2, self.G / 2
        ball = self._ball()
        x, y, _ = self._pose('yellow')
        info = {}
        r = -0.005                                               # time
        d_bg = float(np.linalg.norm(ball - self.goal))
        r += 5.0 * (self.prev_d_ball_goal - d_bg)                # ball towards the goal
        d_bh = self._d_behind()
        r += 1.0 * (self.prev_d_behind - d_bh)                   # robot to the spot behind the ball
        self.prev_d_behind = d_bh
        self.prev_d_ball_goal = d_bg
        d_rb = float(np.linalg.norm(np.array([x, y]) - ball))
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
        if self.guarded:
            info['guard'] = 1                                    # (the stuck penalty below applies too)
        if push and action[0] * self.meas_vw[0] >= 0 and abs(action[0]) > 0.2:
            r -= 0.05                                            # driving into the other robot
            info['push'] = 1
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
            self.prev_d_behind = self._d_behind()
            self.prev_d_robot_ball = float(np.linalg.norm(np.array([x, y])))   # no reward jump from the move
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
