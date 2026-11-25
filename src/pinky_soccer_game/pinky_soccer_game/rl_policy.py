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


def robot_guard(v, opp_rel, priority=None, corridor=0.20):
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
    # tighter (2026-11-25, user: 'smaller avoidance distance, just no contact'): bumpers touch at ~0.15 m
    stop = 0.18 if priority is None else (0.155 if priority else 0.20)
    d = math.hypot(ox, oy)
    return v * max(0.0, min(1.0, (d - stop) / 0.06))


def over_line_ahead(pose, v, half_len, half_wid, goal_w):
    """The white boundary line from the robot's own map pose (shared by the RL environment and the striker,
    next to the camera line guard): the strip the robot sweeps next - front 0.05 / rear 0.08 m + 0.03 m +
    braking 0.25 s, half width 0.075 m - reaches past the boundary line (goal mouths excepted).  Works in
    every direction (the camera only sees ahead: reversing / turning crossed the line, 2026-11-25)."""
    if abs(v) < 1e-3:
        return False
    x, y, yaw = pose
    reach = (0.05 if v > 0 else -0.08) + math.copysign(0.03 + 0.25 * abs(v), v)
    c, s = math.cos(yaw), math.sin(yaw)

    def excess(px, py):                                   # how far past the boundary line (<= 0: inside)
        e = abs(py) - half_wid
        if abs(py) > goal_w / 2 + 0.03:
            e = max(e, abs(px) - half_len)
        return e

    for side in (-0.075, 0.0, 0.075):
        now = excess(x - s * side, y + c * side)
        for k in range(1, 4):
            along = reach * k / 3.0
            e = excess(x + c * along - s * side, y + s * along + c * side)
            # only motion that goes (further) out is stopped: a robot already on the line must still be able
            # to drive back in (both robots froze on the side line, final5k)
            if e > 0.0 and e > now + 1e-3:
                return True
    return False


def straight_cmd(me, ball, goal, half_w, v_max, state, behind=0.16, give_way=False, free=True, half_len=0.83):
    """Straight-line ball play (2026-11-25, user: 'no zig-zag at all - approaching a far ball or dribbling is a
    straight action'): replaces the learned policy's weaving for the approach and the dribble.
      1) robot - ball - goal lined up: straight at the ball at full speed and push it on (heading P-control)
      2) robot on the goal side of the ball: go round it via a point beside the ball (never push it backwards)
      3) otherwise: to the point behind the ball (on the goal -> ball line) - turn on the spot first when the
         heading is off, then drive straight with a gentle correction
    me (x, y, yaw), ball (x, y), goal (x, y) in the map.  Returns (v, w, mode)."""
    x, y, yaw = me
    bx, by = ball
    gx, gy = goal
    ux, uy = gx - bx, gy - by                              # ball -> goal
    n = math.hypot(ux, uy) or 1e-6
    ux, uy = ux / n, uy / n
    rx, ry = bx - x, by - y                                # robot -> ball
    d = math.hypot(rx, ry) or 1e-6
    to_ball = _wrap(math.atan2(ry, rx) - yaw)
    lined = (rx * ux + ry * uy) / d                        # cos(robot->ball, ball->goal)

    def drive(tx, ty, vmax):
        err = _wrap(math.atan2(ty - y, tx - x) - yaw)
        if abs(err) > 0.5:                                 # turn on the spot first: no curved weaving
            return 0.0, max(-2.0, min(2.0, 2.5 * err))
        return vmax * max(0.3, math.cos(err)), max(-1.5, min(1.5, 2.0 * err))

    import time as _t
    now = _t.time()
    if give_way:
        # give way at most 3 s per stand-off, then go in ourselves (both stood still for minutes, final5e)
        t0 = state.setdefault('gw_t', now)
        if now - t0 > 3.0:
            give_way = False
            if now - t0 > 6.0:
                state.pop('gw_t', None)
    else:
        state.pop('gw_t', None)
    if give_way:
        # face-off at the ball and the other robot is closer: it goes first, we step back and wait 0.40 m off
        if d < 0.40:
            return -0.15, max(-1.5, min(1.5, 2.0 * to_ball)), 'GIVE_WAY'
        return 0.0, max(-2.0, min(2.0, 2.5 * to_ball)), 'WAIT'
    near_goal = math.hypot(gx - bx, gy - by) < 0.60
    shot = state.get('shot')
    if shot is not None:
        # shot (free = the other robot > 0.5 m from the ball, else backing off loses the ball): step back a
        # little, then hit the ball at full speed - an impact, not a push
        if shot == 'back':
            if d > 0.24 or abs(to_ball) > 0.6:
                state['shot'] = 'charge'
            return -0.20, max(-1.5, min(1.5, 2.0 * to_ball)), 'SHOT_BACK'
        if d < 0.30 and abs(to_ball) < 0.6:
            return v_max, max(-1.0, min(1.0, 2.0 * to_ball)), 'SHOT'
        state.pop('shot', None)
    if near_goal and lined > 0.93 and abs(to_ball) < 0.3 and d < 0.16 and free:
        state['shot'] = 'back'
        return -0.20, 0.0, 'SHOT_BACK'
    if lined > 0.92 and abs(to_ball) < 0.45:               # (1) lined up: run through the ball toward the goal
        aim = _wrap(math.atan2(gy - by, gx - bx) - yaw)
        w = 2.5 * to_ball if d > 0.2 else 1.5 * to_ball + 1.0 * aim
        return v_max, max(-1.5, min(1.5, w)), 'PUSH'
    px, py = bx - ux * behind, by - uy * behind            # point behind the ball
    py = max(-half_w + 0.10, min(half_w - 0.10, py))
    px = max(-half_len + 0.12, min(half_len - 0.12, px))   # never into a goal mouth (blue got stuck, final5d)
    if (x - bx) * ux + (y - by) * uy > -0.05:              # (2) level with / in front of the ball: round it
        side = 1.0 if (-(uy) * (x - bx) + ux * (y - by)) >= 0 else -1.0
        sx, sy = bx - uy * side * 0.22 - ux * 0.05, by + ux * side * 0.22 - uy * 0.05
        sy = max(-half_w + 0.10, min(half_w - 0.10, sy))
        sx = max(-half_len + 0.12, min(half_len - 0.12, sx))
        if abs(sy - by) < 0.12:                            # no room on that side (ball at a line): other side
            sx, sy = bx + uy * side * 0.22 - ux * 0.05, by - ux * side * 0.22 - uy * 0.05
        v, w = drive(sx, sy, v_max)
        return v, w, 'ROUND'
    if math.hypot(px - x, py - y) < 0.06 or (d < 0.30 and abs(to_ball) < 0.3):
        # behind it (or the spot behind it is clamped off a line / goal: never quite lined up): face the ball,
        # then push it - standing at AIM with nothing to do froze the game for minutes (final5e)
        if abs(to_ball) < 0.3:
            return 0.7 * v_max, max(-1.5, min(1.5, 2.5 * to_ball)), 'PUSH'
        return 0.0, max(-2.0, min(2.0, 2.5 * to_ball)), 'AIM'
    v, w = drive(px, py, v_max)                            # (3) to the point behind the ball
    return v, w, 'BEHIND'


def rl_post_process(v, w, obs, state, v_max, smooth=True, push=True, field_len=1.66, alpha=0.5):
    """Make the learned play look like football, not wiggling (2026-11-25, user: 'it sways left and right on the
    way to the ball and keeps turning while dribbling'):
      * smooth: the policy flips the turn rate between +-4 rad/s from one 0.1 s step to the next (it was never
        penalised for it) -> low-pass the turn rate (EMA); the policy keeps seeing its own raw action
      * push: ball sitting on the nose and the goal ahead -> a straight push toward the goal at full speed, the
        turn rate only for keeping the ball centred and aiming (the policy zig-zagged the ball forward)
    obs: the 19-value policy input (ball / goal in the robot frame).  state: dict kept by the caller."""
    bx, by, seen = obs[0], obs[1], obs[2] > 0.5
    gx, gy = obs[8] * field_len, obs[9] * field_len
    pushing = False
    if push and seen and 0.04 < bx < 0.20 and abs(by) < 0.05:
        aim = math.atan2(gy - by, gx - bx)               # goal direction seen from the ball
        if abs(aim) < 0.6 and gx > 0.0:
            v = v_max
            w = max(-1.5, min(1.5, 2.0 * aim + 2.5 * math.atan2(by, bx)))
            pushing = True
    elif push and seen and 0.04 < bx < 0.9 and gx > 0.0:
        # lined up (robot - ball - goal on one line, kick-off and every run from behind the ball): straight at
        # the ball at full speed instead of the policy's slow weaving approach
        to_ball = math.atan2(by, bx)
        aim = math.atan2(gy - by, gx - bx)
        if abs(to_ball) < 0.5 and abs(_wrap(aim - to_ball)) < 0.35:
            v = v_max
            w = max(-1.5, min(1.5, 3.0 * to_ball))
            pushing = True
    if smooth and not pushing:
        w = alpha * state.get('w', w) + (1.0 - alpha) * w
    state['w'] = w
    state['pushing'] = pushing
    return v, w


def robot_guard_vw(v, w, opp_rel, priority=None):
    """robot_guard for the whole command (2026-11-25): head-on driving as above (corridor widened to 0.20 m),
    plus what it did not cover - side-by-side contact:
      * turning close to the other robot: the body corners lie ~0.10 m from the centre, so the turn rate is
        scaled down from 0.20 m centre distance and is zero at 0.14 m;
      * closer than 0.17 m (personal space; 0.24 m when giving way at the ball): drive slowly (0.08 m/s) away
        from the other robot - both robots do this, so they separate."""
    v = robot_guard(v, opp_rel, priority)
    if opp_rel is None:
        return v, w
    ox, oy = float(opp_rel[0]), float(opp_rel[1])
    d = math.hypot(ox, oy)
    if d < 0.18:
        w *= max(0.0, min(1.0, (d - 0.14) / 0.04))
    # personal space 0.17 m; giving way (the other robot has the right of way at the ball): 0.24 m - backing
    # off frees the ball for it (otherwise a ball between the two robots stayed there: r2s2, 2026-11-25)
    space = 0.20 if priority is False else 0.155
    if d < space:
        away = -0.08 if ox > 0.0 else 0.08          # other robot in front -> back up, behind/beside -> forward
        if v * away <= 0.0 or abs(v) < abs(away):
            v = away
    return v, w


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
