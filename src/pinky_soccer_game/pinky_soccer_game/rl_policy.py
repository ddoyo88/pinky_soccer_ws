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
    # bumpers touch at 0.18 m head-on, 0.21 m corner to corner: 0.155 m let the robots press into each other
    # (user, 2026-11-26: '두 로봇이 너무 가까이 붙는다 - 부딪히지 않게') -> ~3 cm clear even with the right of way
    # minimal avoidance, bold play (user, 2026-11-26: '상대 회피는 최소한으로'): stop just short of the bumpers
    # no collisions, but slow down early and smoothly (user, 2026-11-26): stop before the bumpers meet (0.18 m
    # head-on, 0.21 m corner to corner), slowing from 0.12 m before that; the one closer to the ball goes first
    stop = 0.22 if priority is None else (0.20 if priority else 0.26)
    d = math.hypot(ox, oy)
    return v * max(0.0, min(1.0, (d - stop) / 0.12))


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


import time as _time
clock = _time.time          # straight_cmd's timers (the offline tuner swaps in simulated time)

def ball_filter(state, meas, jump=0.06, alpha=0.3):
    """Camera ball position noise (a few cm) near the spot behind the ball swung the target direction and the
    robot turned back and forth on the spot (~4 rad in 3 s, g21 diagnostics, 2026-11-26).  Small changes are
    averaged (EMA), a real move (> jump) is taken at once."""
    f = state.get('ball_f')
    if f is None or math.hypot(meas[0] - f[0], meas[1] - f[1]) > jump:
        f = (float(meas[0]), float(meas[1]))
    else:
        f = (f[0] + alpha * (meas[0] - f[0]), f[1] + alpha * (meas[1] - f[1]))
    state['ball_f'] = f
    return f


# straight_cmd tuning (scripts/dribble_tune.py)
SC = {'lined_in': 0.92, 'lined_keep': 0.75, 'to_in': 0.45, 'to_keep': 0.7, 'v_near': 1.0, 'aim_k': 1.0,
      'behind': 0.16, 'aim_lined': 2.0, 'behind_tol': 0.06,
      'orbit': 1.0, 'orbit_r': 0.18, 'orbit_step': 1.1,
      'look_age': 0.6, 'w_turn': 1.2}


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
            return 0.0, max(-SC['w_turn'], min(SC['w_turn'], 2.5 * err))
        return vmax * max(0.3, math.cos(err)), max(-1.5, min(1.5, 2.0 * err))

    now = clock()
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
        return 0.0, max(-SC['w_turn'], min(SC['w_turn'], 2.5 * to_ball)), 'WAIT'
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
    # hysteresis: once pushing, keep pushing while still roughly lined up - the ball drifting a few degrees
    # flipped PUSH <-> BEHIND every 0.1 s and the robot stood turning instead of driving (g03, 2026-11-26)
    pushing = state.get('sc_mode') == 'PUSH' and now - state.get('sc_t', 0.0) < 0.3
    if (lined > SC['lined_in'] and abs(to_ball) < SC['to_in']) or \
            (pushing and lined > SC['lined_keep'] and abs(to_ball) < SC['to_keep'] and d < 0.35):
        aim = _wrap(math.atan2(gy - by, gx - bx) - yaw)    # (1) lined up: run through the ball toward the goal
        w = 2.5 * to_ball if d > 0.2 else 1.5 * to_ball + SC['aim_k'] * aim
        state['sc_mode'], state['sc_t'] = 'PUSH', now
        return v_max * (1.0 if d > 0.2 else SC['v_near']), max(-1.5, min(1.5, w)), 'PUSH'
    state['sc_mode'] = None
    if SC['orbit']:
        return _orbit_cmd(x, y, yaw, bx, by, ux, uy, d, to_ball, half_w, half_len, v_max, state, now, drive)
    if behind == 0.16:
        behind = SC['behind']
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
    # close and roughly behind it already: face the ball and push - creeping onto the exact spot behind the
    # ball (turn on the spot, 5 cm, turn again) cost seconds at every approach (g16 diagnostics, 2026-11-26)
    if math.hypot(px - x, py - y) < SC['behind_tol'] or (d < 0.30 and abs(to_ball) < 0.3) or \
            (d < 0.30 and lined > SC['aim_lined']):
        # behind it (or the spot behind it is clamped off a line / goal: never quite lined up): face the ball,
        # then push it - standing at AIM with nothing to do froze the game for minutes (final5e)
        if abs(to_ball) < 0.3:
            return 0.7 * v_max, max(-1.5, min(1.5, 2.5 * to_ball)), 'PUSH'
        return 0.0, max(-SC['w_turn'], min(SC['w_turn'], 2.5 * to_ball)), 'AIM'
    v, w = drive(px, py, v_max)                            # (3) to the point behind the ball
    return v, w, 'BEHIND'


def _orbit_cmd(x, y, yaw, bx, by, ux, uy, d, to_ball, half_w, half_len, v_max, state, now, drive):
    """Getting behind the ball (user, 2026-11-26): keep ~5 cm off the ball on a circle round it and walk
    round that circle in short steps - a step along the circle, then turn and look at the ball (so the camera
    sees it at least about once a second), again - until standing on the far side from the goal, then push.
    The old wide detour (a point 0.22 m beside the ball, then the spot behind it) went far off, lost the ball
    out of the camera's view and spent seconds turning on the spot."""
    R = SC['orbit_r']                                      # robot centre - ball centre (bumper ~5 cm off)
    phi = math.atan2(y - by, x - bx)                       # where we stand round the ball
    phi_t = math.atan2(-uy, -ux)                           # where we want to stand: opposite the goal

    def inside(a):
        return abs(by + R * math.sin(a)) < half_w - 0.07 and abs(bx + R * math.cos(a)) < half_len - 0.07

    if not inside(phi_t):
        # ball near a line: the spot opposite the goal lies over the line - take the nearest spot on the circle
        # that is inside (the push then runs the ball along the line toward the goal)
        for k in range(1, 12):
            if inside(phi_t + 0.15 * k):
                phi_t += 0.15 * k
                break
            if inside(phi_t - 0.15 * k):
                phi_t -= 0.15 * k
                break
    dphi = _wrap(phi_t - phi)
    # keep the direction round the ball once chosen (standing right on the goal side, dphi ~ +-pi flipped
    # the way round every tick)
    if abs(dphi) < 0.25:
        state.pop('orb_dir', None)
    if 'orb_dir' not in state:
        # pick the way round once: the shorter one, unless it runs over a line (ball near a line) - then the
        # way that stays inside the field
        best = None
        for sg in (1.0, -1.0):
            arc = (dphi if sg > 0 else -dphi) % (2 * math.pi)
            n = max(1, int(arc / 0.3))
            bad = sum(1 for i in range(1, n + 1)
                      if abs(by + R * math.sin(phi + sg * arc * i / n)) > half_w - 0.07 or
                      abs(bx + R * math.cos(phi + sg * arc * i / n)) > half_len - 0.07)
            key = (bad, arc)
            if best is None or key < best[0]:
                best = (key, sg)
        state['orb_dir'] = best[1]
    sgn = state['orb_dir']
    if abs(dphi) > 2.6 and (dphi >= 0) != (sgn > 0):
        dphi = dphi - math.copysign(2 * math.pi, dphi)       # the long way only near the opposite point
    elif (dphi >= 0) != (sgn > 0):
        sgn = state['orb_dir'] = 1.0 if dphi >= 0 else -1.0

    def on_circle(a):
        tx, ty = bx + R * math.cos(a), by + R * math.sin(a)
        inside = abs(ty) < half_w - 0.07 and abs(tx) < half_len - 0.07
        return tx, max(-half_w + 0.07, min(half_w - 0.07, ty)), inside

    def face_ball(tag):
        return 0.0, max(-SC['w_turn'], min(SC['w_turn'], 2.5 * to_ball)), tag

    if d > R + 0.08:
        # coming in from further away: head for the near side of the circle, already turned toward the back
        # of the ball - the ball stays (almost) straight ahead in the camera the whole way
        state['orb_phase'] = 'LOOK'
        tx, ty, _ = on_circle(phi + max(-0.6, min(0.6, dphi)))
        tx = max(-half_len + 0.07, min(half_len - 0.07, tx))
        v, w = drive(tx, ty, v_max)
        return v, w, 'APPROACH'
    if abs(dphi) < 0.25:
        # behind the ball: face it, then push it
        state['orb_phase'] = 'LOOK'
        if abs(to_ball) < 0.3:
            state['sc_mode'], state['sc_t'] = 'PUSH', now
            # straight runs at full speed (user, 2026-11-26: fast straight / shots, slow turns)
            return (v_max if abs(to_ball) < 0.15 else 0.7 * v_max), max(-1.5, min(1.5, 2.5 * to_ball)), 'PUSH'
        return face_ball('AIM')
    # walk round the circle continuously (aim at a point further round it), and turn to the ball for a look
    # whenever it has been out of the camera's view for 1 s (stopping to look after every short step was too
    # slow: the attacker never got round, g28)
    age = state.get('ball_age', 0.0)
    looking = state.get('orb_phase') == 'LOOK'
    if age > SC['look_age'] and not looking:
        looking, state['orb_phase'], state['orb_t'] = True, 'LOOK', now
    if looking:
        if (age < 0.2 and abs(to_ball) < 0.3) or now - state.get('orb_t', now) > 1.2:
            state['orb_phase'] = 'STEP'
        else:
            if d < R - 0.04:                               # too close: back off a little, eyes on the ball
                return -0.08, max(-SC['w_turn'], min(SC['w_turn'], 2.5 * to_ball)), 'ORBIT_LOOK'
            return face_ball('ORBIT_LOOK')
    if d < R - 0.05:
        # too close to the ball (it rolled at us): back off onto the circle, eyes on the ball - turning on the
        # spot right next to it spun the robot round 6 rad in 3 s and caught the ball under a wheel (h01)
        return -0.10, max(-SC['w_turn'], min(SC['w_turn'], 2.5 * to_ball)), 'ORBIT_LOOK'
    step = min(abs(dphi), SC['orbit_step'])
    tx, ty, ok = on_circle(phi + sgn * step)
    if not ok:
        tx2, ty2, ok2 = on_circle(phi - sgn * step)
        if ok2:
            tx, ty = tx2, ty2
            state['orb_dir'] = -sgn                          # a line that way: go round the other side
    tx = max(-half_len + 0.07, min(half_len - 0.07, tx))
    err = _wrap(math.atan2(ty - y, tx - x) - yaw)
    if abs(err) > 0.6:
        return 0.0, max(-SC['w_turn'], min(SC['w_turn'], 2.5 * err)), 'ORBIT_STEP'
    return 0.8 * v_max, max(-1.5, min(1.5, 2.5 * err)), 'ORBIT_STEP'


def defend_cmd(me, ball, own_goal_x, half_w, goal_w, v_max, state, clear_dist=0.70):
    """Defender (attack/defence game, 2026-11-25): stand in front of the own goal on the goal -> ball line and
    face the ball; when the ball comes close to the goal, clear it - push it toward the centre line on its own
    side (out of the goal's way, over the centre line or out of the field).
    me (x, y, yaw), ball (x, y) or None, own_goal_x: x of our goal line.  Returns (v, w, mode)."""
    x, y, yaw = me
    inward = -math.copysign(1.0, own_goal_x)              # from our goal line into the field
    if ball is None:
        bx, by = own_goal_x + inward * 0.8, 0.0
    else:
        bx, by = ball
    d_goal = math.hypot(bx - own_goal_x, by)
    clearing = state.get('clearing', False)
    if ball is not None and (d_goal < clear_dist or (clearing and d_goal < clear_dist + 0.10)):
        state['clearing'] = True
        # clear it into the other half, toward the side line on the ball's side: over the centre line or out
        # of the field - both end the attack (user, 2026-11-26: '상대 진영이든 바깥으로')
        aim = (inward * 0.45, math.copysign(half_w + 0.10, by if abs(by) > 0.02 else 1.0))
        v, w, mode = straight_cmd(me, (bx, by), aim, half_w, v_max, state, half_len=abs(own_goal_x))
        return v, w, 'CLEAR_' + mode
    state['clearing'] = False
    # keeper spot: 0.22 m out from the goal centre toward the ball, inside the goal width (+2 cm)
    dx, dy = bx - own_goal_x, by
    n = math.hypot(dx, dy) or 1e-6
    kx, ky = own_goal_x + dx / n * 0.22, dy / n * 0.22
    if inward * (kx - own_goal_x) < 0.14:
        kx = own_goal_x + inward * 0.14
    ky = max(-(goal_w / 2 + 0.02), min(goal_w / 2 + 0.02, ky))
    face = math.atan2(by - y, bx - x)
    dist = math.hypot(kx - x, ky - y)
    keeping = state.get('keep', False)
    if dist < 0.04 or (keeping and dist < 0.09):           # on the spot (hysteresis: no KEEP/GO_KEEP flicker)
        state['keep'] = True
        err = _wrap(face - yaw)
        return 0.0, (max(-SC['w_turn'], min(SC['w_turn'], 2.5 * err)) if abs(err) > 0.1 else 0.0), 'KEEP'
    state['keep'] = False
    to = math.atan2(ky - y, kx - x)
    err = _wrap(to - yaw)
    # short moves: drive backwards when the spot lies behind (keep facing the play instead of turning round)
    if abs(err) > math.pi / 2 and dist < 0.35:
        err_b = _wrap(to + math.pi - yaw)
        if abs(err_b) > 0.5:
            return 0.0, max(-2.0, min(2.0, 2.5 * err_b)), 'GO_KEEP'
        return -0.6 * v_max, max(-1.5, min(1.5, 2.0 * err_b)), 'GO_KEEP'
    if abs(err) > 0.5:
        return 0.0, max(-SC['w_turn'], min(SC['w_turn'], 2.5 * err)), 'GO_KEEP'
    return v_max * max(0.3, math.cos(err)), max(-1.5, min(1.5, 2.0 * err)), 'GO_KEEP'


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
    if d < 0.23:                                     # body corners sweep ~0.105 m round the centre
        w *= max(0.0, min(1.0, (d - 0.17) / 0.06))
    # personal space 0.17 m; giving way (the other robot has the right of way at the ball): 0.24 m - backing
    # off frees the ball for it (otherwise a ball between the two robots stayed there: r2s2, 2026-11-25)
    space = 0.24 if priority is False else 0.19
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
