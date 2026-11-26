"""Rule-based ball play: look - small move - look again (2026-11-26, replaces the learned policy).

The user's rule: find the ball, move a little toward where you want to stand, look at the ball again at once,
repeat.  Turning is slow (the camera must not lose the ball, 10 frames/s), straight driving can be fast.
When the robot stands behind the ball it pushes with the ball held in the middle of the camera image (small
steering corrections only); off the middle -> stop and centre it again.

attack_cmd(...)  -> (v, w, mode)  get behind the ball (opposite the target) and push / shoot it at the target
defend_cmd(...)  -> (v, w, mode)  keeper in front of the own goal; rush and clear a ball that comes close
"""
import math

# speeds halved 2026-11-27 (user: the robots were too fast to control the ball).
# numbers (2026-11-26, user: exact lining up was too slow and frustrating - move quickly, touch the ball a lot):
# push once roughly behind the ball (0.30 rad), steps of 14 cm at 0.24 m/s, 0.7 rad round the ball per step.  (Learned numbers: scripts/tune_ball_play.py, ~/pinky_rl/ball_play_params*.json)
P = {
    'w_look': 0.9,        # rad/s max turn with the ball in / near the camera view (slow: no overshoot, no blur)
    'w_far': 1.6,         # rad/s max turn with the ball well outside the view (nothing to lose there)
    'view': 0.55,         # rad: 'near the camera view' (measured: the ball is seen within +-0.45 rad)
    'line_slow': 0.18,    # m from the boundary line: ...
    'v_line': 0.08,       # ... drive at most this fast there (p02: taps / clears still crossed it by up to 4 cm)
    'k_look': 2.0,        # turn gain
    'look_tol': 0.15,     # rad: ball 'in the middle of the image'
    'look_fresh': 0.3,   # s: a camera sighting this recent counts as 'seen now'
    'v_approach': 0.13,   # m/s driving at the ball while it stays in view (no stopping needed)
    'step_d': 0.14,       # m: one small move
    'step_t': 0.6,        # s: ... or at most this long
    'step_turn': 0.7,     # rad: start the move once the heading is this close to it
    'v_step': 0.12,       # m/s small moves
    'v_push': 0.11,       # m/s dribble
    'v_shot': 0.14,       # m/s shot (near the goal, lined up, nobody at the ball)
    'push_tol': 0.2,     # rad: ball drifting this far off the middle while pushing -> stop and centre it
    'k_push': 1.6,        # steering gain while pushing (fine corrections)
    'w_push': 0.5,        # rad/s max steering while pushing
    'orbit_r': 0.18,      # m robot centre - ball centre while going round it (~5 cm bumper - ball)
    'orbit_step': 0.7,    # rad round the ball per small move
    'behind_tol': 0.3,   # rad round the ball: 'behind it'
    'line_margin': 0.07,  # m: move targets stay this far inside the boundary line
    # going round the ball on a square (user, 2026-11-26): straight legs corner to corner, a look after each
    'square_h': 0.12,     # m ball centre -> square side (user: as small as possible; side 5.5 cm + ball 2.1 cm
                          #   -> ~4 cm clear along a side, corners 0.17 m off: the body corners clear the ball)
    'def_style': 'block', # defender: 'block' (never touches the ball, stands in the shot's way) | 'clear' (old)
    'def_style_free': 'wander',  # the defender: 'wander' (moving obstacle round its goal, never plays the
                          #   ball) | 'roam' (runs about its half) | 'block' (stands in the shot's way)
    'wander_x': (0.15, 0.55),  # m in front of the goal line: where it wanders ...
    'wander_y': 0.35,     # ... and this far to either side
    'wander_keep': 0.30,  # m: never closer than this to the ball (backs away)
    'wander_look': 1.0,   # s looking at the ball after each move  # the defender while no clearing is due: 'roam' (runs about its half) | 'block'
    'roam_v': 0.1,       # m/s roaming
    'roam_t': 3.0,        # s at most per roaming leg
    'block_d': 0.31,      # m ball centre -> blocking defender centre (~20 cm off the ball: opens the angle)
    'v_def': 0.07,        # m/s defender top speed (half of the attacker's 0.28)
    'danger_d': 0.10,     # m from the goal line: a ball this close (in front of the goal) is always cleared
    'free_ball_d': 0.30,  # m: the attacker this far from the ball -> a loose ball the keeper clears (in our half)
    'keep_d': 0.40,       # m in front of the goal line: the keeper's place while the ball is in the other half
    'sway_v': 0.08,       # m/s rocking forward / back on the spot ...
    'sway_t': 3.2,        # ... with this period (s): about +-8 cm
    'sway_w': 0.6,        # rad/s nose swaying left / right ...
    'sway_wt': 1.6,       # ... with this period (s): about +-0.15 rad, the ball stays in view
    'weave_ang': 0.25,    # rad the flick turns left / right of the target, alternating each touch (zig-zag)
    'corner_shot': True,  # close to the goal: aim at a corner (angled shot) ...
    'corner_from': 0.90,  # ... from this far (ball -> goal) ...
    'corner_in': 0.06,    # ... this far inside the post
    'tap': True,          # tap dribble: roughly behind the ball -> run into it, flick it toward the target
    'tap_tol': 1.0,       # rad round the ball that counts as 'roughly behind'
    'v_tap': 0.13,        # m/s
    'tap_contact': 0.14,  # m robot centre - ball centre: contact (the bumper is at 0.09 m)
    'k_flick': 2.5,       # at contact: turn gain toward the target ...
    'w_flick': 1.5,       # ... up to this fast (the nose sweeps the ball sideways toward the target)
    'kick_near': 0.50,    # m ball -> target: the run-up kick there (when nearly lined up)
    'kick': True,         # behind the ball: run-up kick (WINDUP -> KICK) instead of a slow push
    'v_windup': 0.08,     # m/s backing off for the run-up ...
    'windup_d': 0.22,     # ... until this far from the ball ...
    'windup_t': 0.8,      # ... or this long
    'kick_t': 1.0,        # s at full speed (v_shot) into the ball
    'v_leg': 0.12,        # m/s along a leg
    'leg_d': 0.45,        # m at most per leg (one side of the square is 0.36 m)
    'leg_t': 2.2,         # s at most per leg
    'leg_turn': 0.25,     # rad: line up this well before driving a leg (a straight leg, no curve)
}


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _clip(v, m):
    return max(-m, min(m, v))


def _turn(err, ball_off=0.0):
    """Turn command; slow while the ball is (nearly) in the camera image, faster when it is well outside."""
    wm = P['w_look'] if abs(ball_off) < P['view'] or abs(err) < 0.5 else P['w_far']
    return _clip(P['k_look'] * err, wm)


def _inside(x, y, half_len, half_w):
    m = P['line_margin']
    return abs(x) < half_len - m and abs(y) < half_w - m


def _clamp_in(x, y, half_len, half_w):
    m = P['line_margin']
    return max(-half_len + m, min(half_len - m, x)), max(-half_w + m, min(half_w - m, y))


def _look(me, ball, age, state, now, back=False):
    """Turn (slowly) until the ball sits in the middle of the image and the camera has just seen it there.
    Returns a command, or None when done."""
    x, y, yaw = me
    err = _wrap(math.atan2(ball[1] - y, ball[0] - x) - yaw)
    d = math.hypot(ball[0] - x, ball[1] - y)
    if abs(err) > P['look_tol'] or age > P['look_fresh']:
        v = -0.06 if (back and d < P['square_h'] - 0.02) else 0.0     # too close: ease back while looking
        return v, _turn(err, err)
    return None


def _step(me, target, state, now, v_max, ball=None, max_d=None, max_t=None):
    """One move toward target: turn to it, then drive at most max_d / max_t (default: a small step, step_d /
    step_t; a square leg passes its own, longer limits and drives straight).  Returns a command, or None when
    the move is over.  The limits are kept with the move, so the following ticks use the same ones."""
    x, y, yaw = me
    st = state.get('step')
    if st is None:
        leg = max_d is not None
        state['step'] = st = {'t0': None, 'x0': x, 'y0': y, 'tgt': target, 'tturn': now, 'v': v_max,
                              'max_d': max_d if leg else P['step_d'], 'max_t': max_t if leg else P['step_t'],
                              'turn_tol': P['leg_turn'] if leg else P['step_turn']}
    # (a move started by the keeper code carries no limits: the defaults then)
    st.setdefault('v', v_max)
    st.setdefault('max_d', P['step_d'])
    st.setdefault('max_t', P['step_t'])
    st.setdefault('turn_tol', P['step_turn'])
    tx, ty = st['tgt']
    if math.hypot(tx - x, ty - y) < 0.02:
        return None
    err = _wrap(math.atan2(ty - y, tx - x) - yaw)
    off = _wrap(math.atan2(ball[1] - y, ball[0] - x) - yaw) if ball is not None else 0.0
    if st['t0'] is None:
        if abs(err) > st['turn_tol'] and now - st['tturn'] < 2.5:
            return 0.0, _turn(err, off)                   # line up with the move first (on the spot), then
                                                          # drive (a short step may curve gently, a leg straight)
        st['t0'], st['x0'], st['y0'] = now, x, y
    moved = math.hypot(x - st['x0'], y - st['y0'])
    if moved > st['max_d'] or now - st['t0'] > st['max_t']:
        return None
    return st['v'], _clip(2.0 * err, P['w_look'])


def _line_slow(me, v, half_len, half_w, goal_w=None):
    """Near the boundary line (within line_slow; goal mouths excepted) at most v_line - at full speed the robot
    could not stop in time and stood up to 4.5 cm over the line (n01)."""
    x, y, _ = me
    m = P['line_slow']
    near = half_w - abs(y) < m or (half_len - abs(x) < m and (goal_w is None or abs(y) > goal_w / 2))
    return max(-P['v_line'], min(P['v_line'], v)) if near else v


def attack_cmd(me, ball, age, target, half_len, half_w, state, now, free=True, give_way=False, near_goal=0.60,
               goal_w=None, behind_tol=None, v_push=None, opp=None):
    v, w, mode = _attack(me, ball, age, target, half_len, half_w, state, now, free, give_way, near_goal,
                         behind_tol, v_push, opp, goal_w)
    return _line_slow(me, v, half_len, half_w, goal_w), w, mode


def _attack(me, ball, age, target, half_len, half_w, state, now, free=True, give_way=False, near_goal=0.60,
            behind_tol=None, v_push=None, opp=None, goal_w=None):
    """me (x, y, yaw), ball (x, y) remembered position, age: s since the camera saw it (0 while it is carried
    at the nose), target (x, y): where the ball should go.  Returns (v, w, mode)."""
    x, y, yaw = me
    bx, by = ball
    tx, ty = target
    if abs(abs(tx) - half_len) < 0.02:
        # a goal on the end line: aim 0.10 m INSIDE the goal - with the goal-line centre as the aim, a ball
        # resting on the goal line in the mouth was pushed along the line instead of in (drill ep 4 / 13)
        if goal_w is not None and P['corner_shot'] and math.hypot(tx - bx, ty - by) < P['corner_from']:
            # close to the goal: an angled shot into a corner (user, 2026-11-26), the one away from the keeper;
            # the side is kept once chosen (no flipping between the corners)
            side = state.get('corner')
            if side is None:
                if opp is not None and abs(opp[0] - tx) < 0.6:
                    side = -1.0 if opp[1] > ty else 1.0
                else:
                    side = 1.0 if by < ty else -1.0       # the far corner from where the ball comes
                state['corner'] = side
            ty = ty + side * (goal_w / 2 - P['corner_in'])
        else:
            state.pop('corner', None)
        tx += math.copysign(0.10, tx)
    ux, uy = tx - bx, ty - by
    n = math.hypot(ux, uy) or 1e-6
    ux, uy = ux / n, uy / n
    d = math.hypot(bx - x, by - y)
    bearing = _wrap(math.atan2(by - y, bx - x) - yaw)
    R = P['orbit_r']

    if give_way:                                          # face-off: the other robot is closer - let it go first
        state.pop('step', None)
        state['phase'] = 'LOOK'
        if d < 0.40:
            return -0.12, _turn(bearing), 'GIVE_WAY'
        return 0.0, _turn(bearing), 'WAIT'

    phi = math.atan2(y - by, x - bx)                      # where we stand round the ball
    phi_t = math.atan2(-uy, -ux)                          # where we want to stand: opposite the target
    if not _inside(bx + R * math.cos(phi_t), by + R * math.sin(phi_t), half_len, half_w):
        # ball near a line: the spot behind it is over the line - nearest spot on the circle that is inside
        for k in range(1, 12):
            a = 0.15 * k
            if _inside(bx + R * math.cos(phi_t + a), by + R * math.sin(phi_t + a), half_len, half_w):
                phi_t += a
                break
            if _inside(bx + R * math.cos(phi_t - a), by + R * math.sin(phi_t - a), half_len, half_w):
                phi_t -= a
                break
    dphi = _wrap(phi_t - phi)
    btol = P['behind_tol'] if behind_tol is None else behind_tol
    vp = P['v_push'] if v_push is None else v_push
    behind = abs(dphi) < btol and d < R + 0.12

    phase = state.get('phase', 'LOOK')
    if age > 2.5 and phase not in ('WINDUP', 'KICK'):
        # not seen for a while (it may have been put back somewhere else): turn round on the spot, fast, to
        # find it again (the striker's search takes over at 3 s; the offline drill has only this)
        state['phase'] = 'LOOK'
        state.pop('step', None)
        if d < 0.18 and age < 6.0:
            # last seen right in front: it is in the camera's blind spot under the nose (a ball resting on the
            # goal line, drill ep 5) - ease back facing it until it comes into view instead of spinning away
            return -0.08, _clip(P['k_look'] * bearing, P['w_look']), 'NEAR_BACK'
        return 0.0, P['w_far'], 'SEARCH_SPIN'
    # kick with a run-up (user, 2026-11-26: back off once, then hit it so it shoots away - '통' 하고):
    # WINDUP - reverse a little with the eyes on the ball, KICK - full speed into the ball for up to kick_t
    if phase == 'WINDUP':
        if d > P['windup_d'] or now - state.get('wt', now) > P['windup_t']:
            state['phase'], state['kt'] = 'KICK', now
            phase = 'KICK'
        else:
            return -P['v_windup'], _clip(P['k_push'] * bearing, P['w_push']), 'WINDUP'
    if phase == 'KICK':
        # only while the ball still lies ahead toward the target: a ball that bounced off to the side and is
        # chased on would be driven anywhere - once into our own goal (drill ep 28)
        ahead = (bx - x) * ux + (by - y) * uy > 0.0        # (0.5 * d was too strict: more balls kicked out)
        if now - state.get('kt', now) < P['kick_t'] and ahead and (abs(bearing) < 0.5 or d < 0.12):
            return P['v_shot'], _clip(P['k_push'] * bearing, P['w_push']), 'KICK'
        state['phase'] = 'LOOK'                           # kicked: look where the ball went, follow it
        return 0.0, _turn(bearing), 'LOOK'
    # tap dribble (user, 2026-11-26: just go for the goal - no measuring of lines, tap it along, lively): roughly
    # behind the ball (within tap_tol round it) and the ball known -> run at the spot just behind it; at the
    # moment of contact turn the nose toward the target, so the ball is flicked a little sideways toward it.
    # Close to the target and nearly lined up: the run-up kick instead.
    # (not with the ball at a line: tapping it on toward the target ran it into the line - the line guard then
    # stopped every tap; the square / run-up kick with the spot behind the ball kept inside the field instead)
    at_line = half_w - abs(by) < 0.10 or (half_len - abs(bx) < 0.10 and abs(by) > 0.15)
    if P['tap'] and at_line and age < 0.8 and phase not in ('STEP', 'WINDUP', 'KICK'):
        # ball at a line (p05: most attacks ended with the ball by a side line): tap it ALONG the line toward the
        # target, from just inside it, and at the contact turn the nose inward - the ball is drawn back into the
        # field.  Only from behind it along the line; otherwise the square below takes us there.
        if half_w - abs(by) < 0.10:                        # side line: along x toward the target
            lx, ly = (1.0 if tx >= bx else -1.0), 0.0
            ix, iy = 0.0, -math.copysign(1.0, by)
        else:                                              # end line beside the goal: along it toward the mouth
            lx, ly = 0.0, -math.copysign(1.0, by)
            ix, iy = -math.copysign(1.0, bx), 0.0
        behind_line = (x - bx) * lx + (y - by) * ly < -0.02
        if behind_line:
            state['phase'] = 'LOOK'
            state.pop('step', None)
            if d > P['tap_contact']:
                ax, ay = bx - lx * 0.05 + ix * 0.03, by - ly * 0.05 + iy * 0.03
                err = _wrap(math.atan2(ay - y, ax - x) - yaw)
                if abs(err) > 0.8:
                    return 0.0, _clip(P['k_look'] * err, P['w_far']), 'LINE_TAP'
                return P['v_tap'] * max(0.3, math.cos(err)), _clip(2.5 * err, P['w_far']), 'LINE_TAP'
            fx, fy = bx + lx * 0.30 + ix * 0.20, by + ly * 0.30 + iy * 0.20     # forward and inward
            to_f = _wrap(math.atan2(fy - y, fx - x) - yaw)
            return P['v_tap'], _clip(P['k_flick'] * to_f, P['w_flick']), 'LINE_FLICK'
    if P['tap'] and not at_line and abs(dphi) < P['tap_tol'] and age < 0.8 and phase not in ('STEP',):
        to_t = _wrap(math.atan2(ty - y, tx - x) - yaw)
        if math.hypot(tx - bx, ty - by) < P['kick_near'] and abs(dphi) < 0.35 and d < 0.30 and free:
            state['phase'], state['wt'] = 'WINDUP', now
            return -P['v_windup'], _clip(P['k_push'] * bearing, P['w_push']), 'WINDUP'
        state['phase'] = 'LOOK'
        state.pop('step', None)
        if d > P['tap_contact']:
            state['last_tap'] = 'run'
            ax, ay = bx - ux * 0.04, by - uy * 0.04          # just behind the ball (target side opposite)
            err = _wrap(math.atan2(ay - y, ax - x) - yaw)
            if abs(err) > 0.8:
                return 0.0, _clip(P['k_look'] * err, P['w_far']), 'TAP'
            return P['v_tap'] * max(0.3, math.cos(err)), _clip(2.5 * err, P['w_far']), 'TAP'
        # contact: keep driving, nose toward the target (flick).  The keeper standing between the ball and the
        # goal: flick toward the open side of the goal instead of straight into it (p02: every tap ran into it)
        if opp is not None and goal_w is not None:
            ox, oy = opp[0] - bx, opp[1] - by
            along = ox * ux + oy * uy                        # keeper in front of the ball toward the goal ...
            across = -ox * uy + oy * ux
            # (only close to the goal - further out a straight tap gets on faster)
            if 0.0 < along < math.hypot(tx - bx, ty - by) and abs(across) < 0.25 and \
                    math.hypot(tx - bx, ty - by) < 0.70:   # ... and roughly in the way
                side = -1.0 if across > 0 else 1.0           # the other side
                gx2, gy2 = tx, ty + side * (goal_w / 2 - 0.05) * (1.0 if abs(ux) > 0.5 else 0.0)
                to_t = _wrap(math.atan2(gy2 - y, gx2 - x) - yaw)
                return P['v_tap'], _clip(P['k_flick'] * to_t, P['w_flick']), 'TAP_DODGE'
        # zig-zag dribble (user, 2026-11-26): each touch flicks the ball a little to the other side of the line
        # to the target - the ball weaves forward, the run-up is still straight
        if state.get('last_tap') != 'contact':
            state['weave'] = -state.get('weave', 1.0)    # a new touch: the other side
        state['last_tap'] = 'contact'
        # (not near our own goal: a ball flicked sideways there went in twice - offline drill)
        far = math.hypot(tx - bx, ty - by) > P['corner_from'] and math.hypot(bx + tx, by) > 0.60
        to_w = to_t + (state.get('weave', 1.0) * P['weave_ang'] if far else 0.0)
        return P['v_tap'], _clip(P['k_flick'] * _wrap(to_w), P['w_flick']), 'TAP_FLICK'
    if phase == 'PUSH':
        # pushing: keep the ball in the middle of the image with small corrections; it drifts off -> stop, look
        if behind or d < 0.14:
            tol = P['push_tol'] if d > 0.15 else P['push_tol'] + 0.15     # at the nose the camera angle jitters
            if abs(bearing) < tol and age < 0.6:
                lined = (bx - x) * ux + (by - y) * uy > 0 and abs(_wrap(math.atan2(uy, ux) - yaw)) < 0.25
                shot = free and lined and math.hypot(tx - bx, ty - by) < near_goal
                v = P['v_shot'] if shot else vp
                return v, _clip(P['k_push'] * bearing, P['w_push']), 'SHOT' if shot else 'PUSH'
        state['phase'] = 'LOOK'
        return 0.0, _turn(bearing), 'LOOK'
    if phase == 'STEP':
        cmd = _step(me, state['step']['tgt'], state, now, P['v_step'], ball)
        if cmd is not None:
            return cmd[0], cmd[1], 'STEP'
        state.pop('step', None)
        state['phase'] = 'LOOK'                           # step done: look at the ball again at once
    # far away with the ball in view: drive at it without stopping - the camera watches it all the way
    # (stopping to look every 8 cm crawled: 30 cm in 20 s, i01); aim at the circle point on our side,
    # already turned toward the back of the ball - that point lies next to the ball, so the ball stays ahead
    if d > R + 0.10 and abs(bearing) < 0.38 and age < 0.4:
        state['phase'] = 'LOOK'
        state.pop('step', None)
        # the furthest point round the circle toward the back of the ball that still keeps the ball within
        # 0.45 rad of the driving direction (camera view): coming in diagonally saves walking round it later
        to_b = math.atan2(by - y, bx - x)
        gx, gy = _clamp_in(bx + R * math.cos(phi), by + R * math.sin(phi), half_len, half_w)
        for k in range(12, -1, -1):
            a = phi + dphi * k / 12.0
            cx, cy = _clamp_in(bx + (R + 0.04) * math.cos(a), by + (R + 0.04) * math.sin(a), half_len, half_w)
            if abs(_wrap(math.atan2(cy - y, cx - x) - to_b)) < 0.30:     # ball stays inside the +-0.45 view
                gx, gy = cx, cy
                break
        err = _wrap(math.atan2(gy - y, gx - x) - yaw)
        return P['v_approach'] * max(0.3, math.cos(err)), _clip(2.0 * err, P['w_look']), 'APPROACH'
    # LOOK: ball in the middle of the image, freshly seen
    cmd = _look(me, ball, age, state, now, back=True)
    if cmd is not None:
        return cmd[0], cmd[1], 'LOOK'
    if behind:
        if P['kick']:
            state['phase'], state['wt'] = 'WINDUP', now
            return -P['v_windup'], _clip(P['k_push'] * bearing, P['w_push']), 'WINDUP'
        state['phase'] = 'PUSH'
        return vp, _clip(P['k_push'] * bearing, P['w_push']), 'PUSH'
    # next small move: far away -> straight toward the circle point on our side (the ball stays ahead);
    # close -> one step round the circle toward the back of the ball (the way that stays inside the field)
    if d > R + 0.10:
        a = phi + _clip(dphi, 0.6)
        gx, gy = _clamp_in(bx + R * math.cos(a), by + R * math.sin(a), half_len, half_w)
        k = min(1.0, max(P['step_d'] * 2.5, d - R) / max(1e-6, math.hypot(gx - x, gy - y)))
        gx, gy = x + (gx - x) * k, y + (gy - y) * k   # far: a longer stretch (the ball stays in view)
    else:
        # close: round the ball on a SQUARE (user, 2026-11-26) instead of a circle - a few straight legs, a look at
        # the ball after each: out to the side edge, to the back corner, to the middle of the back edge (= the
        # pushing spot).  Square axes: f = toward the target (from the pushing spot through the ball), s = across.
        fx, fy = -math.cos(phi_t), -math.sin(phi_t)
        sx_, sy_ = -fy, fx
        a_f = (x - bx) * fx + (y - by) * fy                # robot: along f (> 0: in front of the ball)
        a_s = (x - bx) * sx_ + (y - by) * sy_              # robot: across
        h = P['square_h']

        def corner(af, as_):
            return _clamp_in(bx + af * fx + as_ * sx_, by + af * fy + as_ * sy_, half_len, half_w)

        sg = state.get('dir')
        if sg is None or abs(dphi) < btol:
            sg = 1.0 if a_s >= 0 else -1.0                 # the side we are on already ...
            px_, py_ = bx - h * fx + sg * h * sx_, by - h * fy + sg * h * sy_
            if not _inside(px_, py_, half_len, half_w):
                sg = -sg                                   # ... unless its back corner is over a line
            state['dir'] = sg
        if a_f > -0.8 * h and abs(a_s) < 0.8 * h:
            gx, gy = corner(max(-h, min(h, a_f)), sg * h)  # level with / in front of the ball: out to the side
        elif a_f > -0.8 * h:
            gx, gy = corner(-h, sg * h)                    # on the side edge: to the back corner
        else:
            gx, gy = corner(-h, 0.0)                       # behind: to the pushing spot
        state['phase'] = 'STEP'
        state.pop('step', None)
        cmd = _step(me, (gx, gy), state, now, P['v_leg'], ball, max_d=P['leg_d'], max_t=P['leg_t'])
        if cmd is None:
            state['phase'] = 'LOOK'
            return 0.0, 0.0, 'LOOK'
        return cmd[0], cmd[1], 'STEP'
    state['phase'] = 'STEP'
    state.pop('step', None)
    cmd = _step(me, (gx, gy), state, now, P['v_step'], ball)
    if cmd is None:
        state['phase'] = 'LOOK'
        return 0.0, 0.0, 'LOOK'
    return cmd[0], cmd[1], 'STEP'


def defend_cmd(me, ball, age, own_goal_x, half_len, half_w, goal_w, state, now, clear_dist=0.50, give_way=False,
               opp=None):
    """Keeper (user, 2026-11-26): back to the own goal, side to side on a line just in front of it, eyes on the
    ball (small moves, look again); a ball within clear_dist of the goal -> go at it and clear it toward the
    centre line / out of the field.  Never past the centre line.  Far from the goal (the roles just swapped)
    -> run back at full speed.  Returns (v, w, mode)."""
    if P['def_style_free'] == 'wander':
        # moving obstacle (user, 2026-11-26): never plays the ball, never goes near it or the other robot
        v, w, mode = _wander(me, ball, age, own_goal_x, half_len, half_w, goal_w, state, now)
        vm = P['roam_v'] if mode == 'RUN_BACK' else P['v_def']   # running back: a little faster
        v = max(-vm, min(vm, v))
        return _line_slow(me, _own_half_guard(me, v, own_goal_x), half_len, half_w, goal_w), w, mode
    if P['def_style'] == 'block':
        free_ball = ball is not None and (opp is None or math.hypot(ball[0] - opp[0], ball[1] - opp[1]) >
                                          P['free_ball_d'])
        # (anywhere in our half - only within 0.6 m of the goal it stood and watched a loose ball, user: 'push it
        # out, far')
        near_goal = ball is not None and -math.copysign(1.0, own_goal_x) * ball[0] < 0.05
        # the ball right in front of our goal (within danger_d of the goal line, inside the goal width): clear it
        # no matter who is near it (user, 2026-11-26: the keeper must not just stand there)
        inward_ = -math.copysign(1.0, own_goal_x)
        danger = ball is not None and inward_ * (ball[0] - own_goal_x) < P['danger_d'] + 0.04 and \
            abs(ball[1]) < goal_w / 2 + 0.05
        if danger or (free_ball and near_goal):
            # a loose ball at our goal and the attacker away from it: push it out toward the centre line (user,
            # 2026-11-26: 'blue should have pushed it out here') - at full speed
            inward = -math.copysign(1.0, own_goal_x)
            state['clearing'] = True
            v, w, mode = attack_cmd(me, ball, age, (inward * 0.20, ball[1] * 0.3), half_len, half_w, state, now,
                                    near_goal=9.0, behind_tol=0.6, v_push=P['v_shot'])
            mode = ('CLEAR_DANGER_' if danger else 'CLEAR_') + mode
            v = max(-P['v_def'], min(P['v_def'], v))       # half speed here too (user: the defender is slower)
        elif P['def_style_free'] == 'roam':
            state['clearing'] = False
            v, w, mode = _roam(me, ball, own_goal_x, half_len, half_w, goal_w, state, now)
        else:
            if state.get('clearing'):
                state.pop('step', None)
                state['phase'] = 'LOOK'
            state['clearing'] = False
            v, w, mode = _block(me, ball, age, own_goal_x, half_len, half_w, goal_w, state, now)
            v = max(-P['v_def'], min(P['v_def'], v))       # the defender moves at half speed (user)
    else:
        v, w, mode = _defend(me, ball, age, own_goal_x, half_len, half_w, goal_w, state, now, clear_dist, give_way)
    return _line_slow(me, _own_half_guard(me, v, own_goal_x), half_len, half_w, goal_w), w, mode


def _wander(me, ball, age, own_goal_x, half_len, half_w, goal_w, state, now):
    """Wandering keeper (user, 2026-11-26): around its goal, to random spots in all directions, a look at the ball
    after each move (its camera sees the ball - shared with the other robot), but it never goes for the ball and
    never touches it or the other robot: the attacker dribbles round it like round an obstacle."""
    import random
    x, y, yaw = me
    inward = -math.copysign(1.0, own_goal_x)
    face = _wrap(math.atan2(ball[1] - y, ball[0] - x) - yaw) if ball is not None else 0.0
    d_ball = math.hypot(ball[0] - x, ball[1] - y) if ball is not None else 9.0
    if d_ball < P['wander_keep']:                          # the ball came close: back away from it, eyes on it
        state.pop('wander', None)
        return -P['v_def'], _clip(P['k_look'] * face, P['w_look']), 'WANDER_BACK'
    # away from our goal (we were attacking a moment ago): run straight back to the goal area first, no stops
    # (user, 2026-11-26) - then wander there
    if inward * (x - own_goal_x) > P['wander_x'][1] + 0.15 or abs(y) > P['wander_y'] + 0.25:
        state.pop('wander', None)
        tx, ty = own_goal_x + inward * 0.35, max(-0.2, min(0.2, y))
        if ball is not None:                               # never through the ball: pass it on the side, 0.30 m off
            px, py = tx - x, ty - y
            L = math.hypot(px, py) or 1e-6
            t = ((ball[0] - x) * px + (ball[1] - y) * py) / (L * L)
            if 0.0 < t < 1.0 and math.hypot(ball[0] - (x + px * t), ball[1] - (y + py * t)) < 0.25:
                side = 1.0 if (-(py) * (ball[0] - x) + px * (ball[1] - y)) < 0 else -1.0
                tx, ty = _clamp_in(ball[0] + side * (-py) / L * 0.30, ball[1] + side * px / L * 0.30,
                                   half_len, half_w)
        to = math.atan2(ty - y, tx - x)
        err, sgn = _wrap(to - yaw), 1.0
        err_b = _wrap(to + math.pi - yaw)
        if abs(err_b) < abs(err):
            err, sgn = err_b, -1.0
        if abs(err) > 0.5:
            return 0.0, _clip(P['k_look'] * err, P['w_far']), 'RUN_BACK'
        return sgn * P['roam_v'] * max(0.3, math.cos(err)), _clip(2.0 * err, P['w_far']), 'RUN_BACK'
    wd = state.get('wander')
    if wd is None:
        for _ in range(30):
            tx = own_goal_x + inward * random.uniform(P['wander_x'][0], P['wander_x'][1])
            ty = random.uniform(-P['wander_y'], P['wander_y'])
            if ball is None or math.hypot(tx - ball[0], ty - ball[1]) > P['wander_keep'] + 0.05:
                break
        state['wander'] = wd = {'tgt': (tx, ty), 't0': now, 'look_until': None}
    if wd['look_until'] is not None:                       # arrived: look at the ball a moment
        if now < wd['look_until']:
            return 0.0, (_turn(face, face) if abs(face) > 0.08 else 0.0), 'WANDER_LOOK'
        state.pop('wander', None)
        return 0.0, 0.0, 'WANDER_LOOK'
    tx, ty = wd['tgt']
    dist = math.hypot(tx - x, ty - y)
    if dist < 0.04 or now - wd['t0'] > 4.0:
        wd['look_until'] = now + P['wander_look']
        return 0.0, 0.0, 'WANDER_LOOK'
    to = math.atan2(ty - y, tx - x)
    err, sgn = _wrap(to - yaw), 1.0
    err_b = _wrap(to + math.pi - yaw)
    if abs(err_b) < abs(err):                              # the spot behind us: reverse, eyes stay on the play
        err, sgn = err_b, -1.0
    if abs(err) > 0.5:
        return 0.0, _clip(P['k_look'] * err, P['w_far']), 'WANDER'
    return sgn * P['v_def'] * max(0.3, math.cos(err)), _clip(2.0 * err, P['w_far']), 'WANDER'


def _roam(me, ball, own_goal_x, half_len, half_w, goal_w, state, now):
    """Roaming defender (user, 2026-11-26): instead of guarding the goal mouth it runs about its own half - straight
    lines to random spots, half of them close to the ball - and so knocks the ball away now and then by chance.
    roam_v, a new spot on arrival or after roam_t."""
    import random
    x, y, yaw = me
    inward = -math.copysign(1.0, own_goal_x)
    tgt = state.get('roam')
    if tgt is None or math.hypot(tgt[0] - x, tgt[1] - y) < 0.06 or now - tgt[2] > P['roam_t']:
        for _ in range(20):
            if ball is not None and random.random() < 0.5:
                a = random.uniform(-math.pi, math.pi)
                tx, ty = ball[0] + 0.25 * math.cos(a), ball[1] + 0.25 * math.sin(a)
            else:
                tx = own_goal_x + inward * random.uniform(0.15, half_len - 0.12)
                ty = random.uniform(-half_w + 0.10, half_w - 0.10)
            if inward * (tx - own_goal_x) >= 0.15 and inward * tx < -0.10:      # in our half, out of the mouth
                break
        tx, ty = _clamp_in(tx, ty, half_len, half_w)
        state['roam'] = tgt = (tx, ty, now)
    tx, ty = tgt[0], tgt[1]
    err = _wrap(math.atan2(ty - y, tx - x) - yaw)
    if abs(err) > 0.6:
        return 0.0, _clip(P['k_look'] * err, P['w_far']), 'ROAM'
    return P['roam_v'] * max(0.3, math.cos(err)), _clip(2.0 * err, P['w_far']), 'ROAM'


def _block(me, ball, age, own_goal_x, half_len, half_w, goal_w, state, now):
    """Blocking defender (user, 2026-11-26): it never plays the ball and always gives way at the ball - it only
    stands in the way of the shot.  Ball in our half: on the ball -> goal-centre line, ~10 cm in front of the
    ball (block_d centre to centre), facing it.  Ball in the other half: in front of the goal.  Too close to the
    ball: back off.  On the wrong side of the ball: back to the goal first, round the ball.  (Speed: half,
    applied by defend_cmd.)"""
    x, y, yaw = me
    inward = -math.copysign(1.0, own_goal_x)
    if ball is None:
        bx, by = own_goal_x + inward * 0.8, 0.0
    else:
        bx, by = ball
    face = _wrap(math.atan2(by - y, bx - x) - yaw)
    d_ball = math.hypot(bx - x, by - y)
    # keeper spot in front of the goal (ball far away / in the other half) - further out (user: come out more)
    kx = own_goal_x + inward * P['keep_d']
    dx = max(P['keep_d'], inward * (bx - own_goal_x))
    ky = max(-(goal_w / 2), min(goal_w / 2, by * P['keep_d'] / dx * 1.3))
    in_our_half = ball is not None and inward * bx < 0.05      # (our half: on the goal's side of the centre)
    if in_our_half:
        gx, gy = own_goal_x, 0.0
        ux, uy = gx - bx, gy - by
        n = math.hypot(ux, uy) or 1e-6
        sx, sy = bx + ux / n * P['block_d'], by + uy / n * P['block_d']
        if inward * (sx - own_goal_x) < 0.14:              # never inside the goal mouth
            sx = own_goal_x + inward * 0.14
        sx, sy = _clamp_in(sx, sy, half_len, half_w)
    else:
        sx, sy = kx, ky
    # too close to the ball (the attacker drove it at us): back away from it, eyes on it - never touch it
    if ball is not None and d_ball < P['block_d'] - 0.03:
        return -P['v_def'], _clip(P['k_look'] * face, P['w_look']), 'BLOCK_BACK'
    # wrong side of the ball (further from our goal than the ball): back to the goal first, round the ball
    bg = math.hypot(bx - own_goal_x, by)
    if ball is not None and math.hypot(x - own_goal_x, y) > bg + 0.05:
        tx, ty = kx, ky
        px, py = kx - x, ky - y
        L = math.hypot(px, py) or 1e-6
        t = ((bx - x) * px + (by - y) * py) / (L * L)
        if 0.0 < t < 1.0 and math.hypot(bx - (x + px * t), by - (y + py * t)) < 0.20:
            side = 1.0 if (-(py) * (bx - x) + px * (by - y)) < 0 else -1.0
            tx, ty = _clamp_in(bx + side * (-py) / L * 0.25, by + side * px / L * 0.25, half_len, half_w)
        sx, sy, mode = tx, ty, 'RUN_BACK'
    else:
        mode = 'BLOCK' if in_our_half else 'GO_KEEP'
    dist = math.hypot(sx - x, sy - y)
    if dist < 0.04 or (state.get('keep') and dist < 0.07):
        # on the spot: not standing still (user, 2026-11-26) - eyes on the ball, rocking a few cm forward and
        # back and swaying the nose a little left and right, like a keeper on its toes
        state['keep'] = True
        ph = now % 1000.0
        v = P['sway_v'] * math.sin(2 * math.pi * ph / P['sway_t'])
        w = _turn(face) + P['sway_w'] * math.sin(2 * math.pi * ph / P['sway_wt'])
        return v, _clip(w, P['w_look']), 'KEEP' if not in_our_half else 'BLOCK_HOLD'
    state['keep'] = False
    if ball is not None and age > 0.6 and mode != 'RUN_BACK':
        return 0.0, _turn(face, face), 'LOOK'              # lost sight of it: look first
    to = math.atan2(sy - y, sx - x)
    err, sgn = _wrap(to - yaw), 1.0
    err_b = _wrap(to + math.pi - yaw)
    if abs(err_b) < abs(err):                              # the spot behind us: reverse, eyes on the ball
        err, sgn = err_b, -1.0
    if abs(err) > 0.5:
        return 0.0, _clip(P['k_look'] * err, P['w_far']), mode
    return sgn * P['v_def'] * max(0.3, math.cos(err)), _clip(2.0 * err, P['w_far']), mode


def _own_half_guard(me, v, own_goal_x):
    """The defender stays in its own half: no motion that takes its body (nose 0.09 m / tail 0.075 m) closer
    than 0.05 m to the centre line."""
    x, y, yaw = me
    side = math.copysign(1.0, own_goal_x)                 # own half: side * x > 0
    reach = (0.09 if v > 0 else -0.075) + 0.25 * v
    nx = x + math.cos(yaw) * reach
    if v != 0.0 and side * nx < 0.05 and side * nx < side * x:
        return 0.0
    return v


def _defend(me, ball, age, own_goal_x, half_len, half_w, goal_w, state, now, clear_dist, give_way=False):
    x, y, yaw = me
    inward = -math.copysign(1.0, own_goal_x)
    if ball is None:
        bx, by = own_goal_x + inward * 0.8, 0.0
    else:
        bx, by = ball
    # keeper spot: on a line 0.20 m (centre) in front of the goal line, where the goal centre -> ball line crosses it,
    # within the goal width
    kx = own_goal_x + inward * 0.20               # 0.30 m saved too much (p03: 1 goal in 10 min) - more room
    dx = max(0.20, inward * (bx - own_goal_x))
    ky = max(-(goal_w / 2 - 0.02), min(goal_w / 2 - 0.02, by * 0.20 / dx * 1.3))
    dist = math.hypot(kx - x, ky - y)
    face = _wrap(math.atan2(by - y, bx - x) - yaw)
    # on the wrong side: further from our goal than the ball is (we would push it toward our own goal) ->
    # straight back to the goal first, then turn to the ball (user, 2026-11-26) - also in the middle of a clear
    bg = math.hypot(bx - own_goal_x, by)
    wrong_side = ball is not None and bg > 0.25 and math.hypot(x - own_goal_x, y) > bg + 0.05
    # (a ball right at the goal is cleared from wherever we are - running round it would let it in)
    if wrong_side:
        state['clearing'] = False
    # far from the goal (we were attacking until a moment ago) or on the wrong side: run back at full speed
    if (dist > 0.30 and not state.get('clearing', False)) or (wrong_side and dist > 0.08):
        state.pop('step', None)
        state['phase'] = 'LOOK'
        state['keep'] = False
        tx, ty = kx, ky
        if ball is not None:
            # the ball in the way: go past it on the side, 0.25 m off (never drive through it toward the goal)
            px, py = kx - x, ky - y
            L = math.hypot(px, py) or 1e-6
            t = ((bx - x) * px + (by - y) * py) / (L * L)
            if 0.0 < t < 1.0:
                cx, cy = x + px * t, y + py * t
                off = math.hypot(bx - cx, by - cy)
                if off < 0.18:
                    sx = 1.0 if (-(py) * (bx - x) + px * (by - y)) < 0 else -1.0     # the side away from the ball
                    tx, ty = _clamp_in(bx + sx * (-py) / L * 0.25, by + sx * px / L * 0.25, half_len, half_w)
        to = math.atan2(ty - y, tx - x)
        err, sgn = _wrap(to - yaw), 1.0
        err_b = _wrap(to + math.pi - yaw)
        if abs(err_b) < abs(err):                          # the goal lies behind: reverse, eyes on the play
            err, sgn = err_b, -1.0
        if abs(err) > 0.5:
            return 0.0, _clip(P['k_look'] * err, P['w_far']), 'RUN_BACK'
        return sgn * P['v_shot'], _clip(2.0 * err, P['w_far']), 'RUN_BACK'
    d_goal = math.hypot(bx - own_goal_x, by)
    clearing = state.get('clearing', False)
    if ball is not None and (d_goal < clear_dist or (clearing and d_goal < clear_dist + 0.10)):
        if not clearing:
            state.pop('step', None)
            state['phase'] = 'LOOK'
        state['clearing'] = True
        # clear it far and hard (user, 2026-11-26): straight away from our goal - no exact lining up (any spot
        # within ~0.6 rad of 'behind' will do), then into the ball at full speed
        # toward the middle of the centre line (p05: cleared to the side lines the ball stayed there; in the
        # middle it stays in play and the attack goes on)
        aim = (inward * 0.20, by * 0.3)
        v, w, mode = attack_cmd(me, (bx, by), age, aim, half_len, half_w, state, now, near_goal=9.0,
                                behind_tol=0.6, v_push=P['v_shot'], give_way=give_way)
        return v, w, 'CLEAR_' + mode
    if clearing:
        state.pop('step', None)
        state['phase'] = 'LOOK'
    state['clearing'] = False
    if state.get('phase') == 'STEP':
        st = state.get('step')
        if st is not None:
            cmd = _keeper_step(me, st, now)
            if cmd is not None:
                return cmd[0], cmd[1], 'GO_KEEP'
        state.pop('step', None)
        state['phase'] = 'LOOK'
    keeping = state.get('keep', False)
    if dist < 0.04 or (keeping and dist < 0.08):          # on the spot: watch the ball
        state['keep'] = True
        return 0.0, (_turn(face) if abs(face) > 0.08 else 0.0), 'KEEP'
    state['keep'] = False
    # look at the ball, then a small move along the line (backwards when that needs less turning)
    if ball is not None and (abs(face) > P['look_tol'] or age > 0.6):
        return 0.0, _turn(face, face), 'LOOK'
    state['phase'] = 'STEP'
    state['step'] = {'tgt': (kx, ky), 't0': None, 'tturn': now, 'x0': x, 'y0': y}
    cmd = _keeper_step(me, state['step'], now)
    if cmd is None:
        state['phase'] = 'LOOK'
        return 0.0, 0.0, 'LOOK'
    return cmd[0], cmd[1], 'GO_KEEP'


def _keeper_step(me, st, now):
    """Keeper's small move: forward or backward, whichever needs less turning; 0.15 m / 0.8 s at most."""
    x, y, yaw = me
    tx, ty = st['tgt']
    dist = math.hypot(tx - x, ty - y)
    if dist < 0.03:
        return None
    to = math.atan2(ty - y, tx - x)
    err_f = _wrap(to - yaw)
    err_b = _wrap(to + math.pi - yaw)
    back = abs(err_b) < abs(err_f)
    err = err_b if back else err_f
    if st['t0'] is None:
        if abs(err) > 0.2 and now - st['tturn'] < 2.5:
            return 0.0, _turn(err)
        st['t0'], st['x0'], st['y0'] = now, x, y
    if math.hypot(x - st['x0'], y - st['y0']) > 0.15 or now - st['t0'] > 0.8:
        return None
    v = P['v_step'] * (-1.0 if back else 1.0)
    return v, _clip(2.0 * err, P['w_look'])


def regroup_spot(own_goal_x, ball, half_len, half_w, ready_offset=0.38):
    """Where the new attacker waits after a time-up swap (user, 2026-11-26: the ball stays, both go back to their
    own half first, then play on): its start pose in front of its own goal - or, when the ball lies close to that
    spot, 0.30 m behind the ball on the ball -> opponent goal line (still on its own side of the ball)."""
    inward = -math.copysign(1.0, own_goal_x)
    rx, ry = own_goal_x + inward * ready_offset, 0.0
    if ball is not None and math.hypot(ball[0] - rx, ball[1] - ry) < 0.30:
        gx = -own_goal_x                                    # opponent goal
        ux, uy = gx - ball[0], -ball[1]
        n = math.hypot(ux, uy) or 1e-6
        rx, ry = ball[0] - ux / n * 0.30, ball[1] - uy / n * 0.30
        rx, ry = _clamp_in(rx, ry, half_len - 0.10, half_w - 0.05)
    return rx, ry
