"""Keeping the two robots apart (and out of the goals).

PrepareRouter - route to the ready pose during 경기 준비 (PREPARE) without getting locked against the other robot.

Both robots drive back to their start spots at the same time, usually on crossing lines.  The plain
"straight to the target" controller then pushes the robots into each other until nobody moves.

PrepareRouter adds two things on top of any point-to-point controller:
  * look-ahead: the other robot is close to the straight line -> drive via a side point first
    (head-on: both keep to their own right, so they pass each other instead of mirroring)
  * stuck:      commanded to move but no progress for `stuck_time` -> back off away from the other robot,
                then make a WIDE detour (`wide_offset` to the side of it) before heading home again
The caller drives to whatever `update()` returns.

avoid_opponent / detour_target - the same idea during play (slow down / steer away / pass beside).
goal_mouth_guard - never drive the nose into a goal: the crossbar is only 7 cm high, a robot that follows the
                   ball in gets jammed under it and cannot back out (and AMCL loses it in there).
"""
import math


class PrepareRouter:
    def __init__(self, half_x, half_y, clearance=0.26, offset=0.34, wide_offset=0.48,
                 stuck_time=1.2, backoff_time=0.6, via_tol=0.08):
        self.half_x, self.half_y = half_x, half_y      # where the robot centre may go (field + outer strip)
        self.clearance, self.offset, self.wide_offset = clearance, offset, wide_offset
        self.stuck_time, self.backoff_time, self.via_tol = stuck_time, backoff_time, via_tol
        self.reset()

    def reset(self):
        self.via = None            # (x, y) side point currently driven to
        self.wide = False          # current via point comes from a stuck event
        self.progress = None       # (x, y, yaw, t) last pose with progress
        self.backoff_until = 0.0
        self.backoff_dir = -1.0

    # ---------------------------------------------------------------- geometry
    def _clamp(self, x, y):
        return max(-self.half_x, min(self.half_x, x)), max(-self.half_y, min(self.half_y, y))

    def _side_point(self, me, target, opp, offset):
        """Point `offset` beside the opponent, on the side away from it (keep right when it is dead ahead)."""
        dx, dy = target[0] - me[0], target[1] - me[1]
        d = math.hypot(dx, dy) or 1e-6
        ux, uy = dx / d, dy / d
        nx, ny = -uy, ux                                   # left of the travel direction
        lat = (opp[0] - me[0]) * nx + (opp[1] - me[1]) * ny
        side = -1.0 if abs(lat) < 0.06 else (-1.0 if lat > 0 else 1.0)
        best = None
        for s in (side, -side):
            px, py = self._clamp(opp[0] + s * nx * offset, opp[1] + s * ny * offset)
            gap = math.hypot(px - opp[0], py - opp[1])
            if gap >= 0.7 * offset:                        # not squeezed against a wall
                return px, py
            if best is None or gap > best[0]:
                best = (gap, (px, py))
        return best[1]

    def _blocking(self, me, target, opp):
        """Opponent close to the straight segment me -> target (and not sitting on the target itself)."""
        dx, dy = target[0] - me[0], target[1] - me[1]
        d = math.hypot(dx, dy)
        if d < 0.10 or math.hypot(opp[0] - target[0], opp[1] - target[1]) < 0.20:
            return False
        t = ((opp[0] - me[0]) * dx + (opp[1] - me[1]) * dy) / (d * d)
        if t <= 0.0 or t >= 1.0:
            return math.hypot(opp[0] - me[0], opp[1] - me[1]) < 0.18 and t > 0.0
        px, py = me[0] + t * dx, me[1] + t * dy
        return math.hypot(opp[0] - px, opp[1] - py) < self.clearance

    # ---------------------------------------------------------------- main
    def update(self, me, target, opp, now, moving_wanted=True):
        """me = (x, y, yaw), target = (x, y), opp = (x, y) or None.
        Returns ('backoff', v) -> drive straight with linear speed v (no turning), or
                ('goto', (x, y), is_via)."""
        if now < self.backoff_until:
            return 'backoff', 0.22 * self.backoff_dir

        # stuck detection: no 2 cm / 0.1 rad progress for stuck_time while we want to move
        ref = self.progress
        if ref is None or not moving_wanted or math.hypot(me[0] - ref[0], me[1] - ref[1]) > 0.02 \
                or abs(_wrap(me[2] - ref[2])) > 0.10:
            self.progress = (me[0], me[1], me[2], now)
        elif now - ref[3] > self.stuck_time:
            self.progress = None
            self.backoff_until = now + self.backoff_time
            if opp is not None:
                ahead = (opp[0] - me[0]) * math.cos(me[2]) + (opp[1] - me[1]) * math.sin(me[2])
                self.backoff_dir = -1.0 if ahead >= 0 else 1.0     # move away from the other robot
                self.via, self.wide = self._side_point(me, target, opp, self.wide_offset), True
            else:                                                  # stuck on a wall: just back off
                self.backoff_dir = -1.0
            return 'backoff', 0.22 * self.backoff_dir

        if self.via is not None and math.hypot(me[0] - self.via[0], me[1] - self.via[1]) < self.via_tol:
            self.via, self.wide = None, False
        if self.via is None and opp is not None and self._blocking(me, target, opp):
            self.via, self.wide = self._side_point(me, target, opp, self.offset), False
        if self.via is not None:
            return 'goto', self.via, True
        return 'goto', (target[0], target[1]), False


def avoid_opponent(me, v, w, opp, ball=None, stop_dist=0.18, slow_dist=0.42, corridor=0.17, w_max=3.0,
                   duel_speed=0.12):
    """Collision filter for any (v, w) command during play.
    The other robot in the corridor we are driving into -> slow down (full stop at stop_dist) and steer
    away from it.  Turning in place and driving away from it are never limited.
    Exception: we are playing the ball (ball right at our nose, nearer than the opponent) -> keep pushing;
    the ball stays between the two robots, so the bodies do not touch and the duel still happens.
    me = (x, y, yaw), opp / ball = (x, y) or None.  Returns (v, w, active)."""
    if opp is None or abs(v) < 1e-3:
        return v, w, False
    c, s = math.cos(me[2]), math.sin(me[2])
    dx, dy = opp[0] - me[0], opp[1] - me[1]
    fx, fy = c * dx + s * dy, -s * dx + c * dy          # opponent in the robot frame
    if ball is not None and v > 0:
        bdx, bdy = ball[0] - me[0], ball[1] - me[1]
        bfx, bfy = c * bdx + s * bdy, -s * bdx + c * bdy
        # ...and only when the opponent is straight beyond the ball (ball between the bodies)
        if 0.0 < bfx < 0.16 and abs(bfy) < 0.06 and bfx < fx and abs(fy - bfy) < 0.07:
            # duel through the ball, but gently: no full-speed shot into the other robot
            if math.hypot(fx, fy) < slow_dist:
                return min(v, duel_speed), w, True
            return v, w, False
    ahead = fx if v > 0 else -fx
    dist = math.hypot(fx, fy)
    if ahead <= 0.0 or abs(fy) > corridor or dist > slow_dist:
        return v, w, False
    scale = max(0.0, min(1.0, (dist - stop_dist) / (slow_dist - stop_dist)))
    # steer so the travel direction leaves the opponent's side (forward: turn away; reverse: the opposite)
    push = (1.0 - scale) * 2.5 * (-1.0 if fy > 0 else 1.0)
    if v < 0:
        push = -push
    w = max(-w_max, min(w_max, w + push))
    return v * scale, w, True


def detour_target(me, target, opp, router):
    """Play-time version of the look-ahead: the other robot near the straight line -> side point instead."""
    if opp is None or math.hypot(opp[0] - target[0], opp[1] - target[1]) < 0.28 \
            or not router._blocking(me, target, opp):      # opponent AT the ball: contest it, no detour
        return target, False
    return router._side_point(me, target, opp, router.offset), True


def goal_mouth_guard(me, v, half_len, goal_w, nose=0.09, margin=0.06):
    """Cut forward/backward speed that would push the robot's leading end past a goal line inside the mouth."""
    if abs(v) < 1e-3:
        return v
    d = nose if v > 0 else -0.08                    # leading end: bumper nose, or the rear when reversing
    px, py = me[0] + d * math.cos(me[2]), me[1] + d * math.sin(me[2])
    outward = math.copysign(1.0, px) * math.cos(me[2]) * math.copysign(1.0, v) > 0
    if outward and abs(py) < goal_w / 2 + margin and abs(px) > half_len - 0.005:
        return 0.0
    return v


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))
