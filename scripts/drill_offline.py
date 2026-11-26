#!/usr/bin/env python3
"""Offline version of the attack drill: yellow alone, the ball at a random spot, 60 s to score - with the measured
camera (+-0.45 rad view, blind under 0.10 m, 2 cm noise, blur on fast turns, the ball carried at the nose).
Prints every failed episode with what went wrong, so the play can be fixed until all of them succeed.

    ~/pinky_rl_venv/bin/python scripts/drill_offline.py [episodes 30] [seconds 60] [first seed 3000]
"""
import collections
import math
import os
import sys

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, 'src', 'pinky_soccer_rl'))
sys.path.insert(0, os.path.join(WS, 'src', 'pinky_soccer_game'))
from pinky_soccer_game import ball_play as bp                        # noqa: E402
from pinky_soccer_game.rl_policy import ball_filter                  # noqa: E402
from pinky_soccer_rl.env import PinkySoccerEnv                       # noqa: E402


DIR = -1.0 if os.environ.get('DIR') == 'blue' else 1.0     # DIR=blue: attack the -x goal (blue's direction)


def run(n, secs, seed0, verbose=True):
    global TARGET
    env = PinkySoccerEnv(stage=3, episode_s=secs)
    TARGET = (DIR * env.L / 2, 0.0)
    rng = np.random.default_rng(seed0)
    ok, times, fails = 0, [], collections.Counter()
    for ep in range(n):
        env.reset(seed=seed0 + ep, options={'opponent': False})
        st, mem, t = {}, [None, -9.0], 0.0
        modes = collections.Counter()
        last_modes = collections.deque(maxlen=150)          # the last 15 s
        outs, guarded, max_unseen, unseen = 0, 0, 0.0, 0.0
        start_ball = env._ball()[:2].copy()
        start_me = np.array(env._pose('yellow'))
        scored, own = False, False
        line_ref, line_t, respots = None, 0.0, 0
        for k in range(env.max_steps):
            t += env.dt
            me, bl = env._pose('yellow'), env._ball()
            dx, dy = bl[0] - me[0], bl[1] - me[1]
            rel = math.atan2(math.sin(math.atan2(dy, dx) - me[2]), math.cos(math.atan2(dy, dx) - me[2]))
            dist = math.hypot(dx, dy)
            blur = abs(env.senses['yellow'].meas_vw[1]) > 1.5
            if abs(rel) < 0.45 and dist > 0.10 and not blur:
                mem = [(bl[0] + rng.normal(0, 0.02), bl[1] + rng.normal(0, 0.02)), t]
            elif dist < 0.13 and abs(rel) < 0.5:
                mem = [(bl[0], bl[1]), t]                    # carried at the nose
            elif mem[0] is None:
                mem = [(bl[0], bl[1]), t]
            age = t - mem[1]
            unseen = unseen + env.dt if age > 0.5 else 0.0
            max_unseen = max(max_unseen, unseen)
            b = ball_filter(st, mem[0])
            v, w, m = bp.attack_cmd(me, b, age, TARGET, env.L / 2, env.W / 2, st, t, goal_w=env.G)
            modes[m] += 1
            last_modes.append(m)
            _, _, term, trunc, info = env.step(np.array([v / env.v_max, w / env.w_max]))
            outs += int(info.get('out', 0))
            # same as the drill referee (and the match rule): a ball out, or stuck on a line for 5 s (no robot can
            # get behind it without crossing the line) -> dropped again at a random spot
            bl2 = env._ball()
            on_line = env.W / 2 - abs(bl2[1]) < 0.06 or (env.L / 2 - abs(bl2[0]) < 0.06 and abs(bl2[1]) > env.G / 2)
            ref = line_ref
            if not on_line or ref is None or math.hypot(bl2[0] - ref[0], bl2[1] - ref[1]) > 0.02:
                line_ref, line_t = (bl2[0], bl2[1]), t
            if info.get('out') or (on_line and t - line_t > 5.0):
                for _ in range(50):
                    rx = rng.uniform(-env.L / 2 + 0.15, env.L / 2 - 0.15)
                    ry = rng.uniform(-env.W / 2 + 0.12, env.W / 2 - 0.12)
                    if math.hypot(rx - me[0], ry - me[1]) > 0.30:
                        break
                env._place_ball(rx, ry)
                import mujoco
                mujoco.mj_forward(env.model, env.data)
                respots += 1
                line_ref, line_t = None, t
            guarded += int(env.guarded)
            if info.get('goal'):
                scored, own = (True, False) if DIR > 0 else (False, True)
            if info.get('own_goal'):
                scored, own = (True, False) if DIR < 0 else (False, True)
            if term or trunc:
                break
        if scored:
            ok += 1
            times.append(t)
            continue
        bl = env._ball()[:2]
        me = env._pose('yellow')
        L2, W2 = env.L / 2, env.W / 2
        near_line = W2 - abs(bl[1]) < 0.08 or (L2 - abs(bl[0]) < 0.08)
        corner = W2 - abs(bl[1]) < 0.15 and L2 - abs(bl[0]) < 0.15
        why = ('own goal' if own else 'ball in a corner' if corner else 'ball on a line' if near_line else
               'lost the ball' if max_unseen > 8 else 'ball out' if outs else 'too slow / stuck')
        fails[why] += 1
        if verbose:
            lm = collections.Counter(last_modes).most_common(4)
            print(f'  FAIL ep {ep:2d} ({why}): start ball ({start_ball[0]:+.2f},{start_ball[1]:+.2f}) robot '
                  f'({start_me[0]:+.2f},{start_me[1]:+.2f}) -> end ball ({bl[0]:+.2f},{bl[1]:+.2f}) robot '
                  f'({me[0]:+.2f},{me[1]:+.2f}); outs {outs}, line-guard {guarded}, longest unseen {max_unseen:.1f}s; '
                  f'last 15 s: {lm}', flush=True)
    env.close()
    print(f'success {ok}/{n}, mean time {np.mean(times) if times else float("nan"):.1f} s, failures {dict(fails)}',
          flush=True)
    return ok


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    run(n, secs, seed0)
