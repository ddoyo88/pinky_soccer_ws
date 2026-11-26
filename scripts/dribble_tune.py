#!/usr/bin/env python3
"""Tune straight_cmd's dribble constants in the fast RL environment (no ROS timing):
    ~/pinky_rl_venv/bin/python scripts/dribble_tune.py N 'name:key=val,key=val' ...
Each variant: N episodes alone (opponent off) and N against the frozen policy opponent; goals, time, turn flips."""
import math, os, sys
import numpy as np
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, 'src', 'pinky_soccer_rl'))
sys.path.insert(0, os.path.join(WS, 'src', 'pinky_soccer_game'))
import pinky_soccer_game.rl_policy as rp
from pinky_soccer_rl.env import PinkySoccerEnv

pol_path = os.path.expanduser('~/pinky_rl/policy_final.npz')
pol = rp.MLPPolicy(pol_path)
N = int(sys.argv[1])
BASE = dict(rp.SC)
NOISE = float(os.environ.get('NOISE', 0))
FILT = os.environ.get('FILT') == '1'
CAM = os.environ.get('CAM') == '1'
BLUR = os.environ.get('BLUR') == '1'
rng = np.random.default_rng(0)


def run(opp):
    env = PinkySoccerEnv(stage=3)
    env.load_opponent(pol_path)
    goals, tg, flips, steps = 0, [], 0, 0
    fake = [0.0]
    rp_time = rp.__dict__.get('_t')
    for ep in range(N):
        obs, _ = env.reset(seed=2000 + ep, options={'opponent': 'policy' if opp else False})
        state, prev = {}, 0
        mem = [None, 0.0]
        for k in range(env.max_steps):
            me, bl = env._pose('yellow'), env._ball()
            free = True
            if env.opp_active:
                ob = env._pose('blue')
                free = math.hypot(bl[0] - ob[0], bl[1] - ob[1]) > 0.5
            bm = bl[:2]
            if NOISE:
                bm = (bm[0] + rng.normal(0, NOISE), bm[1] + rng.normal(0, NOISE))
            if CAM:
                # camera: +-0.6 rad field of view, blind closer than 0.12 m -> otherwise the last seen spot
                rel = math.atan2(bl[1] - me[1], bl[0] - me[0]) - me[2]
                rel = math.atan2(math.sin(rel), math.cos(rel))
                blur = BLUR and abs(env.senses['yellow'].meas_vw[1]) > 1.5      # fast turn: blurred image
                if abs(rel) < 0.6 and math.hypot(bl[0] - me[0], bl[1] - me[1]) > 0.12 and not blur:
                    mem[0], mem[1] = bm, _Clock.t
                elif mem[0] is not None:
                    bm = mem[0]
                state['ball_age'] = _Clock.t - mem[1]
            if FILT:
                bm = rp.ball_filter(state, bm)
            v, w, _ = rp.straight_cmd(me, bm, env.goal, env.W / 2, pol.v_max, state, free=free,
                                      half_len=env.L / 2)
            obs, r, term, trunc, info = env.step(np.array([v / pol.v_max, w / pol.w_max]))
            s = 0 if abs(w) < 0.5 else (1 if w > 0 else -1)
            if s and prev and s != prev:
                flips += 1
            if s:
                prev = s
            steps += 1
            if info.get('goal'):
                goals += 1
                tg.append((k + 1) * env.dt)
            if term or trunc:
                break
    env.close()
    return goals, (np.mean(tg) if tg else float('nan')), flips / (steps * env.dt)


# straight_cmd uses wall-clock time for its hysteresis / shot timers: run it on simulated time instead
import time as _time
class _Clock:
    t = 0.0
    def time(self):
        return _Clock.t
_orig = PinkySoccerEnv.step
def _step(self, a):
    _Clock.t += self.dt
    return _orig(self, a)
PinkySoccerEnv.step = _step
rp.clock = _Clock().time

for spec in sys.argv[2:]:
    name, _, kv = spec.partition(':')
    rp.SC.clear(); rp.SC.update(BASE)
    for item in filter(None, kv.split(',')):
        k, v = item.split('=')
        rp.SC[k] = float(v)
    a = run(False); b = run(True)
    print(f'{name:10s} alone {a[0]:3d}/{N} {a[1]:5.1f}s flips {a[2]:.2f}/s | vs opp {b[0]:3d}/{N} {b[1]:5.1f}s '
          f'flips {b[2]:.2f}/s', flush=True)
