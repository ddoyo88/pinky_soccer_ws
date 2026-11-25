#!/usr/bin/env python3
"""Measure how much the RL policy wiggles (turn-direction flips) and test command post-processing:

    ~/pinky_rl_venv/bin/python scripts/rl_smooth_eval.py [policy.npz] [episodes 40]

Variants (applied to yellow's command, exactly like the striker would):
  raw     - policy output as is
  smooth  - w low-pass filtered (EMA), the policy still sees its raw action
  push    - straight-push dribble assist when the ball sits on the nose and the goal is ahead
  both    - smooth + push
Reports goals, mean time to goal, w sign flips per second, mean |w| while dribbling.
"""
import math
import os
import sys

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, 'src', 'pinky_soccer_rl'))
sys.path.insert(0, os.path.join(WS, 'src', 'pinky_soccer_game'))
from pinky_soccer_game.rl_policy import MLPPolicy, rl_post_process, straight_cmd   # noqa: E402
from pinky_soccer_rl.env import PinkySoccerEnv                        # noqa: E402

pol_path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/pinky_rl/policy_final.npz')
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
pol = MLPPolicy(pol_path)


def run(variant):
    env = PinkySoccerEnv(stage=3)
    env.load_opponent(pol_path)
    goals, t_goal, flips, steps, dribble_w = 0, [], 0, 0, []
    for ep in range(N):
        obs, _ = env.reset(seed=1000 + ep, options={'opponent': 'policy'})
        state = {}
        prev_sign = 0
        for k in range(env.max_steps):
            a = pol(obs)
            v, w = a[0] * pol.v_max, a[1] * pol.w_max
            if variant == 'straight':
                gw = False
                if env.opp_active:
                    me, ob, bl = env._pose('yellow'), env._pose('blue'), env._ball()
                    gw = (math.hypot(bl[0] - me[0], bl[1] - me[1]) < 0.45 and
                          math.hypot(bl[0] - ob[0], bl[1] - ob[1]) < 0.45 and env.priority is False) and os.environ.get("GW", "1") == "1"
                v, w, _ = straight_cmd(env._pose('yellow'), env._ball()[:2], env.goal, env.W / 2, pol.v_max, state,
                                       give_way=gw, free=not env.opp_active or math.hypot(
                                           env._ball()[0] - env._pose('blue')[0],
                                           env._ball()[1] - env._pose('blue')[1]) > 0.5)
            elif variant != 'raw':
                v, w = rl_post_process(v, w, obs, state, pol.v_max,
                                       smooth=variant in ('smooth', 'both'), push=variant in ('push', 'both'))
            act = np.array([v / pol.v_max, w / pol.w_max])
            # the env stores the action it got as 'prev_action' -> the policy input; the striker keeps the raw
            # one, so do the same here: step with the processed command, then put the raw action back
            obs, r, term, trunc, info = env.step(act)
            env.senses['yellow'].prev_action = a
            obs[12], obs[13] = a[0], a[1]
            s = 0 if abs(w) < 0.5 else (1 if w > 0 else -1)
            if s and prev_sign and s != prev_sign:
                flips += 1
            if s:
                prev_sign = s
            steps += 1
            if obs[2] > 0.5 and 0.04 < obs[0] < 0.20 and abs(obs[1]) < 0.05:
                dribble_w.append(abs(w))
            if info.get('goal'):
                goals += 1
                t_goal.append((k + 1) * pol.dt)
            if term or trunc:
                break
    env.close()
    dur = steps * pol.dt
    return (f'{variant:7s} goals {goals:3d}/{N}  time to goal {np.mean(t_goal) if t_goal else float("nan"):5.1f} s  '
            f'turn flips {flips / dur:4.2f}/s  |w| dribbling {np.mean(dribble_w) if dribble_w else 0:4.2f} rad/s')


for v in sys.argv[3].split(',') if len(sys.argv) > 3 else ('raw', 'smooth', 'push', 'both'):
    print(run(v), flush=True)
