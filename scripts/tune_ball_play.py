#!/usr/bin/env python3
"""Learn the NUMBERS of the rule-based ball play (ball_play.P) - the rules themselves stay as they are
(look - small move - look again, slow turns, push with the ball in the middle of the image).

Cross-entropy method in the MuJoCo RL environment (pinky_soccer_rl.env): yellow plays ball_play.attack_cmd with
a camera model (+-0.45 rad view as measured, blind under 0.10 m, 2 cm noise, blur on fast turns, the ball carried at the nose),
blue is the rule-based keeper (ball_play.defend_cmd) in most episodes.  Score per episode: goal + ball progress
toward the goal - time the ball was out of view for over 1 s - turn reversals (no zig-zag).

    ~/pinky_rl_venv/bin/python scripts/tune_ball_play.py [generations 8] [population 16] [episodes 24] [workers 3]
Result: ~/pinky_rl/ball_play_params.json (best numbers) + the log on stdout.
"""
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, 'src', 'pinky_soccer_rl'))
sys.path.insert(0, os.path.join(WS, 'src', 'pinky_soccer_game'))

# name: (low, high) - bounds keep the user's rules: slow turns near the ball, ~5 cm off the ball when going round
BOUNDS = {
    'w_look': (0.5, 1.0), 'w_far': (1.0, 2.0), 'look_tol': (0.10, 0.30), 'look_fresh': (0.20, 0.50),
    'step_d': (0.06, 0.20), 'step_t': (0.4, 1.0), 'v_step': (0.12, 0.28), 'v_push': (0.15, 0.28),
    'push_tol': (0.15, 0.45), 'k_push': (0.8, 2.5), 'w_push': (0.3, 1.0), 'orbit_r': (0.16, 0.24),
    'orbit_step': (0.3, 1.2), 'behind_tol': (0.15, 0.45), 'step_turn': (0.3, 1.2), 'v_approach': (0.15, 0.28),
}
NAMES = list(BOUNDS)


def evaluate(args):
    params, seeds = args
    from pinky_soccer_game import ball_play as bp
    from pinky_soccer_game.rl_policy import ball_filter
    from pinky_soccer_rl.env import PinkySoccerEnv
    bp.P.update(params)
    env = PinkySoccerEnv(stage=3)
    keeper_state = {}

    def keeper(self):
        x, y, yaw = self._pose('blue')
        b = self._ball()
        v, w, _ = bp.defend_cmd((x, y, yaw), (b[0], b[1]), 0.0, self.L / 2, self.L / 2, self.W / 2, self.G,
                                keeper_state, T[0])
        return v, w
    env._scripted_cmd = keeper.__get__(env)
    rng = np.random.default_rng(seeds[0])
    T = [0.0]
    total, goals, unseen_pen, flips_all, time_all = 0.0, 0, 0.0, 0, 0.0
    for sd in seeds:
        keeper_state.clear()
        opp = 'scripted' if (sd % 10) < 7 else False
        env.reset(seed=int(sd), options={'opponent': opp})
        st, mem, prev_sign, flips = {}, [None, -9.0], 0, 0
        d0 = float(np.linalg.norm(env._ball()[:2] - env.goal))
        scored = False
        for k in range(env.max_steps):
            T[0] += env.dt
            me, bl = env._pose('yellow'), env._ball()
            dx, dy = bl[0] - me[0], bl[1] - me[1]
            rel = math.atan2(math.sin(math.atan2(dy, dx) - me[2]), math.cos(math.atan2(dy, dx) - me[2]))
            dist = math.hypot(dx, dy)
            blur = abs(env.senses['yellow'].meas_vw[1]) > 1.5
            # measured in the sim (2026-11-26): seen at 10 frames/s within +-0.45 rad and beyond 0.10 m, never
            # outside (the earlier +-0.6 rad model was too kind - learned numbers lost the ball in real matches)
            if abs(rel) < 0.45 and dist > 0.10 and not blur:
                mem = [(bl[0] + rng.normal(0, 0.02), bl[1] + rng.normal(0, 0.02)), T[0]]
            elif dist < 0.13 and abs(rel) < 0.5:
                mem = [(bl[0], bl[1]), T[0]]                 # carried at the nose (striker: _carry_ball)
            elif mem[0] is None:
                mem = [(bl[0], bl[1]), T[0]]
            age = T[0] - mem[1]
            if age > 1.0:
                unseen_pen += env.dt
            b = ball_filter(st, mem[0])
            v, w, _ = bp.attack_cmd(me, b, age, env.goal, env.L / 2, env.W / 2, st, T[0], goal_w=env.G)
            s = 0 if abs(w) < 0.4 else (1 if w > 0 else -1)
            if s and prev_sign and s != prev_sign:
                flips += 1
            if s:
                prev_sign = s
            _, _, term, trunc, info = env.step(np.array([v / env.v_max, w / env.w_max]))
            if info.get('goal'):
                scored = True
            if term or trunc:
                break
        dur = (k + 1) * env.dt
        d1 = float(np.linalg.norm(env._ball()[:2] - env.goal))
        progress = 1.0 if scored else max(-0.5, min(1.0, (d0 - d1) / max(0.3, d0)))
        total += (1.0 if scored else 0.0) + 0.3 * progress - 0.2 * flips / dur
        goals += int(scored)
        flips_all += flips
        time_all += dur
    env.close()
    n = len(seeds)
    return total / n - 0.02 * unseen_pen / n, goals, flips_all / time_all, unseen_pen / n


def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    pop = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    neps = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    from pinky_soccer_game import ball_play as bp
    lo = np.array([BOUNDS[k][0] for k in NAMES])
    hi = np.array([BOUNDS[k][1] for k in NAMES])
    mean = np.clip(np.array([bp.P[k] for k in NAMES]), lo, hi)
    std = (hi - lo) * 0.25
    rng = np.random.default_rng(7)
    best = (-1e9, None, None)
    out = os.path.expanduser('~/pinky_rl/ball_play_params.json')
    with Pool(workers) as pool:
        for g in range(gens):
            seeds = [int(s) for s in rng.integers(10000, 99999, size=neps)]     # same episodes for everyone
            cands = [mean] + [np.clip(mean + std * rng.standard_normal(len(NAMES)), lo, hi) for _ in range(pop - 1)]
            res = pool.map(evaluate, [({k: float(c[i]) for i, k in enumerate(NAMES)}, seeds) for c in cands])
            scores = np.array([r[0] for r in res])
            order = np.argsort(-scores)
            elite = np.array([cands[i] for i in order[:max(2, pop // 4)]])
            r0 = res[0]
            rb = res[order[0]]
            print(f'gen {g}: current mean score {r0[0]:+.3f} goals {r0[1]}/{neps} | best {rb[0]:+.3f} goals {rb[1]}/{neps} '
                  f'flips {rb[2]:.2f}/s unseen>1s {rb[3]:.1f}s/ep', flush=True)
            if rb[0] > best[0]:
                best = (rb[0], cands[order[0]], rb)
                with open(out, 'w') as f:
                    json.dump({'score': rb[0], 'goals': rb[1], 'episodes': neps,
                               'params': {k: round(float(best[1][i]), 4) for i, k in enumerate(NAMES)}}, f, indent=1)
            mean = elite.mean(axis=0)
            std = np.maximum(elite.std(axis=0), (hi - lo) * 0.03)
    print('best:', json.dumps({k: round(float(best[1][i]), 4) for i, k in enumerate(NAMES)}), flush=True)


if __name__ == '__main__':
    main()
