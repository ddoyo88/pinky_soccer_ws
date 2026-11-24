"""Watch a trained policy in the MuJoCo viewer (real time, full visuals) and print each episode's outcome.

    ~/pinky_rl_venv/bin/python -m pinky_soccer_rl.play ~/pinky_rl/stage1/best [--stage 1] [--episodes 10]
    --scripted   the simple hand-written controller instead (comparison)
    --follow     keep playing while training runs: reload <policy> whenever the trainer saves a new one
                 (use ~/pinky_rl/stage1/latest - saved at every evaluation)
"""
import argparse
import os
import math
import time

import mujoco
import numpy as np

from pinky_soccer_rl.env import PinkySoccerEnv, _wrap


def scripted(env):
    x, y, yaw = env._pose('yellow')
    b = env._ball()
    u = env.goal - b
    u = u / np.linalg.norm(u)
    to = b - 0.12 * u - np.array([x, y])
    d = float(np.linalg.norm(to))
    tgt = to if d > 0.04 else u
    err = _wrap(math.atan2(tgt[1], tgt[0]) - yaw)
    v = (1.0 if d < 0.06 else min(1.0, 3 * d)) * max(0, math.cos(err)) * (1 if abs(err) < 0.8 else 0)
    return np.array([v, max(-1, min(1, 0.8 * err))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('policy', nargs='?', default='')
    ap.add_argument('--stage', type=int, default=1)
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--scripted', action='store_true')
    ap.add_argument('--speed', type=float, default=1.0, help='playback speed (2 = twice real time)')
    ap.add_argument('--follow', action='store_true')
    ap.add_argument('--fallback', default='', help='--follow: play this policy until <policy> is first saved')
    ap.add_argument('--opponent_policy', default='~/pinky_rl/policy_stage2.npz',
                    help='stage 3: the self-play opponent shown in half of the opponent episodes')
    a = ap.parse_args()
    env = PinkySoccerEnv(stage=a.stage, render_mode='human')
    if a.stage >= 3:
        env.load_opponent(os.path.expanduser(a.opponent_policy))
    import pickle
    model = vn = None
    loaded = 0.0

    label = 'untrained (random actions) - waiting for the first saved policy'

    def load():
        nonlocal model, vn, loaded, label
        from stable_baselines3 import PPO
        src = a.policy
        if not os.path.exists(src + '.zip'):
            if not a.follow:
                raise SystemExit(f'{src}.zip not found')
            if not a.fallback or model is not None:
                return                                        # --follow: keep playing until it exists
            src = os.path.expanduser(a.fallback)
        time.sleep(1.0)                                       # let the trainer finish writing
        model = PPO.load(src + '.zip', device='cpu')
        with open(src + '_vecnorm.pkl', 'rb') as f:
            vn = pickle.load(f)
        loaded = os.path.getmtime(src + '.zip') if src == a.policy else 0.0
        info = src + '.txt'
        label = open(info).read().strip() if (src == a.policy and os.path.exists(info)) else \
            f'{os.path.basename(os.path.dirname(src))}/{os.path.basename(src)} (until the new one is saved)'
        print('policy loaded: ' + label, flush=True)
    if not a.scripted:
        load()
    ep, tally = 0, {'GOAL': 0, 'own goal': 0, 'out': 0, 'time up': 0}
    if a.scripted:
        label = 'hand-written controller'
    while ep < a.episodes or a.follow:
        if a.follow and not a.scripted and os.path.exists(a.policy + '.zip') and \
                os.path.getmtime(a.policy + '.zip') > loaded:
            load()
            tally = {k: 0 for k in tally}                     # count the new policy's episodes
        obs, _ = env.reset(seed=1000 + ep % 50)
        done, total = False, 0.0
        while not done:
            t0 = time.time()
            if model is None and a.scripted:
                action = scripted(env)
            elif model is None:
                action = env.action_space.sample()
            else:
                action = model.predict(vn.normalize_obs(obs[None]), deterministic=True)[0][0]
            obs, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
            if env.viewer is not None:
                if ep == 0 and env.steps == 1:                # top-down view of the whole field
                    env.viewer.cam.lookat[:] = [0, 0, 0]
                    env.viewer.cam.distance, env.viewer.cam.elevation, env.viewer.cam.azimuth = 2.3, -70, 90
                if not env.viewer.is_running():
                    env.close()
                    return
                n = sum(tally.values())
                env.viewer.set_texts([
                    (None, mujoco.mjtGridPos.mjGRID_TOPLEFT, 'policy', label),
                    (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'episode\ntime\ngoals so far',
                     f'{ep + 1}\n{env.steps * env.dt:4.1f} s\n{tally["GOAL"]} / {n}')])
            time.sleep(max(0.0, env.dt / a.speed - (time.time() - t0)))
        what = 'GOAL' if info.get('goal') else 'own goal' if info.get('own_goal') else 'out' if info.get('out') \
            else 'time up'
        tally[what] += 1
        print(f'episode {ep + 1}: {what} after {env.steps * env.dt:.1f} s, reward {total:.1f}', flush=True)
        ep += 1
    env.close()


if __name__ == '__main__':
    main()
