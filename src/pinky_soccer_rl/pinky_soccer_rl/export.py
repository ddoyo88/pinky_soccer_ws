"""Export a trained PPO policy to a numpy .npz for the ROS striker (no PyTorch on the robot PC side).

    ~/pinky_rl_venv/bin/python -m pinky_soccer_rl.export ~/pinky_rl/stage1/best ~/pinky_rl/policy_stage1.npz \
        [--opponent]      # policy trained with an opponent (stage >= 2): it reads the lidar opponent input

The file holds the actor MLP (deterministic mean action: Linear-Tanh-Linear-Tanh-Linear), the VecNormalize
observation statistics and a few facts the striker needs (v_max, w_max, dt, observation size).
A self-check compares the numpy forward pass with stable-baselines3's predict() on random observations.
"""
import argparse
import os
import pickle

import numpy as np


def export_policy(model, vn, out, opponent):
    """PPO model + VecNormalize -> .npz (numpy MLP).  Used by main() and by train.py (self-play snapshots)."""
    import torch
    pol = model.policy
    lins = [m for m in pol.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    acts = [type(m).__name__ for m in pol.mlp_extractor.policy_net if not isinstance(m, torch.nn.Linear)]
    assert all(x == 'Tanh' for x in acts), acts
    arrays = {}
    for i, lin in enumerate(lins):
        arrays[f'w{i}'] = lin.weight.detach().numpy().astype(np.float64)
        arrays[f'b{i}'] = lin.bias.detach().numpy().astype(np.float64)
    arrays[f'w{len(lins)}'] = pol.action_net.weight.detach().numpy().astype(np.float64)
    arrays[f'b{len(lins)}'] = pol.action_net.bias.detach().numpy().astype(np.float64)
    np.savez(out, n_layers=len(lins) + 1, obs_mean=vn.obs_rms.mean, obs_var=vn.obs_rms.var,
             clip_obs=vn.clip_obs, epsilon=vn.epsilon, use_opponent=opponent, v_max=0.30, w_max=4.0, dt=0.1,
             obs_size=vn.obs_rms.mean.shape[0], **arrays)
    return len(lins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('policy', help='<dir>/best (without .zip)')
    ap.add_argument('out')
    ap.add_argument('--opponent', action='store_true')
    a = ap.parse_args()
    from stable_baselines3 import PPO
    model = PPO.load(os.path.expanduser(a.policy) + '.zip', device='cpu')
    with open(os.path.expanduser(a.policy) + '_vecnorm.pkl', 'rb') as f:
        vn = pickle.load(f)
    out = os.path.expanduser(a.out)
    n_hidden = export_policy(model, vn, out, a.opponent)
    # self-check against stable-baselines3
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pinky_soccer_game'))
    from pinky_soccer_game.rl_policy import MLPPolicy
    np_pol = MLPPolicy(out)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        obs = rng.normal(0, 1, vn.obs_rms.mean.shape[0]) * np.sqrt(vn.obs_rms.var) + vn.obs_rms.mean
        ref = model.predict(vn.normalize_obs(obs[None]), deterministic=True)[0][0]
        worst = max(worst, float(np.abs(np_pol(obs) - ref).max()))
    print(f'exported {out}: {n_hidden} hidden layers, obs {vn.obs_rms.mean.shape[0]}, '
          f'max |numpy - sb3| = {worst:.2e}')
    assert worst < 1e-4


if __name__ == '__main__':
    main()
