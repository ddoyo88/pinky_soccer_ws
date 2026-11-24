#!/usr/bin/env bash
# PPO training of the striker in MuJoCo (no ROS needed while training).
#   ./scripts/rl_train.sh 1 3000000                 stage 1: no opponent   -> ~/pinky_rl/stage1
#   ./scripts/rl_train.sh 2 4000000 ~/pinky_rl/stage1/best    stage 2: scripted opponent, continue from stage 1
# Needs the venv:  python3 -m venv --system-site-packages ~/pinky_rl_venv
#                  ~/pinky_rl_venv/bin/pip install "numpy==1.26.4" torch gymnasium stable-baselines3 tensorboard
set +u
WS=$(cd "$(dirname "$0")/.." && pwd)
source /opt/ros/jazzy/setup.bash && source "$WS/install/setup.bash"
export PYTHONPATH=$WS/src/pinky_soccer_rl:$PYTHONPATH OMP_NUM_THREADS=1
STAGE=${1:-1}; STEPS=${2:-3000000}; INIT=${3:-}
OUT=~/pinky_rl/stage$STAGE; mkdir -p "$OUT"
exec ~/pinky_rl_venv/bin/python -m pinky_soccer_rl.train --stage "$STAGE" --steps "$STEPS" --out "$OUT" \
     ${INIT:+--init "$INIT"} --eval_every 200000
