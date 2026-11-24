#!/usr/bin/env bash
# Watch a trained policy in the MuJoCo viewer.
#   ./scripts/rl_play.sh                          follow ~/pinky_rl/stage1/latest while training (stage 1)
#   ./scripts/rl_play.sh ~/pinky_rl/stage2/best 2
#   ./scripts/rl_play.sh scripted                 the hand-written controller, for comparison
set +u
WS=$(cd "$(dirname "$0")/.." && pwd)
source /opt/ros/jazzy/setup.bash && source "$WS/install/setup.bash"
export PYTHONPATH=$WS/src/pinky_soccer_rl:$PYTHONPATH MUJOCO_GL=glfw
P=${1:-$HOME/pinky_rl/stage1/latest}; STAGE=${2:-1}
if [ "$P" = scripted ]; then
  exec ~/pinky_rl_venv/bin/python -m pinky_soccer_rl.play --scripted --stage "$STAGE"
fi
exec ~/pinky_rl_venv/bin/python -m pinky_soccer_rl.play "$P" --stage "$STAGE" --follow
