#!/usr/bin/env bash
# Camera calibration window for a REAL robot, with the same DDS settings as real_duel.
#   ./scripts/camera_calib.sh blue      (yellow: ./scripts/camera_calib.sh yellow)
# The robot must be running robot.launch.py.  Result: ~/.config/pinky_soccer/camera_<team>.yaml
set +u
WS=$(cd "$(dirname "$0")/.." && pwd)
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file://$HOME/cyclonedds_pc.xml
exec ros2 run pinky_soccer_vision camera_calib --team "${1:-blue}"
