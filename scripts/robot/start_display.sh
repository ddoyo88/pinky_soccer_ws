#!/bin/bash
# ON THE ROBOT: team LED + LCD pictures + goal sounds (pinky_display.py).   stop: ~/stop_soccer.sh
#   needs ~/pinky_display.py (src/pinky_soccer_vision/robot/) and ~/soccer_media/*.png
TEAM=${TEAM:-yellow}
DOMAIN=$([ "$TEAM" = blue ] && echo 38 || echo 37)
source /opt/ros/jazzy/setup.bash
source "$HOME/pinky_pro/install/local_setup.bash"
export ROS_DOMAIN_ID=$DOMAIN RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
exec python3 "$HOME/pinky_display.py" --ros-args -p team:=$TEAM
