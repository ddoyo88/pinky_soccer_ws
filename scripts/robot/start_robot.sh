#!/bin/bash
# ON THE ROBOT: soccer drivers (wheels/odom, lidar, camera, IMU, TF).   stop: ~/stop_soccer.sh
#   copy to the robot as ~/start_robot.sh, then:  TEAM=blue ~/start_robot.sh   (or edit the defaults below)
# WHEEL_RADIUS: from the calibration window (5. 바퀴 보정) - wheel_radius in camera_<team>.yaml
TEAM=${TEAM:-yellow}
WHEEL_RADIUS=${WHEEL_RADIUS:-0.027}
WHEEL_SEPARATION=${WHEEL_SEPARATION:-0.0961}
DOMAIN=$([ "$TEAM" = blue ] && echo 38 || echo 37)
source /opt/ros/jazzy/setup.bash
source "$HOME/pinky_pro/install/local_setup.bash"
source "$HOME/pinky_soccer_ws/install/local_setup.bash"
export ROS_DOMAIN_ID=$DOMAIN RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
exec ros2 launch pinky_soccer_bringup robot.launch.py imu_enabled:=true adc_enabled:=true \
     wheel_radius:=$WHEEL_RADIUS wheel_separation:=$WHEEL_SEPARATION
