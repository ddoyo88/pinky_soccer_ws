#!/usr/bin/env bash
# One-shot setup: apt dependencies + official pinky_pro repo (pinned) + build.
# Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic (ros_gz).  Run from anywhere:  ./scripts/setup_workspace.sh
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINKY_URL="https://github.com/pinklab-art/pinky_pro.git"
# commit this workspace was written and checked against ("refactor(pinky_description): move wheel joint origin to wheel center")
PINKY_COMMIT="b1600c30b49b572fefed7dcad57527948da88b45"

if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  echo "ERROR: ROS 2 Jazzy not found at /opt/ros/jazzy (Ubuntu 24.04 required)." >&2
  exit 1
fi

echo "[1/4] apt dependencies"
sudo apt update
sudo apt install -y git python3-colcon-common-extensions python3-rosdep python3-opencv python3-numpy python3-yaml \
  ros-jazzy-ros-gz ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-slam-toolbox \
  ros-jazzy-xacro ros-jazzy-robot-state-publisher ros-jazzy-joint-state-publisher \
  ros-jazzy-joint-state-publisher-gui ros-jazzy-rviz2 ros-jazzy-teleop-twist-keyboard \
  ros-jazzy-rqt-image-view

echo "[2/4] official Pinky Pro packages (pinklab-art/pinky_pro @ ${PINKY_COMMIT:0:7})"
if [ ! -d "$WS/src/pinky_pro/.git" ]; then
  git clone "$PINKY_URL" "$WS/src/pinky_pro"
fi
git -C "$WS/src/pinky_pro" fetch --quiet origin || true
git -C "$WS/src/pinky_pro" checkout --quiet "$PINKY_COMMIT" || {
  echo "WARN: could not check out ${PINKY_COMMIT:0:7}; using the current HEAD of pinky_pro (may differ from what this workspace expects)." >&2
}

echo "[3/4] rosdep (only the packages we build)"
set +u; source /opt/ros/jazzy/setup.bash; set -u
sudo rosdep init 2>/dev/null || true
rosdep update --rosdistro jazzy || true
rosdep install --rosdistro jazzy --ignore-src -r -y \
  --from-paths "$WS"/src/pinky_soccer_* "$WS/src/pinky_pro/pinky_description" "$WS/src/pinky_pro/pinky_navigation" || true

echo "[4/4] colcon build"
cd "$WS"
colcon build --symlink-install --packages-up-to pinky_soccer_bringup

cat <<EOF

Done.  In every new terminal:
  source /opt/ros/jazzy/setup.bash
  source $WS/install/setup.bash
First run:
  ros2 launch pinky_soccer_bringup stage1_robot.launch.py
EOF
