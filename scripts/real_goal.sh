#!/usr/bin/env bash
# REAL ROBOT, one command: the yellow Pinky finds the orange ball and scores into the +X goal on its own.
#
#   1. turn the Pinky on; PC on the robot's Wi-Fi (pinky_XXXX) or on the same Wi-Fi as the robot
#   2. ./scripts/real_goal.sh                         robot's own AP (192.168.4.1)
#      ROBOT=pinky@<robot IP> ./scripts/real_goal.sh   robot on the same Wi-Fi (IP on its LCD)
#      (asks the robot ssh password once)
#   3. put the robot on the yellow start spot when asked, press Enter
#   Ctrl-C stops everything (robot stopped, robot-side nodes this script started are killed).
#
# What it does: checks no simulation is running (its /cmd_vel would drive the robot), reads the robot's
# ROS_DOMAIN_ID, copies + starts the camera publisher (and the official bringup if it is not running yet)
# on the robot, waits for /scan /odom /camera/image_raw, then starts real_yellow.launch.py on the PC.
# Env: ROBOT=pinky@<ip> to use another address, RVIZ=false to skip RViz, MAX_SPEED=0.2 to go slower,
#      CAM_PITCH=6.0 camera pitch down not in the robot model, HFOV=53.5 (OV5647 = camera module v1).
# DDS: the robot's ~/.bashrc decides (ROS_DOMAIN_ID, RMW_IMPLEMENTATION).  With CycloneDDS + multicast off (as on
#      pinky1) the PC gets a Cyclone config with the robot as unicast peer; the robot's ~/cyclonedds.xml must
#      list this PC's IP as a peer too (it is checked and reported).
set -u
ROBOT=${ROBOT:-pinky@192.168.4.1}
HOST=${ROBOT#*@}
WS=$(cd "$(dirname "$0")/.." && pwd)
SOCK=/tmp/pinky_ssh_$$
SSHOPT=(-o ControlMaster=auto -o ControlPath="$SOCK" -o ControlPersist=30m -o StrictHostKeyChecking=accept-new
        -o ConnectTimeout=5)
rsh() { ssh "${SSHOPT[@]}" "$ROBOT" "$@"; }
say() { printf '\n\033[1;33m[real_goal]\033[0m %s\n' "$*"; }
die() { say "ERROR: $*"; exit 1; }

set +u                                     # the ROS setup scripts use unset variables
source /opt/ros/jazzy/setup.bash 2>/dev/null
source "$WS/install/setup.bash" 2>/dev/null || die "build the workspace first (colcon build)"
set -u

# --- 0. safety: no simulation on this PC
if pgrep -f "^gz sim" >/dev/null || pgrep -f "stage[0-9]_.*launch" >/dev/null; then
  die "a simulation is running on this PC - stop it first (its /cmd_vel would drive the real robot)"
fi

# --- 1. robot reachable?
say "robot $HOST ..."
ping -c 1 -W 2 "$HOST" >/dev/null || die "no answer from $HOST - is the PC on the robot's pinky_XXXX Wi-Fi?"
say "ssh login (robot password)"
rsh true || die "ssh to $ROBOT failed"

# --- 2. same ROS_DOMAIN_ID as the robot
ENVOUT=$(rsh "bash -ic 'echo DOMAIN=\${ROS_DOMAIN_ID:-0}; echo RMW=\${RMW_IMPLEMENTATION:-}; cat ~/cyclonedds.xml 2>/dev/null'" 2>/dev/null)
export ROS_DOMAIN_ID=$(sed -n 's/^DOMAIN=//p' <<<"$ENVOUT" | tail -1)
RMW=$(sed -n 's/^RMW=//p' <<<"$ENVOUT" | tail -1)
unset ROS_AUTOMATIC_DISCOVERY_RANGE ROS_LOCALHOST_ONLY ROS_STATIC_PEERS
if [ "$RMW" = rmw_cyclonedds_cpp ]; then
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  IFACE=$(ip route get "$HOST" | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)
  MYIP=$(ip route get "$HOST" | sed -n 's/.* src \([^ ]*\).*/\1/p' | head -1)
  CFG=/tmp/pinky_cyclonedds_pc.xml
  cat > "$CFG" <<XML
<CycloneDDS><Domain>
  <General><Interfaces><NetworkInterface name="$IFACE"/></Interfaces><AllowMulticast>false</AllowMulticast></General>
  <Discovery><Peers><Peer Address="localhost"/><Peer Address="$HOST"/></Peers></Discovery>
</Domain></CycloneDDS>
XML
  export CYCLONEDDS_URI=file://$CFG
  if grep -q "<Peer" <<<"$ENVOUT" && ! grep -q "$MYIP" <<<"$ENVOUT"; then
    say "WARNING: the robot's ~/cyclonedds.xml has no <Peer Address=\"$MYIP\"/> (this PC) - it may not see us"
  fi
fi
say "ROS_DOMAIN_ID=$ROS_DOMAIN_ID ${RMW_IMPLEMENTATION:+RMW=$RMW_IMPLEMENTATION} (taken from the robot)"
ros2 daemon stop >/dev/null 2>&1

# --- 3. camera publisher + bringup on the robot
RS=$WS/install/pinky_soccer_vision/share/pinky_soccer_vision/robot
scp "${SSHOPT[@]}" -q "$RS/pi_camera_publisher.py" "$RS/pinky_effects.py" "$ROBOT:~/" \
    || die "copying the robot scripts failed"
STARTED=$(rsh "bash -ic '
  started=\"\"
  if ! pgrep -f \"[b]ringup_robot.launch\" >/dev/null; then
    setsid nohup ros2 launch pinky_bringup bringup_robot.launch.xml </dev/null >/tmp/soccer_bringup.log 2>&1 &
    started=\"\$started bringup\"
  fi
  if ! pgrep -f \"[p]ython3 .*pi_camera_publisher\" >/dev/null; then
    setsid nohup python3 ~/pi_camera_publisher.py --ros-args -p hfov_deg:=${HFOV:-53.5} </dev/null >/tmp/soccer_camera.log 2>&1 &
    started=\"\$started camera\"
  fi
  if ! ps -eo cmd | grep -q \"[i]nstall/pinky_led/lib\"; then
    setsid nohup ros2 run pinky_led led_server </dev/null >/tmp/soccer_led.log 2>&1 &
    started=\"\$started led\"
  fi
  if ! pgrep -f \"[p]ython3 .*pinky_effects\" >/dev/null; then
    setsid nohup python3 ~/pinky_effects.py </dev/null >/tmp/soccer_effects.log 2>&1 &
    started=\"\$started effects\"
  fi
  echo STARTED=\$started'" 2>/dev/null | sed -n 's/^STARTED=//p' | tail -1)
say "started on the robot:${STARTED:- (already running)}"

cleanup() {
  say "stopping: robot halt + robot-side nodes this script started"
  timeout 3 ros2 topic pub -t 5 -r 10 /cmd_vel geometry_msgs/msg/Twist '{}' >/dev/null 2>&1
  case "$STARTED" in *camera*) rsh "pkill -f '[p]ython3 .*pi_camera_publisher'" ;; esac
  case "$STARTED" in *effects*) rsh "pkill -f '[p]ython3 .*pinky_effects'" ;; esac
  case "$STARTED" in *bringup*) rsh "pkill -INT -f '[b]ringup_robot.launch'" ;; esac
  ssh "${SSHOPT[@]}" -O exit "$ROBOT" >/dev/null 2>&1
}
trap cleanup EXIT

# --- 4. wait for the robot's topics
say "waiting for /scan /odom /camera/image_raw/compressed from the robot ..."
for i in $(seq 1 45); do
  topics=$(timeout 5 ros2 topic list 2>/dev/null)
  if grep -qx /scan <<<"$topics" && grep -qx /odom <<<"$topics" && grep -qx /camera/image_raw/compressed <<<"$topics"; then
    ok=1; break
  fi
  sleep 1
done
if [ -z "${ok:-}" ]; then
  say "robot logs:"; rsh "tail -n 15 /tmp/soccer_bringup.log /tmp/soccer_camera.log" 2>/dev/null
  die "robot topics did not show up (see the logs above)"
fi
say "robot is up.  camera: $(timeout 6 ros2 topic hz /camera/image_raw/compressed 2>/dev/null | sed -n 's/.*average rate: //p' | head -1) Hz"

# --- 5. go
say "Put the robot on the YELLOW START SPOT: 0.38 m in front of the -X goal line, on the centre line,
             facing the +X goal.  Put the orange ball anywhere on the field.  Press Enter to start."
read -r _
ros2 launch pinky_soccer_bringup real_yellow.launch.py rviz:="${RVIZ:-true}" max_speed:="${MAX_SPEED:-0.28}" cam_pitch_deg:="${CAM_PITCH:-6.0}"
