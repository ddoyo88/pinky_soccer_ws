#!/usr/bin/env bash
# Write the wheel calibration (wheel_radius / wheel_separation in ~/.config/pinky_soccer/camera_<team>.yaml)
# into the robot's ~/start_robot.sh and restart its drivers.
#   ./scripts/apply_wheel_calib.sh yellow pinky@<yellow robot IP>
#   ./scripts/apply_wheel_calib.sh blue   pinky@<blue robot IP>
set -e
TEAM=${1:?team}; ROBOT=${2:?user@robot}
CFG=~/.config/pinky_soccer/camera_$TEAM.yaml
read R S < <(python3 -c "import yaml; c=yaml.safe_load(open('$CFG')); print(c.get('wheel_radius', ''), c.get('wheel_separation', ''))")
[ -n "$R" ] || { echo "no wheel_radius in $CFG"; exit 1; }
echo "$TEAM: wheel_radius $R  wheel_separation ${S:-unchanged} -> $ROBOT:~/start_robot.sh"
ssh "$ROBOT" "R=$R S=$S bash -s" <<'REMOTE'
set -e
f=~/start_robot.sh
cp "$f" "$f.bak_$(date +%Y%m%d_%H%M%S)"
if grep -q '^WHEEL_RADIUS=' "$f"; then                       # scripts/robot/start_robot.sh layout
  sed -i "s/^WHEEL_RADIUS=.*/WHEEL_RADIUS=\${WHEEL_RADIUS:-$R}/" "$f"
  [ -n "$S" ] && sed -i "s/^WHEEL_SEPARATION=.*/WHEEL_SEPARATION=\${WHEEL_SEPARATION:-$S}/" "$f"
else                                                         # older one-line launch: replace / append the args
  sed -i -E "s/ wheel_radius:=[0-9.]+//; s/ wheel_separation:=[0-9.]+//" "$f"
  sed -i -E "/ros2 launch pinky_soccer_bringup robot.launch.py/ s/\$/ wheel_radius:=$R${S:+ wheel_separation:=$S}/" "$f"
fi
grep -E "wheel_radius|WHEEL_" "$f"
if [ -x ~/stop_soccer.sh ]; then ~/stop_soccer.sh >/dev/null 2>&1 || true
elif [ -x ~/stop_robot.sh ]; then ~/stop_robot.sh >/dev/null 2>&1 || true
else echo "no ~/stop_soccer.sh / ~/stop_robot.sh: stop the running drivers first"; exit 1; fi
setsid nohup "$f" > ~/robot.log 2>&1 < /dev/null &
sleep 12
grep -m2 -E "Wheel radius|Wheel separation" ~/robot.log || echo "(driver log not ready yet: tail ~/robot.log)"
REMOTE
