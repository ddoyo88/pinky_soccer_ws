#!/usr/bin/env bash
# Quick health check while a stage 2..6 simulation is running.  Usage: ./scripts/check_sim.sh
set +e
source /opt/ros/jazzy/setup.bash 2>/dev/null
[ -f "$(dirname "$0")/../install/setup.bash" ] && source "$(dirname "$0")/../install/setup.bash"

echo "== gazebo servers =="
nsrv=$(pgrep -fc "^gz sim .*-s ")
if [ "${nsrv:-0}" -gt 1 ]; then
  echo "  WARNING  $nsrv gz sim servers running -> two worlds publish /clock /scan /tf (robot spins, AMCL drifts)."
  echo "           A Ctrl-C'd launch can leave 'gz sim -s' and component_container behind:"
  pgrep -fa "^gz sim .*-s |component_container" | sed 's/^/             /'
  echo "           stop everything, then:  pkill -f '^gz sim'; pkill -f component_container"
else
  echo "  OK       ${nsrv:-0} server"
fi

topics="$(ros2 topic list 2>/dev/null)"
need="/clock /scan /odom /tf /joint_states /cmd_vel /camera/image_raw /camera/camera_info /gz/dynamic_pose"
echo "== topics =="
for t in $need; do
  if echo "$topics" | grep -qx "$t"; then echo "  OK       $t"; else echo "  MISSING  $t"; fi
done

echo "== rates (3 s each) =="
for t in /scan /odom /camera/image_raw; do
  r=$(timeout 4 ros2 topic hz "$t" --window 20 2>/dev/null | grep -m1 "average rate" | awk '{print $3}')
  echo "  $t : ${r:-no messages} Hz"
done

echo "== TF =="
timeout 4 ros2 run tf2_ros tf2_echo odom base_footprint 2>/dev/null | grep -m1 -A3 "Translation" | sed 's/^/  odom -> base_footprint: /' \
  || echo "  odom -> base_footprint: not available"
timeout 4 ros2 run tf2_ros tf2_echo map base_footprint 2>/dev/null | grep -m1 -A3 "Translation" | sed 's/^/  map  -> base_footprint: /' \
  || echo "  map  -> base_footprint: not available (needs stage 3 SLAM or stage 4+ AMCL)"

echo "== ground truth (ball / robots) =="
timeout 3 ros2 topic echo /gz/dynamic_pose --once 2>/dev/null | grep child_frame_id | sort -u | sed 's/^/  /'

echo "== nav2 lifecycle (stage 4+) =="
for n in /map_server /amcl /controller_server /planner_server /bt_navigator /velocity_smoother; do
  st=$(ros2 lifecycle get "$n" 2>/dev/null | head -1)
  echo "  $n : ${st:-not running}"
done
