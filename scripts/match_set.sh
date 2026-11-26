#!/usr/bin/env bash
# One simulated set: start sim_duel (RL vs RL by default), play to WIN goals, watch it, stop, summarise.
#   ./scripts/match_set.sh <name> [win_score 3] [extra sim_duel args ...]
# Results: ~/pinky_rl/matches/<name>/{sim.log, watch.log, summary.txt}
set +u
WS=$(cd "$(dirname "$0")/.." && pwd)
NAME=${1:?name}; WIN=${2:-3}; shift 2 2>/dev/null
OUT=~/pinky_rl/matches/$NAME; mkdir -p "$OUT"
source /opt/ros/jazzy/setup.bash && source "$WS/install/setup.bash"
POL=${POLICY:-$HOME/pinky_rl/policy_final.npz}
# its own process group (setsid): the whole sim (launch + every node) is stopped together afterwards
# (a plain `kill -INT` does nothing here: background jobs of a script ignore SIGINT)
setsid ros2 launch pinky_soccer_bringup sim_duel.launch.py yellow_domain:=47 blue_domain:=48 win_score:=$WIN auto_ball_reset:=true "$@" \
    > "$OUT/sim.log" 2>&1 &
LP=$!
python3 "$WS/scripts/match_watch.py" 47 48 35 "${DUR:-1200}" > "$OUT/watch.log" 2>&1   # DUR=s of watching
kill -INT -- -$LP 2>/dev/null                       # like Ctrl-C on the whole group
for i in $(seq 1 25); do
  pgrep -g $LP >/dev/null || break; sleep 1
done
kill -KILL -- -$LP 2>/dev/null                      # anything that did not stop
sleep 2
python3 "$WS/scripts/match_summary.py" "$OUT" | tee "$OUT/summary.txt"
