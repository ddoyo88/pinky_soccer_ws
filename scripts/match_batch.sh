#!/usr/bin/env bash
# Several simulated matches in a row: ./scripts/match_batch.sh <prefix> <first> <last> [win_score 3] [sim_duel args ...]
#   e.g. DUR=1100 ./scripts/match_batch.sh u 1 5 3 headless:=true sensor_view:=false
#   -> ~/pinky_rl/matches/u01 .. u05 (summary.txt each); DUR = seconds of watching per match (match_set.sh)
WS=$(cd "$(dirname "$0")/.." && pwd)
P=${1:?prefix}; A=${2:?first}; B=${3:?last}; WIN=${4:-3}
shift 4 2>/dev/null
for i in $(seq "$A" "$B"); do
  n=$(printf '%s%02d' "$P" "$i")
  "$WS/scripts/match_set.sh" "$n" "$WIN" "$@" > ~/pinky_rl/matches/$n.out 2>&1
  echo "== $n: $(grep -m1 '^goals' ~/pinky_rl/matches/$n/summary.txt)"
done
