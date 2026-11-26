#!/usr/bin/env bash
# Several simulated matches in a row: ./scripts/match_batch.sh <prefix> <first> <last> [win_score 3]
#   e.g. ./scripts/match_batch.sh g 1 5   -> ~/pinky_rl/matches/g01 .. g05 (summary.txt each)
WS=$(cd "$(dirname "$0")/.." && pwd)
P=${1:?prefix}; A=${2:?first}; B=${3:?last}; WIN=${4:-3}
for i in $(seq "$A" "$B"); do
  n=$(printf '%s%02d' "$P" "$i")
  "$WS/scripts/match_set.sh" "$n" "$WIN" > ~/pinky_rl/matches/$n.out 2>&1
  echo "== $n: $(grep -m1 '^goals' ~/pinky_rl/matches/$n/summary.txt)"
done
