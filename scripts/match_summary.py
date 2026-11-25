#!/usr/bin/env python3
"""Summarise one set from match_watch.py / sim_duel logs: score, duration, flow problems."""
import re
import sys
from collections import Counter

out = sys.argv[1]
watch = open(f'{out}/watch.log', errors='replace').read().splitlines()
sim = open(f'{out}/sim.log', errors='replace').read()
goals, events, states, start_t, end_t = [], Counter(), {'yellow': Counter(), 'blue': Counter()}, None, None
for ln in watch:
    m = re.match(r'\s*([\d.]+)\s+(.*)', ln)
    if not m:
        continue
    t, rest = float(m.group(1)), m.group(2)
    if rest.startswith('EVENT'):
        e = rest[6:]
        if '골!' in e:
            goals.append((t, e.split('  ')[1] if '  ' in e else e))
        elif '초 동안 안 움직임' in e:
            events['ball still (deadlock) -> centre'] += 1
        elif '라인에 붙어' in e:
            events['ball on the line -> centre'] += 1
        elif '골대 옆' in e:
            events['ball out beside the goal -> centre'] += 1
        elif '모서리' in e:
            events['ball in a corner -> centre'] += 1
        elif '끼어' in e:
            events['robot stuck in a goal -> lifted'] += 1
        elif '초과' in e:
            events['start-pose timeout'] += 1
        elif '경기 시작 - 가운데 공, 노랑' in e and start_t is None:
            start_t = t
    elif 'PHASE' in rest and 'FINISHED' in rest:
        end_t = t
    sm = re.match(r'(yellow|blue) state \S+ -> (\S+)', rest)
    if sm:
        states[sm.group(1)][sm.group(2)] += 1
print(f'set: {out.rsplit("/", 1)[-1]}')
print(f'duration: {((end_t or t) - (start_t or 0)) / 60:.1f} min (kickoff -> finish)' if start_t else 'duration: ?')
print('goals: ' + (' | '.join(f'{(gt - (start_t or 0)) / 60:.1f}m {g}' for gt, g in goals) or 'none'))
print('flow events: ' + (', '.join(f'{k} x{v}' for k, v in events.items()) or 'none'))
for team in ('yellow', 'blue'):
    c = states[team]
    print(f'{team}: ball search {c["SEARCH"] + c["LOOK_BACK"] + c["ROAM"]} (turn to last seen {c["LOOK_TOWARD"]}, near back-up {c["NEAR_BACKUP"]}), line-escape {c["RL_ESCAPE"]}, '
          f'backoff {c["BACKOFF"] + c["ESCAPE"]}, concede-centre {c["CONCEDE_CENTRE"]}')
print(f"pinch back-offs: {len(re.findall(r'squeezed between', sim))}"); print(f'robot guard stops (log lines): {len(re.findall(r"in the way: stop", sim))}, '
      f'right-of-way changes: {len(re.findall(r"right of way at the ball", sim))}, '
      f'line-guard stops: {len(re.findall(r"boundary line (ahead|behind|\(map\) in the way)", sim))}, '
      f'RL idle -> rules: {len(re.findall(r"RL policy idle", sim))}')
for ln in watch[-8:]:
    if any(k in ln for k in ('AMCL error', 'boundary line', 'robots:')):
        print(ln.split(None, 1)[1])
