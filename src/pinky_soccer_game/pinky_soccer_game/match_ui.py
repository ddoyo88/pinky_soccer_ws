#!/usr/bin/env python3
"""Tk match window: score, status and the 경기 시작 / 원위치 / 직전 골 취소 buttons (the referee runs the match)."""
import re
import threading

import rclpy
import tkinter as tk
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

class MatchUiNode(Node):
    """Display + buttons only: the referee runs the match (auto goals, restarts, first to `win_score`)."""

    def __init__(self):
        super().__init__('match_ui')
        self.declare_parameter('win_score', 5)
        self.win_score = self.get_parameter('win_score').value
        self.lock = threading.Lock()
        self.phase = 'WAIT'
        self.score = (0, 0)
        self.kickoff = 'YELLOW'
        self.last_event = '경기 시작을 누르면 로봇이 출발 위치로 가고 자동으로 킥오프합니다.'
        self.control_pub = self.create_publisher(String, '/game/control', 10)
        self.create_subscription(String, '/game/score', self._score, 10)
        self.create_subscription(String, '/game/phase', self._phase, 10)
        self.create_subscription(String, '/game/event', self._event, 10)
        self.create_subscription(String, '/game/kickoff', self._kickoff, 10)
        self.goals = []                     # [(mm:ss, team, yellow, blue, kind)]
        self.create_subscription(String, '/game/goals', self._goals, 10)

    def _goals(self, msg):
        rows = [ln.split('|') for ln in msg.data.splitlines() if ln.count('|') == 4]
        with self.lock:
            self.goals = [(t, team, int(y), int(b), kind) for t, team, y, b, kind in rows]

    def _score(self, msg):
        match = re.fullmatch(r'YELLOW\s+(\d+)\s*-\s*(\d+)\s+BLUE', msg.data.strip())
        if match is None:
            self.get_logger().error(f'unrecognized score message: {msg.data!r}')
            return
        with self.lock:
            self.score = (int(match.group(1)), int(match.group(2)))

    def _phase(self, msg):
        with self.lock:
            self.phase = msg.data.strip().upper()

    def _kickoff(self, msg):
        with self.lock:
            self.kickoff = msg.data.strip().upper()

    def _event(self, msg):
        with self.lock:
            self.last_event = msg.data

    def snapshot(self):
        with self.lock:
            return self.phase, self.score, self.last_event, self.kickoff, list(self.goals)

    def send_control(self, command):
        self.control_pub.publish(String(data=command))

class MatchWindow:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title('핑키프로 축구 경기')
        self.root.geometry('600x680')
        self.root.minsize(540, 620)
        self.root.configure(bg='#101820')
        self.root.protocol('WM_DELETE_WINDOW', self.close)

        tk.Label(self.root, text=f'PINKY PRO  ·  SOCCER  ·  {node.win_score}점 먼저 내면 승리',
                 font=('DejaVu Sans', 12, 'bold'), fg='#a8bac8', bg='#101820').pack(pady=(18, 8))
        self.score_label = tk.Label(self.root, text='YELLOW  0  :  0  BLUE',
                                    font=('DejaVu Sans', 30, 'bold'), fg='white', bg='#101820')
        self.score_label.pack(pady=8)
        self.status_label = tk.Label(self.root, text='', font=('DejaVu Sans', 13, 'bold'),
                                     fg='#73d7a1', bg='#101820')
        self.status_label.pack(pady=(4, 14))

        controls = tk.Frame(self.root, bg='#101820')
        controls.pack(pady=4)
        style = dict(font=('DejaVu Sans', 12, 'bold'), width=13, height=2, fg='white', relief='flat')
        self.start_button = tk.Button(controls, text='경기 시작', command=self.new_game,
                                      bg='#16834a', activebackground='#20a860', **style)
        self.start_button.grid(row=0, column=0, padx=6)
        self.reset_button = tk.Button(controls, text='원위치', command=self.reset,
                                      bg='#31465a', activebackground='#45617c', **style)
        self.reset_button.grid(row=0, column=1, padx=6)
        self.undo_button = tk.Button(controls, text='직전 골 취소', command=self.undo,
                                     bg='#8a3b3b', activebackground='#a84c4c', **style)
        self.undo_button.grid(row=0, column=2, padx=6)
        self.stop_button = tk.Button(controls, text='정지', command=self.stop,
                                     bg='#5a1f1f', activebackground='#7a2a2a', **style)
        self.stop_button.grid(row=1, column=1, columnspan=2, pady=(8, 0), sticky='ew', padx=6)
        self.force_button = tk.Button(controls, text='바로 시작', command=self.force_start,
                                      bg='#b8860b', activebackground='#d4a017', **style)
        self.force_button.grid(row=1, column=0, pady=(8, 0), padx=6)
        tk.Label(self.root, text='원위치: 문제가 생기면 로봇을 출발 위치로 돌리고 직전 킥오프부터 다시 (점수 유지)\n'
                                 '직전 골 취소: 잘못 감지된 골을 되돌리고 그 골 전 킥오프부터 다시\n'
                                 '바로 시작: 로봇·공이 어디 있든 지금 바로 경기 진행',
                 font=('DejaVu Sans', 9), fg='#7f93a3', bg='#101820', justify='center').pack(pady=(10, 0))
        self.event_label = tk.Label(self.root, text='', wraplength=540, justify='center',
                                    font=('DejaVu Sans', 10), fg='#d3dce3', bg='#101820')
        self.event_label.pack(side='bottom', fill='x', padx=20, pady=12)

        # goal log, broadcast style:  ⚽ 03:12  노랑  선제골   1 : 0
        tk.Label(self.root, text='득점 기록', font=('DejaVu Sans', 11, 'bold'),
                 fg='#a8bac8', bg='#101820').pack(pady=(14, 4))
        self.goal_list = tk.Text(self.root, height=8, width=46, font=('DejaVu Sans Mono', 12),
                                 bg='#18242f', fg='white', relief='flat', padx=10, pady=6,
                                 state='disabled', cursor='arrow')
        self.goal_list.tag_configure('YELLOW', foreground='#f5cc1f')
        self.goal_list.tag_configure('BLUE', foreground='#6f95ff')
        self.goal_list.tag_configure('score', foreground='white', font=('DejaVu Sans Mono', 12, 'bold'))
        self.goal_list.tag_configure('empty', foreground='#5f7383')
        self.goal_list.pack(padx=20, fill='both', expand=True)
        self.shown_goals = None
        self.refresh()

    def new_game(self):
        self.node.send_control('NEW_GAME')

    def reset(self):
        self.node.send_control('RESET')

    def undo(self):
        self.node.send_control('UNDO_GOAL')

    def stop(self):
        self.node.send_control('STOP')

    def force_start(self):
        self.node.send_control('FORCE_START')

    def refresh(self):
        phase, (yellow, blue), event, kickoff, goals = self.node.snapshot()
        if goals != self.shown_goals:
            self.shown_goals = goals
            self.goal_list.configure(state='normal')
            self.goal_list.delete('1.0', 'end')
            if not goals:
                self.goal_list.insert('end', '아직 골이 없습니다', 'empty')
            for t, team, y, b, kind in goals:
                name = '노랑' if team == 'YELLOW' else '파랑'
                self.goal_list.insert('end', f'⚽ {t}  ')
                self.goal_list.insert('end', f'{name}  {kind:<6}', team)
                self.goal_list.insert('end', f'  {y} : {b}\n', 'score')
            self.goal_list.see('end')
            self.goal_list.configure(state='disabled')
        self.score_label.configure(text=f'YELLOW  {yellow}  :  {blue}  BLUE')
        if phase == 'WAIT':
            status = '경기 시작 전'
        elif phase == 'PREPARE':
            status = f'출발 위치로 이동 중... ({kickoff} 킥오프)'
        elif phase == 'GOAL' and goals:
            t, team, y, b, kind = goals[-1]
            status = f'⚽ GOAL!  {"노랑" if team == "YELLOW" else "파랑"} {kind}  {y} : {b}'
        elif phase == 'GOAL':
            status = '골! 잠시 후 자동으로 다시 시작합니다'
        elif phase == 'CORNER':
            status = '코너! 로봇이 제자리로 돌아갑니다'
        elif phase == 'FINISHED':
            winner = '노랑' if yellow > blue else '파랑'
            status = f'경기 종료 - {winner} 승리!  {yellow} : {blue}'
        elif kickoff in ('YELLOW', 'BLUE'):
            status = f'{kickoff} 킥오프 - {kickoff}가 먼저 공을 칩니다'
        else:
            status = '경기 진행 중'
        self.status_label.configure(text=status, fg='#f5cc1f' if phase == 'FINISHED' else '#73d7a1')
        self.event_label.configure(text=event)
        self.start_button.configure(text='새 경기' if phase == 'FINISHED' else '경기 시작',
                                    state='normal' if phase in ('WAIT', 'FINISHED') else 'disabled')
        self.reset_button.configure(state='normal' if phase in ('PREPARE', 'START', 'GOAL', 'CORNER') else 'disabled')
        self.force_button.configure(state='disabled' if phase == 'START' else 'normal')
        self.stop_button.configure(state='normal' if phase in ('PREPARE', 'START', 'GOAL', 'CORNER') else 'disabled')
        self.undo_button.configure(state='normal' if yellow + blue > 0 and phase != 'WAIT' else 'disabled')
        self.root.after(150, self.refresh)

    def close(self):
        self.node.send_control('STOP')
        self.root.destroy()

    def run(self):
        self.root.mainloop()

def main():
    rclpy.init()
    node = MatchUiNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        MatchWindow(node).run()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
