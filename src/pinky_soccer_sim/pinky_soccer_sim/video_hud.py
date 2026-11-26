"""Match information drawn into the simulator's own video (mujoco_soccer --record): the score, who attacks and the
attack time left, the latest referee messages and both robots' camera pictures - one file, recorded in simulation
time, so it never stutters like a screen recorder (2026-11-26).

    hud = VideoHud(960, 540)
    bgr = hud.compose(field_rgb, {'yellow': cam_rgb_or_None, 'blue': ...}, info)
info: {'score': (yellow, blue), 'clock': '/game/attack_clock text', 'events': [latest texts], 'phase': str}

Speed: the text layer is drawn (PIL, Korean font) only when what it shows changes - about once a second - and
blended onto every frame with numpy; the cameras are pasted with cv2.  ~5 ms a frame instead of ~57 ms.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_FILES = ['/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf']
YELLOW = (245, 204, 31)
BLUE = (111, 149, 255)
WHITE = (240, 240, 240)
GREY = (170, 180, 190)
NAME = {'YELLOW': '노랑', 'BLUE': '파랑'}


def _font(size):
    for path in FONT_FILES:
        if os.path.exists(path):
            try:
                # Noto Sans CJK collection: face 1 = Korean
                return ImageFont.truetype(path, size, index=1 if path.endswith('.ttc') else 0)
            except OSError:
                continue
    return ImageFont.load_default()


class VideoHud:
    def __init__(self, width, height):
        self.w, self.h = width, height
        s = height / 540.0
        self.f_score = _font(int(30 * s))
        self.f_team = _font(int(20 * s))
        self.f_small = _font(int(15 * s))
        self.bar_h = int(48 * s)
        self.cam_w = int(240 * s)
        self.cam_h = int(136 * s)
        self.pad = int(10 * s)
        self.x0 = self.w - 2 * self.cam_w - 2 * self.pad          # first camera's left edge
        self.ycam = self.h - self.cam_h - self.pad
        self.key, self.layer, self.alpha = None, None, None

    def _texts(self, info):
        """What the overlay shows (the clock in whole seconds): drawn again only when this changes."""
        clk = info.get('clock', '').split()
        clock = ''
        if len(clk) >= 2 and info.get('phase', 'START') == 'START':
            team, left = clk[0], float(clk[1])
            clock = (team, 'none' if left < 0 else 'wait' if len(clk) > 2 else int(max(0.0, left) + 0.999))
        return (tuple(info.get('score', (0, 0))), clock, info.get('phase', ''), tuple(info.get('events', [])[-2:]),
                info.get('time', ''))

    def _draw(self, key):
        score, clock, phase, events, mtime = key
        W, H, p = self.w, self.h, self.pad
        over = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        # --- top bar: score in the middle, attack / time on the right, phase on the left
        d.rectangle([0, 0, W, self.bar_h], fill=(10, 18, 28, 190))
        cx = W // 2
        txt = f'{score[0]} : {score[1]}'
        sw = d.textlength(txt, font=self.f_score)
        d.text((cx - sw / 2, self.bar_h / 2), txt, font=self.f_score, fill=WHITE, anchor='lm')
        d.text((cx - sw / 2 - p, self.bar_h / 2), '노랑', font=self.f_team, fill=YELLOW, anchor='rm')
        d.text((cx + sw / 2 + p, self.bar_h / 2), '파랑', font=self.f_team, fill=BLUE, anchor='lm')
        if clock:
            team, left = clock
            name = NAME.get(team, team)
            text = (f'{name} 공격' if left == 'none' else f'{name} 공격 · 공 터치 대기' if left == 'wait' else
                    f'{name} 공격 · {left}초')
            d.text((W - p, self.bar_h / 2), text, font=self.f_team, fill=YELLOW if team == 'YELLOW' else BLUE,
                   anchor='rm')
        # left: match time, then the phase when it is not play
        left = mtime or ''
        if phase and phase != 'START':
            left += ('   ' if left else '') + {'PREPARE': '준비', 'GOAL': '골!', 'FINISHED': '경기 종료',
                                                'WAIT': '대기'}.get(phase, phase)
        if left:
            d.text((p, self.bar_h / 2), left, font=self.f_team, fill=WHITE if phase == 'START' else GREY, anchor='lm')
        # --- frames for the two cameras (the pictures go in per frame)
        lab = int(22 * H / 540)
        for i, (team, col) in enumerate((('yellow', YELLOW), ('blue', BLUE))):
            x = self.x0 + i * (self.cam_w + p)
            d.rectangle([x - 2, self.ycam - lab - 2, x + self.cam_w + 2, self.ycam + self.cam_h + 2],
                        fill=(10, 18, 28, 170), outline=col, width=2)
            d.text((x + 4, self.ycam - lab / 2), f'{NAME[team.upper()]} 카메라', font=self.f_small, fill=col,
                   anchor='lm')
        # --- bottom left: the latest referee messages
        if events:
            lines = [t if len(t) <= 34 else t[:33] + '…' for t in events]
            lh = int(22 * H / 540)
            top = H - p - lh * len(lines) - 6
            d.rectangle([p - 4, top - 4, self.x0 - 2 * p, H - p], fill=(10, 18, 28, 170))
            for k, t in enumerate(lines):
                d.text((p, top + k * lh), t, font=self.f_small, fill=WHITE if k == len(lines) - 1 else GREY)
        a = np.asarray(over, dtype=np.uint16)
        rows = np.nonzero(a[:, :, 3].any(axis=1))[0]
        # blend only the bands that carry something (top bar, bottom strip): the field in between stays as is
        bands, start = [], None
        for r0, r1 in zip(rows, list(rows[1:]) + [None]):
            start = r0 if start is None else start
            if r1 is None or r1 != r0 + 1:
                bands.append((start, r0 + 1))
                start = None
        # per band: (rows, 255 - alpha as 3-channel uint8, premultiplied colour / 255 as uint8) for cv2's fast
        # saturating arithmetic
        self.bands = []
        for r0, r1 in bands:
            al = a[r0:r1, :, 3:4]
            inv = np.repeat(255 - al, 3, axis=2).astype(np.uint8)
            col = ((a[r0:r1, :, :3] * al + 127) // 255).astype(np.uint8)
            self.bands.append((r0, r1, inv, col))
        self.key = key

    def compose(self, field_rgb, cams, info):
        key = self._texts(info)
        if key != self.key:
            self._draw(key)
        out = field_rgb.copy()
        for r0, r1, inv, col in self.bands:                         # out = out * (1 - alpha) + colour * alpha
            out[r0:r1] = cv2.add(cv2.multiply(out[r0:r1], inv, scale=1.0 / 255), col)
        for i, team in enumerate(('yellow', 'blue')):
            cam = cams.get(team)
            if cam is not None:
                x = self.x0 + i * (self.cam_w + self.pad)
                out[self.ycam:self.ycam + self.cam_h, x:x + self.cam_w] = \
                    cv2.resize(cam, (self.cam_w, self.cam_h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
