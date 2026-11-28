#!/usr/bin/env python3
"""Run ON THE REAL PINKY: team colour on the LED strip and pictures on the front LCD.

    python3 ~/pinky_display.py --ros-args -p team:=yellow       (or team:=blue)

    normal          LED = team colour, LCD = the team's flag
    own celebration LCD = the player photo, LED blinking + goal melody     (/game/state CELEBRATE_*:
                    the robot's own striker, running on the PC in this robot's ROS domain)
    /game/fx        CORNER -> two beeps, FINISHED -> fanfare (from the referee / domain_relay)

Pictures: ~/soccer_media/<file> (copied from pinky_soccer_sim/media).  Hardware through pinkylib (LED, Buzzer)
and the official LCD driver pinky_emotion.pinky_lcd (240x320 SPI LCD; img_show rotates / scales itself).
Only one process may own the LED / LCD: do not run led_server or emotion_server at the same time.
"""
import os
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

TEAM = {
    'yellow': {'rgb': (255, 190, 0), 'flag': 'portugal.png', 'goal': 'ronaldo.png'},
    'blue': {'rgb': (0, 60, 255), 'flag': 'taegukgi.png', 'goal': 'son.png'},
}
MELODY = {
    'GOAL': [(523, 0.12), (659, 0.12), (784, 0.12), (1047, 0.40)],
    'CORNER': [(440, 0.25), (0, 0.12), (440, 0.25)],
    'FINISHED': [(523, 0.15), (523, 0.15), (523, 0.15), (659, 0.45), (784, 0.15), (659, 0.15), (1047, 0.7)],
}


class PinkyDisplay(Node):
    def __init__(self):
        super().__init__('pinky_display')
        self.declare_parameter('team', 'yellow')
        self.declare_parameter('media_dir', os.path.expanduser('~/soccer_media'))
        self.declare_parameter('brightness', 0.6)
        self.declare_parameter('volume', 50)
        team = self.get_parameter('team').value
        self.cfg = TEAM[team]
        self.lock = threading.Lock()
        self.led_srv = None
        self.led = self._open_led()
        self.lcd = self._open_lcd()
        self.images = {k: self._load(self.cfg[k]) for k in ('flag', 'goal')}
        self.celebrating = False
        self.normal()
        self.create_subscription(String, '/game/state', self._state, 10)
        self.create_subscription(String, '/game/fx', self._fx, 10)
        self.get_logger().info(f'{team}: LED {self.cfg["rgb"]}, LCD {self.cfg["flag"]} / {self.cfg["goal"]}')

    # ---------------------------------------------------------------- hardware
    def _open_led(self):
        try:
            from pinkylib import LED
            led = LED()
            try:
                led.set_brightness(self.get_parameter('brightness').value)
            except Exception:                                          # noqa: BLE001
                pass
            return led
        except Exception as e:                                         # noqa: BLE001
            # the robot driver's led_server (robot.launch.py ensure_led) already owns the strip ("Device or
            # resource busy"): drive it through its /set_led service instead (2026-11-28)
            try:
                from pinky_interfaces.srv import SetLed
                self.led_srv = (self.create_client(SetLed, 'set_led'), SetLed)
                self.get_logger().info(f'LED owned by the driver ({e}); using its set_led service')
            except Exception as e2:                                    # noqa: BLE001
                self.get_logger().warning(f'LED not available: {e} / {e2}')
            return None

    def _open_lcd(self):
        try:
            from pinky_emotion.pinky_lcd import LCD
            return LCD()
        except Exception as e:                                         # noqa: BLE001
            self.get_logger().warning(f'LCD not available: {e}')
            return None

    def _load(self, name):
        from PIL import Image
        path = os.path.join(self.get_parameter('media_dir').value, name)
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:                                         # noqa: BLE001
            self.get_logger().warning(f'picture {path}: {e}')
            return None

    def _led(self, rgb):
        if self.led is None and self.led_srv is not None:
            client, SetLed = self.led_srv
            if client.wait_for_service(timeout_sec=3.0):
                r, g, b = rgb if rgb is not None else (0, 0, 0)
                # 'set_pixel' (unlike 'fill') also calls show() in led_server
                client.call_async(SetLed.Request(command='set_pixel', pixels=list(range(8)), r=r, g=g, b=b))
            return
        if self.led is None:
            return
        with self.lock:
            if rgb is None:
                self.led.clear()
            else:
                self.led.fill(rgb)
            try:
                self.led.show()
            except Exception:                                          # noqa: BLE001
                pass

    def _show(self, key):
        if self.lcd is not None and self.images.get(key) is not None:
            with self.lock:
                self.lcd.img_show(self.images[key])

    def normal(self):
        self._led(self.cfg['rgb'])
        self._show('flag')

    # ---------------------------------------------------------------- events
    def _state(self, msg):
        celebrating = msg.data.startswith('CELEBRATE')
        if celebrating and not self.celebrating:
            self.celebrating = True
            threading.Thread(target=self._celebrate, daemon=True).start()
        elif not celebrating and self.celebrating:
            self.celebrating = False
            self.normal()

    def _celebrate(self):
        self._show('goal')
        blink = threading.Thread(target=self._blink, args=(8,), daemon=True)
        blink.start()
        self._sound(MELODY['GOAL'])
        blink.join()
        if self.celebrating:
            self._led(self.cfg['rgb'])                                 # photo stays until the celebration ends

    def _fx(self, msg):
        kind = msg.data.strip().upper()
        if kind in ('CORNER', 'FINISHED'):
            threading.Thread(target=self._sound, args=(MELODY[kind],), daemon=True).start()

    def _blink(self, n):
        for i in range(n * 2):
            self._led(None if i % 2 == 0 else self.cfg['rgb'])
            time.sleep(0.15)

    def _sound(self, notes):
        try:
            from pinkylib import Buzzer
            bz = Buzzer()
        except Exception as e:                                         # noqa: BLE001  (busy / no GPIO)
            self.get_logger().warning(f'buzzer not available: {e}')
            return
        try:
            bz.buzzer_start(1000)
            for freq, dur in notes:
                if freq > 0:
                    bz.set_buzzer_freq(freq)
                    bz.set_buzzer_duty(self.get_parameter('volume').value)
                else:
                    bz.set_buzzer_duty(0)
                time.sleep(dur)
                bz.set_buzzer_duty(0)
                time.sleep(0.03)
            bz.buzzer_stop()
        finally:
            try:
                bz.close()
            except Exception:                                          # noqa: BLE001
                pass

    def close(self):
        try:
            self._led(None)
            if self.lcd is not None:
                self.lcd.clear()
        except Exception:                                              # noqa: BLE001
            pass


def main():
    rclpy.init()
    node = PinkyDisplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
