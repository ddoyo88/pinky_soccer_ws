#!/usr/bin/env python3
"""Run ON THE REAL PINKY: team LED colour + goal / corner effects (LED blinking, buzzer).

    ros2 run pinky_led led_server &            # official LED service (/set_led), owns the WS2812 strip
    python3 ~/pinky_effects.py                 # this node (team:=yellow by default)

Listens to /game/fx (String) from the PC's real referee:
    GOAL      LED blinks in the team colour + rising goal melody
    CORNER    two low beeps (the robot then drives back to its start spot)
    FINISHED  long blinking + fanfare
At start (and after every effect) the LED is filled with the team colour.
The buzzer GPIO is opened only while a sound plays (like ~/ap/battery_buzzer.py), never kept.
"""
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

TEAM_RGB = {'yellow': (255, 190, 0), 'blue': (0, 60, 255)}
# (frequency Hz, seconds); 0 Hz = rest
MELODY = {
    'GOAL': [(523, 0.12), (659, 0.12), (784, 0.12), (1047, 0.40)],
    'CORNER': [(440, 0.25), (0, 0.12), (440, 0.25)],
    'FINISHED': [(523, 0.15), (523, 0.15), (523, 0.15), (659, 0.45), (784, 0.15), (659, 0.15), (1047, 0.7)],
}
BLINKS = {'GOAL': 8, 'CORNER': 0, 'FINISHED': 16}


class PinkyEffects(Node):
    def __init__(self):
        super().__init__('pinky_effects')
        self.declare_parameter('team', 'yellow')
        self.declare_parameter('volume', 50)                # buzzer duty cycle %
        self.rgb = TEAM_RGB.get(self.get_parameter('team').value, TEAM_RGB['yellow'])
        from pinky_interfaces.srv import SetLed
        self.SetLed = SetLed
        self.led = self.create_client(SetLed, 'set_led')
        self.busy = threading.Lock()
        self.create_subscription(String, '/game/fx', self._fx, 10)
        self.create_timer(1.0, self._initial_colour)
        self.coloured = False
        self.get_logger().info(f'effects ready, team colour {self.rgb}')

    def _set(self, rgb):
        if not self.led.service_is_ready():
            return
        req = self.SetLed.Request()
        if rgb is None:
            req.command = 'clear'
        else:
            req.command, (req.r, req.g, req.b) = 'fill', rgb
        self.led.call_async(req)

    def _initial_colour(self):
        if not self.coloured and self.led.service_is_ready():
            self._set(self.rgb)
            self.coloured = True
            self.get_logger().info('LED on in the team colour')

    def _fx(self, msg):
        kind = msg.data.strip().upper()
        if kind not in MELODY or not self.busy.acquire(blocking=False):
            return
        threading.Thread(target=self._play, args=(kind,), daemon=True).start()

    def _play(self, kind):
        try:
            self.get_logger().info(f'effect {kind}')
            blink = threading.Thread(target=self._blink, args=(BLINKS[kind],), daemon=True)
            blink.start()
            self._sound(MELODY[kind])
            blink.join()
            self._set(self.rgb)
        finally:
            self.busy.release()

    def _blink(self, n):
        for i in range(n * 2):
            self._set(None if i % 2 == 0 else self.rgb)
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


def main():
    rclpy.init()
    node = PinkyEffects()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
