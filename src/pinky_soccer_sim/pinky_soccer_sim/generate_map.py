"""CLI: ros2 run pinky_soccer_sim generate_map [--style hardboard|rcj|box] [--length L --width W --goal G] -o soccer_map"""
import argparse

from pinky_soccer_sim.field import FieldSpec, write_map


def main():
    p = argparse.ArgumentParser(description='Rasterise the ideal field into a Nav2 map (pgm + yaml).')
    p.add_argument('--style', choices=('hardboard', 'rcj', 'box'), default='hardboard')
    p.add_argument('--length', type=float, default=None, help='playing length [m]; default: field YAML')
    p.add_argument('--width', type=float, default=None, help='playing width [m]; default: field YAML')
    p.add_argument('--goal', type=float, default=None, help='goal opening [m]; default: field YAML')
    p.add_argument('--goal-depth', type=float, default=0.15, help='box style only: goal rail length [m]')
    p.add_argument('--resolution', type=float, default=0.02)
    p.add_argument('-o', '--output', default='soccer', help='output prefix (no extension)')
    a = p.parse_args()
    spec = FieldSpec(a.length, a.width, a.goal, a.goal_depth, style=a.style)
    print('wrote', write_map(spec, a.output, a.resolution))


if __name__ == '__main__':
    main()
