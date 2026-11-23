"""CLI: ros2 run pinky_soccer_sim generate_world [--style hardboard|rcj|box] [--length L --width W --goal G] -o soccer.sdf"""
import argparse

from pinky_soccer_sim.field import FieldSpec, build_world_sdf


def main():
    p = argparse.ArgumentParser(description='Generate the soccer field Gazebo world (SDF).')
    p.add_argument('--style', choices=('hardboard', 'rcj', 'box'), default='hardboard')
    p.add_argument('--length', type=float, default=None, help='playing length [m]; default: field YAML')
    p.add_argument('--width', type=float, default=None, help='playing width [m]; default: field YAML')
    p.add_argument('--goal', type=float, default=None, help='goal opening [m]; default: field YAML')
    p.add_argument('--goal-depth', type=float, default=0.15, help='box style only: goal rail length [m]')
    p.add_argument('--wall-height', type=float, default=None)
    p.add_argument('--mesh-dir', default='rcj_meshes', help='rcj style: where the ramp STL meshes are written')
    p.add_argument('-o', '--output', default='soccer.sdf')
    a = p.parse_args()
    spec = FieldSpec(a.length, a.width, a.goal, a.goal_depth, a.wall_height, style=a.style)
    with open(a.output, 'w') as f:
        f.write(build_world_sdf(spec, a.mesh_dir))
    print(f'wrote {a.output}')


if __name__ == '__main__':
    main()
