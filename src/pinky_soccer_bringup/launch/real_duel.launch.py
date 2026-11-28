"""REAL ROBOTS, 1:1 - two Pinky Pro on one PC, each in its own ROS domain (the sim_duel layout on hardware).

On each robot (its ~/.bashrc sets ROS_DOMAIN_ID 37 / 38, CycloneDDS, ~/cyclonedds.xml with this PC as peer):
    ros2 launch pinky_soccer_bringup robot.launch.py imu_enabled:=true adc_enabled:=true
On the PC (both robots just in front of their own penalty line, rear on the line, facing the other goal;
ball on the centre spot):
    ros2 launch pinky_soccer_bringup real_duel.launch.py

Per robot, in its domain: map + AMCL + Nav2, ball detector (JPEG camera, intrinsics from the FOV if the robot
publishes no calibration), striker (yellow attacks +X, blue attacks -X).  The robots publish plain names
(/scan, /odom, /cmd_vel, /tf ...) - separate domains keep them apart, so no namespaces are needed here.
Shared, in the yellow domain: real referee (goals / corners from the robots' camera ball estimates, both robots
READY + ball still on the centre spot -> start), match window, sensor window; domain_relay copies /game/* to
the blue domain and blue's /game/state + /ball/position_map back as /blue/...

PC DDS: CycloneDDS with ~/cyclonedds_pc.xml (multicast off, both robot IPs as peers) - see cyclonedds_pc arg.
Nothing moves before 경기 시작 / 바로 시작 in the match window; 정지 stops both robots.
"""
import math
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pinky_soccer_bringup.common import field_declarations, field_forward, include, solo_striker_params
from pinky_soccer_sim.field import field_config, resolve_dims


def launch_setup(context):
    v = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    L, W, G, strip = resolve_dims(v('field_style'), v('field_length'), v('field_width'), v('goal_width'))
    vmax = float(v('max_speed'))
    # start pose: the whole robot just in front of the white penalty line (its rear on the line):
    # robot centre = penalty_depth + 0.075 m (base_footprint -> rear) from its own goal line
    start = float(field_config(v('field_style')).get('penalty_depth', 0.17)) + 0.075
    uri = os.path.expanduser(v('cyclonedds_pc'))
    if not os.path.isfile(uri):
        raise RuntimeError(f'{uri} not found - PC CycloneDDS config with both robot IPs as peers')
    dds = [SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'),
           SetEnvironmentVariable('CYCLONEDDS_URI', 'file://' + uri)]
    yd, bd = int(v('yellow_domain')), int(v('blue_domain'))
    win = int(v('win_score'))

    def cam_cfg(team):
        """Camera calibration window result (~/.config/pinky_soccer/camera_<team>.yaml) or the launch args."""
        import yaml
        cfg = {'cam_pitch_deg': float(v(f'cam_pitch_{team}'))}
        try:
            with open(os.path.expanduser(f'~/.config/pinky_soccer/camera_{team}.yaml')) as f:
                saved = yaml.safe_load(f) or {}
            cfg.update({k: (float(val) if k == 'cam_pitch_deg' else int(val)) for k, val in saved.items()
                        if k in ('cam_pitch_deg', 'hsv_h_min', 'hsv_h_max', 'hsv_s_min', 'hsv_v_min')})
            for k in ('intrinsics', 'distortion'):                     # checkerboard lens calibration
                if saved.get(k):
                    cfg[k] = [float(x) for x in saved[k]]
            print(f'[real_duel] {team} camera calibration from ~/.config/pinky_soccer/camera_{team}.yaml: {cfg}')
        except FileNotFoundError:
            pass
        return cfg

    def robot(team, domain):
        sign = 1.0 if team == 'yellow' else -1.0                   # yellow attacks +X, blue -X
        x0, yaw0 = -sign * (L / 2 - start), (0.0 if team == 'yellow' else math.pi)
        striker = solo_striker_params(L, W, G, strip, vmax, match=True, sim_time=False)
        striker['policy_path'] = v(f'policy_{team}')       # '' = rule-based striker
        striker.update({'search_w': float(v('search_w')), 'w_max': float(v('turn_max'))})
        striker.update({'team': team, 'goal_x': sign * L / 2, 'attack_yaw': yaw0, 'ready_offset': start,
                        'use_opponent': v('avoid_opponent').lower() == 'true',
                        'no_lines': v('no_lines').lower() == 'true',   # play to the outer walls
                        'orbit_radius': 0.24,                # as in sim_duel (the last sim version)
                        'play_debug': v('play_debug').lower() == 'true'})
        return GroupAction(scoped=True, actions=[
            *dds, SetEnvironmentVariable('ROS_DOMAIN_ID', str(domain)),
            include('pinky_soccer_nav', 'nav.launch.py',
                    {**field_forward(), 'use_sim_time': 'false', 'real_robot': 'true', 'max_linear': str(vmax),
                     'composition': 'false',       # separate processes: each keeps its own ROS_DOMAIN_ID
                     'initial_x': str(x0), 'initial_y': '0.0', 'initial_yaw': str(yaw0)}),
            Node(package='pinky_soccer_vision', executable='ball_detector', name='ball_detector', output='log',
                 parameters=[{'use_sim_time': False, 'backend': 'hsv', 'compressed': True,
                              'process_every_n': 1, 'fallback_hfov_deg': float(v('camera_hfov')),
                              'range_from_size': v('ball_range') == 'size',   # colour blob size -> range
                              'field_length': L, 'field_width': W, 'goal_width': G,   # boundary line
                              # the URDF says 8 deg down; the real cameras look less down -> correction per
                              # robot (negative = up): camera_calib window result, else cam_pitch_<team>
                              **cam_cfg(team)}]),
            Node(package='pinky_soccer_game', executable='striker', name='striker', output='screen',
                 parameters=[striker]),
        ])

    actions = [
        robot('yellow', yd),
        robot('blue', bd),
        GroupAction(scoped=True, actions=[
            *dds, SetEnvironmentVariable('ROS_DOMAIN_ID', str(yd)),
            Node(package='pinky_soccer_game', executable='real_referee', name='referee', output='screen',
                 parameters=[{'duel': True, 'field_length': L, 'field_width': W, 'goal_width': G,
                              'outer_strip': strip, 'win_score': win, 'game_mode': v('game_mode'),
                              'no_lines': v('no_lines').lower() == 'true'}]),
            Node(package='pinky_soccer_game', executable='match_ui', name='match_ui', output='screen',
                 parameters=[{'win_score': win}]),
        ]),
        GroupAction(scoped=True, actions=[
            *dds,
            Node(package='pinky_soccer_game', executable='domain_relay', output='screen',
                 arguments=['--from-domain', str(yd), '--to-domain', str(bd)]),
        ]),
    ]
    if v('sensor_view').lower() == 'true':
        actions.append(GroupAction(scoped=True, actions=[*dds, Node(
            package='pinky_soccer_game', executable='sensor_view', output='log',
            arguments=['--duel', '--plain', '--yellow-domain', str(yd), '--blue-domain', str(bd)])]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        *field_declarations(),
        DeclareLaunchArgument('yellow_domain', default_value='37'),
        DeclareLaunchArgument('blue_domain', default_value='38'),
        DeclareLaunchArgument('cyclonedds_pc', default_value='~/cyclonedds_pc.xml',
                              description='PC CycloneDDS config: multicast off, both robot IPs as peers'),
        DeclareLaunchArgument('max_speed', default_value='0.28', description='m/s, the Pinky wheel limit'),
        DeclareLaunchArgument('camera_hfov', default_value='53.5',
                              description='deg, used when the robot publishes no camera calibration (OV5647)'),
        DeclareLaunchArgument('cam_pitch_yellow', default_value='-5.3',
                              description='deg, camera pitch correction of the yellow robot (measured)'),
        DeclareLaunchArgument('cam_pitch_blue', default_value='-5.3',
                              description='deg, camera pitch correction of the blue robot (= yellow, the robots are treated as alike)'),
        DeclareLaunchArgument('turn_max', default_value='1.0', description='rad/s max turn rate (rule-based striker)'),
        DeclareLaunchArgument('search_w', default_value='1.6', description='rad/s spin while looking for the ball'),
        DeclareLaunchArgument('policy_yellow', default_value='',
                              description='RL policy .npz (pinky_soccer_rl.export) for yellow; empty = rules'),
        DeclareLaunchArgument('policy_blue', default_value='',
                              description='RL policy .npz for blue; empty = rules'),
        DeclareLaunchArgument('avoid_opponent', default_value='true'),
        DeclareLaunchArgument('win_score', default_value='0'),   # 0: no 'first to N' - the timed match decides
        DeclareLaunchArgument('sensor_view', default_value='true'),
        DeclareLaunchArgument('game_mode', default_value='free',
                              description='free (both attack, yellow kicks off) | attack_defense (old roles)'),
        DeclareLaunchArgument('play_debug', default_value='false'),
        DeclareLaunchArgument('no_lines', default_value='true',
                              description='true: ignore the white lines, the field goes to the outer walls'),
        DeclareLaunchArgument('ball_range', default_value='size',
                              description='size: ball range from its apparent size (colour blob) | floor'),
        OpaqueFunction(function=launch_setup),
    ])
