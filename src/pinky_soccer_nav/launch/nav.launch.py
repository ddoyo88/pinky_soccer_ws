"""Nav2 + AMCL on the soccer field = the OFFICIAL pinky_navigation/launch/gz_bringup_launch.xml with our params.

Only two things are prepared here:
  * the map: generated from the field size (default) or any map:=/path/scanned_field.yaml you saved in stage 3
  * the AMCL initial pose, patched into a copy of nav2_params.yaml (yellow spawns at x = -field_length/4, y = 0)

ros2 launch pinky_soccer_nav nav.launch.py                                   # ideal map of the field YAML (hardboard)
ros2 launch pinky_soccer_nav nav.launch.py field_style:=box field_length:=2.4 field_width:=1.8 # old field
ros2 launch pinky_soccer_nav nav.launch.py map:=$HOME/maps/scanned_field.yaml # your SLAM map
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch_ros.actions import PushRosNamespace
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from pinky_soccer_sim.field import FieldSpec, resolve_dims, write_map

OUT_DIR = '/tmp/pinky_soccer'


def _val(context, name):
    return LaunchConfiguration(name).perform(context)


def launch_setup(context):
    share = get_package_share_directory('pinky_soccer_nav')
    nav_share = get_package_share_directory('pinky_navigation')
    os.makedirs(OUT_DIR, exist_ok=True)

    style = _val(context, 'field_style')
    L, W, G, _ = resolve_dims(style, _val(context, 'field_length'), _val(context, 'field_width'),
                              _val(context, 'goal_width'))
    custom_map = _val(context, 'map')
    if custom_map:
        map_yaml = custom_map
        # a scanned map has its origin at the robot's start pose -> robot starts at (0, 0, 0) in it
        auto_pose = (0.0, 0.0, 0.0)
    else:
        spec = FieldSpec(length=L, width=W, goal_width=G, style=style)
        map_yaml = write_map(spec, os.path.join(OUT_DIR, 'soccer_map'))
        auto_pose = (-L / 4.0, 0.0, 0.0)

    def pick(name, default):
        v = _val(context, name)
        return default if v == 'auto' else float(v)

    pose = {'x': pick('initial_x', auto_pose[0]), 'y': pick('initial_y', auto_pose[1]),
            'z': 0.0, 'yaw': pick('initial_yaw', auto_pose[2])}

    with open(os.path.join(share, 'config', 'nav2_params.yaml')) as f:
        params = yaml.safe_load(f)
    params['amcl']['ros__parameters']['initial_pose'] = pose
    sim_time = _val(context, 'use_sim_time').lower() == 'true'
    for node in params.values():                        # one switch for every Nav2 node
        if isinstance(node, dict) and isinstance(node.get('ros__parameters'), dict):
            node['ros__parameters']['use_sim_time'] = sim_time
    if _val(context, 'real_robot').lower() == 'true':
        # real Pinky: wheels slip against walls / the goal -> trust the lidar more, the odometry less
        amcl = params['amcl']['ros__parameters']
        amcl.update({'z_hit': 0.9, 'z_rand': 0.1, 'sigma_hit': 0.05,
                     'alpha1': 0.4, 'alpha2': 0.4, 'alpha3': 0.4, 'alpha4': 0.4, 'alpha5': 0.4,
                     'laser_likelihood_max_dist': 0.5, 'max_beams': 120})
    vmax = _val(context, 'max_linear')
    if vmax != 'auto':                                  # real Pinky: 100 rpm wheels = 0.28 m/s
        vmax = float(vmax)
        rpp = params['controller_server']['ros__parameters']['FollowPath']
        rpp['desired_linear_vel'] = min(rpp['desired_linear_vel'], vmax)
        vs = params['velocity_smoother']['ros__parameters']
        vs['max_velocity'][0] = min(vs['max_velocity'][0], vmax)
        vs['min_velocity'][0] = max(vs['min_velocity'][0], -vmax)
    # soccer behaviour tree: the default one has a Spin recovery (robot turning in place next to the ball)
    params['bt_navigator']['ros__parameters']['default_nav_to_pose_bt_xml'] = os.path.join(
        share, 'behavior_trees', 'soccer_navigate_to_pose.xml')
    ns = _val(context, 'namespace').strip('/')
    if ns:
        # several robots: /<ns>/scan ... (Nav2's own launch already remaps /tf -> /<ns>/tf);
        # absolute topics in the params become relative, and the file is nested under the namespace so
        # that /<ns>/amcl etc. pick their parameters up
        def relative(node):
            if isinstance(node, dict):
                return {k: (v.lstrip('/') if k in ('topic', 'scan_topic', 'odom_topic', 'map_topic')
                            and isinstance(v, str) else relative(v)) for k, v in node.items()}
            return node
        params = {ns: relative(params)}
    # one file per robot: several robots (namespaces and/or ROS domains) are launched together and would
    # otherwise overwrite each other's file (initial pose!) before their AMCL reads it
    tag = (f'_{ns}' if ns else '') + f'_d{os.environ.get("ROS_DOMAIN_ID", "0")}'
    params_path = os.path.join(OUT_DIR, f'nav2_params{tag}.yaml')
    with open(params_path, 'w') as f:
        yaml.safe_dump(params, f, sort_keys=False)

    comp = _val(context, 'composition').lower()
    # composition loads into 'nav2_container' by its plain name (not found inside a namespace) and through the
    # launch process's own ROS node, which lives in whatever domain it was created in -> with several robots in
    # several domains every robot's Nav2 would land in the first robot's container: separate nodes then
    composition = ('False' if ns else 'True') if comp == 'auto' else ('True' if comp == 'true' else 'False')
    nav = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(nav_share, 'launch', 'gz_bringup_launch.xml')),
        # composition loads into 'nav2_container' by its plain name -> not found inside a namespace: separate nodes.
        # The namespace is pushed here once: the official XML would push it again in every sub-launch
        # (launch arguments are inherited) -> /<ns>/<ns>/amcl
        launch_arguments={'map': map_yaml, 'params_file': params_path, 'namespace': '',
                          'use_composition': composition,
                          'use_sim_time': 'True' if sim_time else 'False'}.items())
    if not ns:
        return [nav]
    return [GroupAction(scoped=True, actions=[PushRosNamespace(ns), nav])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map', default_value='', description='empty = generate the ideal field map'),
        DeclareLaunchArgument('field_style', default_value='hardboard'),
        DeclareLaunchArgument('field_length', default_value='auto'),
        DeclareLaunchArgument('field_width', default_value='auto'),
        DeclareLaunchArgument('goal_width', default_value='auto'),
        DeclareLaunchArgument('initial_x', default_value='auto'),
        DeclareLaunchArgument('initial_y', default_value='auto'),
        DeclareLaunchArgument('initial_yaw', default_value='auto'),
        DeclareLaunchArgument('use_sim_time', default_value='true', description='false on the real robot'),
        DeclareLaunchArgument('namespace', default_value='',
                              description='robot namespace (yellow / blue): /<ns>/scan, /<ns>/cmd_vel, /<ns>/tf'),
        DeclareLaunchArgument('composition', default_value='auto',
                              description='auto | true | false (false: Nav2 as separate processes)'),
        DeclareLaunchArgument('real_robot', default_value='false',
                              description='true: AMCL tuned for the real Pinky (lidar over slipping odometry)'),
        DeclareLaunchArgument('max_linear', default_value='auto', description='m/s cap (real Pinky: 0.28)'),
        OpaqueFunction(function=launch_setup),
    ])
