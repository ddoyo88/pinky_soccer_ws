"""Game nodes only (sim, nav and detector are started elsewhere - see pinky_soccer_bringup stage6)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pinky_soccer_sim.field import resolve_dims


def launch_setup(context):
    v = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    L, W, G, strip = resolve_dims(v('field_style'), v('field_length'), v('field_width'), v('goal_width'))
    sim = {'use_sim_time': True}
    field = {'field_length': L, 'field_width': W, 'goal_width': G, 'outer_strip': strip}
    match_control = v('match_ui').lower() == 'true'
    nodes = [
        Node(package='pinky_soccer_game', executable='referee', name='referee', output='screen',
             parameters=[sim, field, {'match_control': match_control,
                                      'win_score': int(v('win_score'))}]),
        Node(package='pinky_soccer_game', executable='striker', name='striker', output='screen',
             parameters=[sim, field, {'goal_x': float(v('goal_x')), 'goal_y': float(v('goal_y')),
                                      'match_control': match_control}]),
        Node(package='pinky_soccer_game', executable='opponent_bot', name='opponent_bot', output='screen',
             parameters=[sim, field, {'mode': v('blue_mode'), 'match_control': match_control}]),
    ]
    if match_control:
        nodes.append(Node(package='pinky_soccer_game', executable='match_ui', name='match_ui', output='screen',
                          parameters=[{'win_score': int(v('win_score'))}]))
    if v('sensor_view').lower() == 'true':
        nodes.append(Node(package='pinky_soccer_game', executable='sensor_view', name='sensor_view',
                          output='screen', parameters=[sim]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('field_style', default_value='hardboard'),
        DeclareLaunchArgument('field_length', default_value='auto', description='auto = from the field YAML'),
        DeclareLaunchArgument('field_width', default_value='auto'),
        DeclareLaunchArgument('goal_width', default_value='auto'),
        DeclareLaunchArgument('blue_mode', default_value='attacker', description='attacker|keeper|chaser|patrol|idle'),
        DeclareLaunchArgument('win_score', default_value='5', description='first to this many goals wins (0 = endless)'),
        DeclareLaunchArgument('match_ui', default_value='true',
                              description='show the match control and manual score window'),
        DeclareLaunchArgument('goal_x', default_value='nan', description='target x in the map frame; nan = field_length/2'),
        DeclareLaunchArgument('goal_y', default_value='0.0'),
        DeclareLaunchArgument('sensor_view', default_value='true',
                              description="window with each robot's camera + lidar"),
        OpaqueFunction(function=launch_setup),
    ])
