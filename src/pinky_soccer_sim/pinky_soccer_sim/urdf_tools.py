"""Build the URDF string for a soccer robot (official Pinky model + bumper / optional guard), ready for ROS / Gazebo."""
import os
import xml.etree.ElementTree as ET

XACRO_REL = os.path.join('urdf', 'pinky_soccer.urdf.xacro')

# Model-level ground-truth pose on /model/<model name>/pose (Pose_V, child_frame_id = model name).
POSE_PUBLISHER = '''<gazebo>
  <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
    <publish_link_pose>false</publish_link_pose>
    <publish_sensor_pose>false</publish_sensor_pose>
    <publish_collision_pose>false</publish_collision_pose>
    <publish_visual_pose>false</publish_visual_pose>
    <publish_nested_model_pose>false</publish_nested_model_pose>
    <publish_model_pose>true</publish_model_pose>
    <use_pose_vector_msg>true</use_pose_vector_msg>
    <static_publisher>false</static_publisher>
    <update_frequency>50</update_frequency>
  </plugin>
</gazebo>'''


def expand_xacro(namespace='', cam_tilt_deg=20.0, dribbler=False, bumper=True, xacro_path=None):
    """namespace: '' for yellow, 'blue/' for the opponent (trailing slash, official upload_robot convention)."""
    import xacro
    if xacro_path is None:
        from ament_index_python.packages import get_package_share_directory
        xacro_path = os.path.join(get_package_share_directory('pinky_soccer_description'), XACRO_REL)
    mappings = {'namespace': namespace, 'is_sim': 'true', 'cam_tilt_deg': str(cam_tilt_deg),
                'dribbler': 'true' if dribbler else 'false', 'bumper': 'true' if bumper else 'false'}
    return xacro.process_file(xacro_path, mappings=mappings).toxml()


TEAM_LED_RGBA = {'yellow': (1.0, 0.82, 0.0, 1.0), 'blue': (0.05, 0.35, 1.0, 1.0)}


def _fix_namespaced_references(root):
    """The official pinky_gz.urdf.xacro writes <gazebo reference="${namespace}link"> but the links themselves
    are NOT prefixed (only joints are).  With namespace "blue/" the sensors and wheel friction blocks would
    point at non-existent links and be dropped -> strip the prefix where only the bare link exists."""
    links = {lk.get('name') for lk in root.findall('link')}
    joints = {j.get('name') for j in root.findall('joint')}
    for g in root.findall('gazebo'):
        ref = g.get('reference')
        if ref and ref not in links and ref not in joints and '/' in ref:
            bare = ref.split('/', 1)[1]
            if bare in links:
                g.set('reference', bare)


def _set_camera(root, width, height, rate):
    for sensor in root.iter('sensor'):
        if sensor.get('type') != 'camera':
            continue
        img = sensor.find('camera/image')
        if img is not None:
            img.find('width').text, img.find('height').text = str(int(width)), str(int(height))
        if sensor.find('update_rate') is not None:
            sensor.find('update_rate').text = str(rate)


def _team_led(root, team):
    """Colour the robot_lamp (WS2812B strip at the rear top) in the team colour, glowing (emissive)."""
    r, g, b, a = TEAM_LED_RGBA[team]
    rgba = f'{r} {g} {b} {a}'
    for lk in root.findall('link'):
        if lk.get('name') != 'robot_lamp':
            continue
        for vis in lk.findall('visual'):
            for m in vis.findall('material'):
                vis.remove(m)
            mat = ET.SubElement(vis, 'material', name=f'{team}_led')
            ET.SubElement(mat, 'color', rgba=rgba)
    root.append(ET.fromstring(
        f'<gazebo reference="robot_lamp"><visual><material>'
        f'<ambient>{rgba}</ambient><diffuse>{rgba}</diffuse><specular>0.2 0.2 0.2 1</specular>'
        f'<emissive>{r * 0.9} {g * 0.9} {b * 0.9} 1</emissive>'
        f'</material></visual></gazebo>'))


def _plain_frames(root):
    """Frame ids without the namespace prefix ('blue/odom' -> 'odom'): every robot then looks like a real Pinky
    (plain frames, its own /<ns>/tf topic).  The gz topics keep their prefix, so the two models stay apart."""
    for tag in ('frame_id', 'child_frame_id', 'gz_frame_id'):
        for el in root.iter(tag):
            if el.text and '/' in el.text:
                el.text = el.text.split('/', 1)[1]


def postprocess(urdf_xml, strip_sensors=False, pose_publisher=True, team=None, camera=None, odom_tf=True,
                plain_frames=False):
    """Soccer does not need the lamp-control plugin (it also needs a compiled gz plugin, x86 only).
    strip_sensors=True removes lidar/camera/IMU (opponent body only -> much cheaper rendering).
    pose_publisher=True adds the ground-truth model pose publisher (referee / opponent / gt backend).
    team='yellow'|'blue' lights the LED lamp in the team colour.
    camera=(width, height, rate) overrides the official 1280x720 @ 30 Hz (rendering cost on small GPUs).
    odom_tf=False: the DiffDrive plugin stops publishing odom -> base_footprint (an EKF does it instead)."""
    root = ET.fromstring(urdf_xml)
    if not odom_tf:
        for tf_topic in root.iter('tf_topic'):
            tf_topic.text = '/diff_drive_tf_unused'
    _fix_namespaced_references(root)
    if plain_frames:
        _plain_frames(root)
    for g in list(root.findall('gazebo')):
        for pl in list(g.findall('plugin')):
            if 'LampControl' in (pl.get('name') or ''):
                g.remove(pl)
        if strip_sensors and g.find('sensor') is not None:
            root.remove(g)
        elif g.get('reference') is None and len(list(g)) == 0:
            root.remove(g)
    if camera is not None:
        _set_camera(root, *camera)
    if team is not None:
        _team_led(root, team)
    if pose_publisher:
        root.append(ET.fromstring(POSE_PUBLISHER))
    return ET.tostring(root, encoding='unicode')


def build_urdf(namespace='', cam_tilt_deg=20.0, dribbler=False, strip_sensors=False, xacro_path=None,
               bumper=True, team=None, camera=None, odom_tf=True, plain_frames=False):
    return postprocess(expand_xacro(namespace, cam_tilt_deg, dribbler, bumper, xacro_path), strip_sensors,
                       team=team, camera=camera, odom_tf=odom_tf, plain_frames=plain_frames)
