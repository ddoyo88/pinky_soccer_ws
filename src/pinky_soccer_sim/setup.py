import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'pinky_soccer_sim'
DATA_DIRS = ['launch', 'config', 'media']

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]
for d in DATA_DIRS:
    files = [f for f in glob(os.path.join(d, '*')) if os.path.isfile(f)]
    if files:
        data_files.append((os.path.join('share', package_name, d), files))

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pinky-soccer',
    maintainer_email='user@example.com',
    description='Parametric soccer field (Gazebo SDF + occupancy map generator), ros_gz bridges, sim launch',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'generate_world = pinky_soccer_sim.generate_world:main',
            'generate_map = pinky_soccer_sim.generate_map:main',
            'mujoco_soccer = pinky_soccer_sim.mujoco_soccer:main',
            'sim_recorder = pinky_soccer_sim.sim_recorder:main',
        ],
    },
)
