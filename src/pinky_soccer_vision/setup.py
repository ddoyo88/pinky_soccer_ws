import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'pinky_soccer_vision'
DATA_DIRS = ['launch', 'config', 'robot']

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
    description='Golf-ball perception: HSV baseline, YOLO detector, auto-labelled dataset collector, trainer',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'ball_detector = pinky_soccer_vision.ball_detector:main',
            'collect_dataset = pinky_soccer_vision.collect_dataset:main',
            'train_yolo = pinky_soccer_vision.train_yolo:main',
            'camera_calib = pinky_soccer_vision.camera_calib_ui:main',
        ],
    },
)
