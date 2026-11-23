import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'pinky_soccer_game'
DATA_DIRS = ['launch', 'config']

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
    description='Game logic: striker (approach/dribble/shoot/avoid), referee (score/reset), scripted opponent',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'striker = pinky_soccer_game.striker:main',
            'referee = pinky_soccer_game.referee:main',
            'real_referee = pinky_soccer_game.real_referee:main',
            'domain_relay = pinky_soccer_game.domain_relay:main',
            'opponent_bot = pinky_soccer_game.opponent_bot:main',
            'match_ui = pinky_soccer_game.match_ui:main',
            'sensor_view = pinky_soccer_game.sensor_view:main',
        ],
    },
)
