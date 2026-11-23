import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'pinky_soccer_bringup'
DATA_DIRS = ['launch']

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
    description='Stage-by-stage launch files (stage1 robot -> stage6 full game)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [

        ],
    },
)
