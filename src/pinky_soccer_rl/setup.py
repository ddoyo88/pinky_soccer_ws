from setuptools import setup

package_name = 'pinky_soccer_rl'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml'])],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pinky_soccer',
    maintainer_email='noreply@example.com',
    description='RL striker for the Pinky soccer robots',
    license='Apache-2.0',
    entry_points={'console_scripts': []},
)
