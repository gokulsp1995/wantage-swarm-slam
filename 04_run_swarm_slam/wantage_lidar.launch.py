"""Swarm-SLAM on the Wantage walks: one Swarm-SLAM instance per walk ("robot").

Odometry is NOT computed here. Each robot expects, in its namespace /rN:
  /rN/odom        nav_msgs/Odometry     (FAST-LIO /Odometry, remapped at bag play)
  /rN/pointcloud  sensor_msgs/PointCloud2 (FAST-LIO /cloud_registered_body)
  /rN/gps/fix     sensor_msgs/NavSatFix  (RTK, only used when GPS recording is on)

Run:  ros2 launch <this_file> max_nb_robots:=4
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, PopLaunchConfigurations,
                            PushLaunchConfigurations, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

HERE = os.path.dirname(os.path.realpath(__file__))


def launch_setup(context, *args, **kwargs):
    max_nb_robots = int(LaunchConfiguration('max_nb_robots').perform(context))
    robot_delay_s = float(LaunchConfiguration('robot_delay_s').perform(context))
    config_file = LaunchConfiguration('config_file').perform(context)
    rendezvous = os.path.join(
        HERE, LaunchConfiguration('rendezvous_config').perform(context))
    cslam_lidar = os.path.join(
        get_package_share_directory('cslam_experiments'),
        'launch', 'cslam', 'cslam_lidar.launch.py')

    schedule = []
    for i in range(max_nb_robots):
        robot = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(cslam_lidar),
            launch_arguments={
                'config_path': HERE + '/',
                'config_file': config_file,
                'robot_id': str(i),
                'namespace': '/r' + str(i),
                'max_nb_robots': str(max_nb_robots),
                'enable_simulated_rendezvous':
                    LaunchConfiguration('enable_simulated_rendezvous'),
                'rendezvous_schedule_file': rendezvous,
            }.items(),
        )
        # Scope each include so robot i's arguments don't leak into robot i+1.
        schedule.append(PushLaunchConfigurations())
        schedule.append(TimerAction(period=robot_delay_s * i, actions=[robot]))
        schedule.append(PopLaunchConfigurations())
    return schedule


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('max_nb_robots', default_value='4'),
        DeclareLaunchArgument('config_file', default_value='wantage_lidar.yaml'),
        # Keep 0 when using a rendezvous schedule: schedule times are counted
        # from each robot's own start, so staggered starts shift the windows.
        DeclareLaunchArgument('robot_delay_s', default_value='0'),
        DeclareLaunchArgument('enable_simulated_rendezvous', default_value='false'),
        DeclareLaunchArgument('rendezvous_config',
                              default_value='rendezvous_end.config'),
        OpaqueFunction(function=launch_setup),
    ])
