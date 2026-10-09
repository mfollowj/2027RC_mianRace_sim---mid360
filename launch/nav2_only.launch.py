#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只起 Nav2 全套（先验 2D 地图 map_server + 规划/控制/行为树）。

为什么把 map_server 也放进来：
  * 这一整套由 nav2_supervisor.py 守护（守护命令必须是**纯字面量**，
    所以参数全部走默认值 / 环境变量，不靠 ":=" 传参）；
  * map_server 的 lifecycle_manager 在本机（只有 Fast DDS）偶发卡在
    "Configuring map_server" 不动（change_state 响应发不出去，实测踩过），
    把它一起交给守护进程，卡死时整轮重启即可自愈。

环境变量（可覆盖默认，便于外层 launch 传路径而不破坏"纯字面量"约束）：
    RC_MAP_YAML   先验 2D 代价地图 yaml（默认包内 maps/prior_costmap.yaml）

也可以单独手动跑（先把仿真 + LIO + 定位器起好）：
    ros2 launch rc_field_nav nav2_only.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('rc_field_nav')
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    map_yaml = LaunchConfiguration('map')

    default_map = os.environ.get(
        'RC_MAP_YAML', os.path.join(pkg_share, 'maps', 'prior_costmap.yaml'))
    # Nav2 参数文件（footprint / 图层 / 膨胀 / 分辨率都在里面）同理走环境变量，
    # 这样从 nav_prior_map.launch.py 传 params_file:= 或单独起本文件都能指定。
    default_params = os.environ.get(
        'RC_NAV2_PARAMS', os.path.join(pkg_share, 'config', 'nav2_omni_params.yaml'))

    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'yaml_filename': map_yaml}],
    )
    lifecycle_map = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_map', output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': autostart},
                    {'node_names': ['map_server']}],
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('nav2_bringup'), 'launch',
                         'navigation_launch.py')),
        launch_arguments={'use_sim_time': use_sim_time,
                          'params_file': params_file,
                          'autostart': autostart,
                          'use_composition': 'False',
                          'use_respawn': 'False'}.items())

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('map', default_value=default_map,
                              description='先验 2D 代价地图 yaml'),
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='Nav2 参数文件（footprint/图层/膨胀/分辨率）'),
        map_server,
        lifecycle_map,
        navigation,
    ])
