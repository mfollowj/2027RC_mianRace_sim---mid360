#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""① 仿真建图：Gazebo(physics 驱动) + FAST-LIO2(LiDAR-惯性紧耦合) + 轨迹记录 + RViz

流程（配合仓库根目录 run_sim_map.sh 使用）：
    起本 launch -> 另开终端 teleop.sh 用键盘把场地走一遍
    -> 另开终端 save_map.sh 存成 prior_map.pcd / prior_costmap.yaml
    -> 再起 nav_prior_map.launch.py 用这张先验图导航

TF 约定：
  * odom->base_link 由 FAST-LIO 发布（sim.launch.py 传 publish_odom_tf:=false，
    轮速里程计改名 /odom_gz 只用于对照漂移）；
  * FAST-LIO 的地图在 odom 帧，odom 原点 = 启动那一刻车的位置
    -> **建图和导航要用同一个出生点**（默认 -4.3, 1.5, yaw 0）。

用法
----
    ros2 launch rc_field_nav lio_map.launch.py gui:=true rviz:=true
   ros2 run rc_field_nav save_field_map3d.sh       # 存 3D 点云 + 轨迹

雷达数据链路（重要）：本 launch 默认 **不做 Python 量程滤波**（filter:=false）。
量程滤波是早期版本用来丢"无回波假点"的补丁；现在 rc_field_nav 自带的
ros2_livox_simulation 插件已经修掉了根因（无回波射线不再发布 (0,0,0)，
且量程不再系统性偏小 0.1m），所以不需要再串一个 Python 节点。
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction, SetLaunchConfiguration)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DEFAULT_FIELD_DIR = os.environ.get('RC_FIELD_DIR',
                                   '/home/zx/ITEM/RC_ground_sim/gazebo_models')


def generate_launch_description():
    pkg_share = get_package_share_directory('rc_field_nav')

    try:
        fastlio_share = get_package_share_directory('fast_lio')
    except Exception as exc:                      # noqa: BLE001
        raise RuntimeError(
            '[rc_field_nav] 找不到 fast_lio 包。请先 source 你们的工作空间：\n'
            '  source /home/zx/SCURC_Nav_Sim/install/setup.bash\n'
            '（原始错误：%s）' % exc)

    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')
    # ★ sim.launch.py 的 rviz 参数要传 false（不要它自带的 2D RViz），
    #   但 ROS 2 launch 的配置是**全局共享**的，这个参数会把本文件自己的
    #   rviz 也改成 false —— 结果就是 RViz 根本不启动。
    #   所以先把用户给的值存到 rviz_self，后面就用 rviz_self 判断。
    rviz_self = LaunchConfiguration('rviz_self')

    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    yaw = LaunchConfiguration('yaw')
    lio_config_dir = LaunchConfiguration('lio_config_dir')
    lio_config_file = LaunchConfiguration('lio_config_file')

    record_traj = LaunchConfiguration('record_trajectory')
    traj_file = LaunchConfiguration('traj_file')
    lidar_samples = LaunchConfiguration('lidar_samples')
    lidar_downsample = LaunchConfiguration('lidar_downsample')

    default_lio_cfg_dir = os.path.join(pkg_share, 'config')
    default_lio_cfg_file = 'fastlio_field_mid360.yaml'
    default_traj = os.path.expanduser('~/rc_field_traj.txt')

    declare_args = [
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),

        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world',
                              default_value=os.path.join(DEFAULT_FIELD_DIR, 'robocon_track.world')),
        DeclareLaunchArgument('x', default_value='-4.3'),
        DeclareLaunchArgument('y', default_value='1.5'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('lio_config_dir', default_value=default_lio_cfg_dir,
                              description='FAST-LIO 参数文件所在目录'),
        DeclareLaunchArgument('lio_config_file', default_value=default_lio_cfg_file,
                              description='FAST-LIO 参数文件名'),
        DeclareLaunchArgument('record_trajectory', default_value='true',
                              description='是否把 LIO 轨迹记下来（生成 2D 代价地图时射线投射要用）'),
        DeclareLaunchArgument('traj_file', default_value=default_traj,
                              description='轨迹输出文件'),
        DeclareLaunchArgument('lidar_samples', default_value='20000',
                              description='Livox 插件每帧消费的 CSV 行数（=一个完整扫描花样）'),
        DeclareLaunchArgument('lidar_downsample', default_value='4'),
    ]

    # ---- 1) 仿真：Livox 后端 + 关掉插件的 odom TF（交给 LIO）----
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'sim.launch.py')),
        launch_arguments={'rviz': 'false', 'gui': gui, 'world': world,
                          'publish_odom_tf': 'false',        # odom->base_link 交给 FAST-LIO
                          'odom_topic': 'odom_gz',
                          'use_sim_time': use_sim_time,
                          'lidar_samples': lidar_samples,
                          'lidar_downsample': lidar_downsample,
                          'x': x, 'y': y, 'yaw': yaw}.items(),
    )


    # ---- 2) FAST-LIO2 ----
    fastlio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fastlio_share, 'launch', 'mapping.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time,
                          'config_path': lio_config_dir,
                          'config_file': lio_config_file,
                          'rviz': 'false'}.items(),
    )

    # ---- 2.5) 轨迹记录：给 map3d_to_costmap.py 做射线投射（生成空闲区）----
    traj = Node(
        package='rc_field_nav',
        executable='record_trajectory.py',
        name='record_trajectory',
        output='screen',
        condition=IfCondition(record_traj),
        arguments=['-t', '/Odometry', '-o', traj_file],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ---- 2.7) 全程累积点云：边建图边把 /cloud_registered 攒成一张稠密地图 ----
    #   为什么不用 /Laser_map：那是 LIO 内部按 filter_size_map(0.1 m) 降采样后的地图，
    #   实测存下来只有 ~77 点/m²、73% 的 5 cm 格子只有 1 个点 —— 做 2D 代价地图必然
    #   "墙是虚线、空地随机出点"。这里改成累积每一帧配准后的点云（体素去重，内存与
    #   地图大小有关、与跑多久无关），收到 /save_prior_map 触发再落盘。
    #   落盘位置用临时名 accum.pcd，由 save_prior_map.sh 触发后改名成 <名字>_map.pcd。
    accumulator = Node(
        package='rc_field_nav',
        executable='save_cloud_pcd.py',
        name='cloud_accumulator',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['--topic', '/cloud_registered',
                   '-o', os.path.join(pkg_share, 'maps', 'accum.pcd'),
                   '--accumulate', '--trigger-topic', '/save_prior_map',
                   '--voxel', '0.02', '--verbose'],
    )

    # ---- 3) RViz（3D 视图）----
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',        # 输出到终端：RViz 起不来时能直接看到原因
        condition=IfCondition(rviz_self),
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'field_lio.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    ld = LaunchDescription()
    for a in declare_args:
        ld.add_action(a)
    # 在 sim 包进来之前先把用户给的 rviz 存起来
    ld.add_action(SetLaunchConfiguration('rviz_self', rviz))
    ld.add_action(sim)
    # FAST-LIO 要等雷达/IMU 有数据
    ld.add_action(TimerAction(period=10.0, actions=[fastlio, traj, accumulator]))
    ld.add_action(rviz_node)
    return ld
