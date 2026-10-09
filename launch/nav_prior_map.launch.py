#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""先验地图导航 + 导航中持续重定位（一条命令起全套）

② 仿真导航：先验地图 + 导航中持续重定位（**不用 AMCL**）
    map->odom 由 prior_map_localizer.py（当前点云对先验点云做体素重叠匹配）持续估计，
    LIO 只负责 odom->base_link。先验图由 save_prior_map.sh 产出，两者坐标系一致。
    ★ 出生点必须与建图时相同（默认 -4.3, 1.5, yaw 0）：LIO 的 odom 原点就是启动位置。

    Gazebo 仿真 (Livox Mid-360 + IMU)
        ├─ /livox/lidar ─┐
        └─ /livox/imu ───┴─> FAST-LIO ──> odom->base_link, /cloud_registered(odom 系)
                                 │
    /cloud_registered ─> prior_map_localizer ──> TF map->odom（周期重定位 + 门限 + 平滑）
                                 │                 /localization_pose
                                 v                 /relocalization_result
                        先验地图 maps/prior_map.pcd
                                 │
    map_server(先验 2D 代价地图) ─┴─> Nav2 (planner/controller/BT) ─> /cmd_vel

用法
----
    ros2 launch rc_field_nav nav_prior_map.launch.py gui:=true
    # 目标点：
    ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
      "{pose: {header: {frame_id: map}, pose: {position: {x: 3.9, y: 4.5}, orientation: {w: 1.0}}}}"

"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, SetEnvironmentVariable,
                            TimerAction, SetLaunchConfiguration)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

DEFAULT_FIELD_DIR = os.environ.get('RC_FIELD_DIR',
                                   '/home/zx/ITEM/RC_ground_sim/gazebo_models')


def generate_launch_description():
    pkg_share = get_package_share_directory('rc_field_nav')
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')

    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')
    # ★ sim.launch.py 的 rviz 参数要传 false（不要它自带的 2D RViz），
    #   但 ROS 2 launch 的配置是**全局共享**的，这个参数会把本文件自己的
    #   rviz 也改成 false —— 结果就是 RViz 根本不启动。
    #   所以先把用户给的值存到 rviz_self，后面就用 rviz_self 判断。
    rviz_self = LaunchConfiguration('rviz_self')
    world = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    yaw = LaunchConfiguration('yaw')
    prior_map = LaunchConfiguration('prior_map')
    costmap_yaml = LaunchConfiguration('map')
    # 2D 图的 meta.json（同一次存图产出）里有权威的 ground_z，
    # 重定位直接读它，不再写死 -0.30 —— 以后改 IMU 安装高度也不用动这里
    costmap_meta = PythonExpression(
        ["'", costmap_yaml, "'.replace('.yaml', '.meta.json')"])
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    lio_config_dir = LaunchConfiguration('lio_config_dir')
    lio_config_file = LaunchConfiguration('lio_config_file')

    reloc_period = LaunchConfiguration('reloc_period')
    initial_pose = LaunchConfiguration('initial_pose')
    min_score = LaunchConfiguration('min_score')
    nav2_timeout = LaunchConfiguration('nav2_timeout')

    declare_args = [
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('world',
                              default_value=os.path.join(DEFAULT_FIELD_DIR, 'robocon_track.world')),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('x', default_value='-4.3', description='出生点 x'),
        DeclareLaunchArgument('y', default_value='1.5', description='出生点 y'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('autostart', default_value='true'),
        # ★ 这两个文件必须来自**同一次** save_prior_map.sh 的产物（同一坐标系）：
        #   prior_map.pcd 是 LIO 的 odom 系点云，prior_costmap.yaml 是同一套点云
        #   投影出来的 2D 图。混用不同次建图的文件会出现"重定位结果对、但代价地图
        #   整体错位"（实测踩过）。出生点也要和建图时一致（默认 -4.3, 1.5, yaw 0）。
        DeclareLaunchArgument(
            'prior_map',
            default_value=os.path.join(pkg_share, 'maps', 'prior_map.pcd'),
            description='先验 3D 点云地图（重定位用；由 save_prior_map.sh 产出）'),
        DeclareLaunchArgument(
            'map',
            default_value=os.path.join(pkg_share, 'maps', 'prior_costmap.yaml'),
            description='先验 2D 代价地图（Nav2 map_server 用；同一次建图的产物）'),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(pkg_share, 'config', 'nav2_omni_params.yaml')),
        DeclareLaunchArgument('lio_config_dir',
                              default_value=os.path.join(pkg_share, 'config')),
        DeclareLaunchArgument('lio_config_file', default_value='fastlio_field_mid360.yaml'),
        DeclareLaunchArgument('reloc_period', default_value='6.0',
                              description='重定位周期（秒）'),
        DeclareLaunchArgument(
            'initial_pose', default_value='0,0,0',
            description='重定位初始位姿 "x,y,yaw_度"（建图/导航同一起点时就是 0,0,0，'
                        '直接跳过 ~60 s 的全图搜索）；车停在别处就给真实值；'
                        '想强制全图搜索给 auto'),
        DeclareLaunchArgument('min_score', default_value='0.55',
                              description='体素重叠度阈值，低于此值丢弃该次重定位'),
        DeclareLaunchArgument('nav2_timeout', default_value='75.0',
                              description='Nav2 bringup 单轮等待激活的最长时间（秒），'
                                          '超时由 nav2_supervisor.py 整轮重启'),
    ]

    # ---------------- 1) 仿真：odom->base_link 交给 LIO ----------------
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'sim.launch.py')),
        launch_arguments={'rviz': 'false', 'gui': gui, 'world': world,
                          'publish_odom_tf': 'false',
                          'odom_topic': 'odom_gz',
                          'use_sim_time': use_sim_time,
                          'x': x, 'y': y, 'yaw': yaw}.items(),
    )


    # ---------------- 2) LIO 前端（只提供 odom->base_link，与建图同一套） ----------------
    fastlio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('fast_lio'), 'launch',
                         'mapping.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time,
                          'config_path': lio_config_dir,
                          'config_file': lio_config_file,
                          'rviz': 'false'}.items(),
    )

    # ---------------- 3) 先验地图定位器：map->odom ----------------
    # ★ 默认给一个初始位姿 (0,0,0)：本工程的规矩是"建图与导航用同一个出生点"，
    #   所以 map->odom 本来就该接近 0。给了初值就跳过全图搜索 ——
    #   实测 15.7x15.7 m 的图全图搜索要 60 s，而给初值后 1~2 s 就锁上；
    #   初值不对也没关系：跟踪几帧分数太低会自动触发全图重锁（force_full）。
    #   车确实停在别处时：initial_pose:="x,y,yaw(度)"；想强制全图搜索：initial_pose:=auto
    loc_base = ['--prior-map', prior_map,
                '--cloud-topic', '/cloud_registered',
                '--odom-topic', '/Odometry',
                '--period', reloc_period,
                '--min-score', min_score,
                '--window', '40', '--window-sec', '6.0',
                '--acc-voxel', '0.03',
                # ground_z 从存图时的 meta.json 读（没有 meta 时节点会退回按
                # IMU 离地高度估算，所以这里不再写死数值）
                '--meta', costmap_meta]
    use_seed = PythonExpression(["'", initial_pose, "' != 'auto'"])
    use_auto = PythonExpression(["'", initial_pose, "' == 'auto'"])
    localizer = Node(
        package='rc_field_nav', executable='prior_map_localizer.py',
        name='prior_map_localizer', output='screen',
        condition=IfCondition(use_seed),
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=loc_base + ['--initial-pose', initial_pose],
        # 地图名与导航给的不一致时给出"照着做"的报错（见 check_prior_map）
    )
    localizer_auto = Node(
        package='rc_field_nav', executable='prior_map_localizer.py',
        name='prior_map_localizer', output='screen',
        condition=IfCondition(use_auto),
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=loc_base,
    )

    # ---------------- 4) Nav2 全套（map_server + 规划/控制/BT/行为） ----------------
    #   map_server 也放进被守护的 nav2_only.launch.py：本机只有 Fast DDS 时，
    #   map_server 的 lifecycle_manager 偶发卡在 "Configuring map_server" 不动
    #   （change_state 响应发不出去，实测踩过），一起交给守护进程才能自愈。
    # ★ 用 nav2_supervisor.py 起，而不是直接 IncludeLaunchDescription：
    #   本机只有 Fast DDS，实测在"仿真+LIO+定位器"同时跑的时候，
    #   lifecycle_manager 连续切换 7 个节点的生命周期时偶发某次 change_state 的
    #   **响应发不出去**（服务端写超时），客户端永远等不到响应 ->
    #   lifecycle_manager 停在 "Configuring velocity_smoother"，
    #   bt_navigator 永远是 inactive（同一套配置 4 次里成功 2 次）。
    #   守护进程会轮询 bt_navigator 状态，超时没激活就把整轮 bringup 杀掉重启。
    # 注意：这里调的是本包自己的 nav2_only.launch.py（参数全有默认值），
    # 所以守护命令是一条**纯字面量**。不要在这里拼 "use_sim_time:=<值>" ——
    # ExecuteProcess 的 cmd 列表会把 "名:=" 和 "值" 拆成两个 argv 元素，
    # ros2 launch 直接报 malformed launch argument 退出（实测踩过）。
    navigation = ExecuteProcess(
        output='screen',
        cmd=['ros2', 'run', 'rc_field_nav', 'nav2_supervisor.py',
             '--node', 'bt_navigator',
             '--timeout', nav2_timeout,
             '--retries', '4',
             '--watch-timeout', '30.0',
             '--',
             'ros2', 'launch', 'rc_field_nav', 'nav2_only.launch.py'],
    )

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        condition=IfCondition(rviz_self),
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'field_nav.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    ld = LaunchDescription()
    for a in declare_args:
        ld.add_action(a)
    # 在 sim 包进来之前先把用户给的 rviz 存起来
    ld.add_action(SetLaunchConfiguration('rviz_self', rviz))
    ld.add_action(sim)
    ld.add_action(TimerAction(period=10.0, actions=[fastlio]))
    # 定位器要等 /cloud_registered；地图服务器与它同时起
    ld.add_action(TimerAction(period=14.0, actions=[localizer, localizer_auto]))
    # 先验 2D 地图路径 / Nav2 参数文件都通过**环境变量**传给被守护的
    # nav2_only.launch.py —— 守护进程的命令必须是纯字面量，不能在里面拼
    # "map:=<路径>" / "params_file:=<路径>"（ExecuteProcess 会把 name:=value 拆成两个 argv）
    ld.add_action(SetEnvironmentVariable('RC_MAP_YAML', costmap_yaml))
    ld.add_action(SetEnvironmentVariable('RC_NAV2_PARAMS', params_file))
    # Nav2 必须等 map->odom 存在（否则代价地图一直报 "frame does not exist"）
    ld.add_action(TimerAction(period=50.0, actions=[navigation]))
    ld.add_action(rviz_node)
    return ld
