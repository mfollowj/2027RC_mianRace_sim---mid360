#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROBOCON 场地 + 四全向轮小车 仿真启动（只起仿真与感知，不带 SLAM/Nav2）

  ros2 launch rc_field_nav sim.launch.py                    # 默认: Mid-360(livox) + rviz
  ros2 launch rc_field_nav sim.launch.py gui:=true          # 同时开 gzclient
  ros2 launch rc_field_nav sim.launch.py rviz:=false        # 无界面（远程/压测）

场地路径默认取环境变量 RC_FIELD_DIR（默认本仓库的 gazebo_models），
也可用 world:=/abs/path/xxx.world 直接覆盖。
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            ExecuteProcess, IncludeLaunchDescription, OpaqueFunction,
                            SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

DEFAULT_FIELD_DIR = os.environ.get('RC_FIELD_DIR',
                                   '/home/zx/ITEM/RC_ground_sim/gazebo_models')


def _livox_available():
    """检测你们的 Livox 仿真后端是否可被找到（需要 source 其 install/setup.bash）。"""
    try:
        get_package_share_directory('ros2_livox_simulation')
        return True
    except Exception:
        return False


def _preflight(context, *args, **kwargs):
    """
    启动前自检：Gazebo Classic 的 master 固定监听 11345，如果上一轮仿真没退干净
    （残留 gzserver/gzclient），新 gzserver 不会加载新世界，而是**连到旧 master**上，
    于是出现这些诡异现象：
        * 窗口里有旧场地但没有新车（spawn_entity 报实体已存在 / 落到旧世界里）
        * /cmd_vel 没人响应（驱动插件属于旧实例）或车动不了
        * Gazebo 界面卡死（两套实例抢同一个 master，或两个 gzclient 叠加）
    实测踩过：残留 300~500 秒的 gzserver/gzclient/RSP/fastlio/octomap 一直在跑。
    因此这里默认先清掉残留（clean_stale:=false 可关闭）。
    """
    import subprocess
    clean = context.launch_configurations.get('clean_stale', 'true').lower() in ('1', 'true', 'yes')
    # ★ 只清理"属于本项目"的残留：判据是进程环境里的 ROS_DOMAIN_ID 等于我们的私有域。
    #   绝不能按进程名(gzserver/gzclient/fastlio_mapping)无差别 kill —— 同一台机器/
    #   同一网络里可能还有别人的 ROS 2 仿真在跑（实测存在：别的工程也用 gzserver、
    #   rviz2、Fast-LIO），无差别清理会把别人的东西杀掉。
    domain = os.environ.get('ROS_DOMAIN_ID', '42')
    pat = 'gzserver|gzclient|fastlio_mapping|octomap_server_node|spawn_entity.py|rviz2'

    def _ours(pid):
        try:
            with open('/proc/%s/environ' % pid, 'rb') as f:
                return ('ROS_DOMAIN_ID=%s' % domain).encode() in f.read()
        except OSError:
            return False

    pids = subprocess.run(['pgrep', '-f', pat], capture_output=True, text=True).stdout.split()
    stale = [p for p in pids if _ours(p)]
    if stale:
        if clean:
            print('[rc_field_nav] 清理本项目(DOMAIN %s)的残留进程: %s'
                  % (domain, ','.join(stale)))
            for p in stale:
                subprocess.run(['kill', '-9', p], check=False)
            import time
            time.sleep(2)
        else:
            print('[rc_field_nav][警告] 本项目已有进程在运行(%s)，新 launch 可能连到旧 master 上，'
                  '可能出现"没有小车/车动不了/界面卡死"。建议先手动清理，或用 clean_stale:=true'
                  % ','.join(stale))
    return []


def generate_launch_description():
    pkg_share = get_package_share_directory('rc_field_nav')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # 传感器后端固定为 Livox Mid-360 插件（本项目只保留这一条链路：
    # LiveX CustomMsg 是 LIO 做运动去畸变的前提，stock 射线后端已归档）
    lidar_sensor_xacro = os.path.join(pkg_share, 'urdf', 'lidar_livox_mid360.xacro')

    # ------------------------- 参数声明 -------------------------
    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    spawn_x = LaunchConfiguration('x')
    spawn_y = LaunchConfiguration('y')
    spawn_z = LaunchConfiguration('z')
    spawn_yaw = LaunchConfiguration('yaw')
    robot_name = LaunchConfiguration('robot_name')
    publish_odom_tf = LaunchConfiguration('publish_odom_tf')
    odom_topic = LaunchConfiguration('odom_topic')
    lidar_samples = LaunchConfiguration('lidar_samples')
    lidar_downsample = LaunchConfiguration('lidar_downsample')

    declare_args = [
        DeclareLaunchArgument('world',
                              default_value=os.path.join(DEFAULT_FIELD_DIR, 'robocon_track.world'),
                              description='场地 world 文件绝对路径'),
        DeclareLaunchArgument('gui', default_value='false',
                              description='是否启动 gzclient（Gazebo 图形界面）'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='是否启动 RViz2'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('robot_name', default_value='omni4_field'),
        DeclareLaunchArgument('x', default_value='-4.3', description='出生点 x（红方半场西侧走廊）'),
        DeclareLaunchArgument('y', default_value='1.5', description='出生点 y'),
        DeclareLaunchArgument('z', default_value='0.05', description='出生点 z（略抬高，落地即稳定）'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        # LIO 模式：lio_map.launch.py 会传 publish_odom_tf:=false odom_topic:=odom_gz
        DeclareLaunchArgument('publish_odom_tf', default_value='true',
                              description='Gazebo 插件是否广播 odom->base_link（用 FAST-LIO 时应为 false）'),
        DeclareLaunchArgument('odom_topic', default_value='odom',
                              description='Gazebo 插件里程计话题名'),
        DeclareLaunchArgument('clean_stale', default_value='true',
                              description='启动前清理残留的 gzserver/gzclient（Gazebo master 端口冲突会导致世界不加载/卡死）'),
        DeclareLaunchArgument('lidar_samples', default_value='20000',
                              description='Livox 后端每帧消费的 CSV 行数（=一个完整扫描花样的长度；'
                                          '小于 20000 会出现"整帧无回波"）'),
        DeclareLaunchArgument('lidar_downsample', default_value='4',
                              description='Livox 后端抽稀倍数（每帧点数 = lidar_samples / 本值）'),
    ]

    # ------------------------- 机器人描述 -------------------------
    xacro_file = os.path.join(pkg_share, 'urdf', 'omni4_field.urdf.xacro')
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file,
                 ' lidar_sensor_xacro:=', lidar_sensor_xacro,
                 ' publish_odom_tf:=', publish_odom_tf,
                 ' odom_topic:=', odom_topic,
                 ' lidar_samples:=', lidar_samples,
                 ' lidar_downsample:=', lidar_downsample,
                 ' controllers_yaml:=', os.path.join(pkg_share, 'config',
                                                     'omni4_controllers.yaml')]),
        value_type=str)

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time,
                     'robot_description': robot_description}],
    )

    # ------------------------- Gazebo -------------------------
    # 把 gazebo_models 加进 GAZEBO_MODEL_PATH，这样 model://robocon_ground 可解析，
    # 不必再手动拷到 ~/.gazebo/models，也不依赖 run_gazebo.sh。
    # ★ 用 SetEnvironmentVariable 直接"设定"而不是追加：
    #   gazebo_ros 的 gzclient 启动时会扫描 GAZEBO_MODEL_PATH 下所有模型目录，
    #   如果这里继承了 ROS/SCURC 工作区的一大堆 models（turtlebot3 等），
    #   GUI 启动会长时间卡在建模型列表上（日志里满屏 Missing model.config），
    #   表现为"Gazebo 窗口卡死"。本仿真只需要场地模型，因此只给这一个目录。
    gazebo_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH', value=DEFAULT_FIELD_DIR)

    # 关键（实测）：gzclient 需要 GAZEBO_RESOURCE_PATH 才能找到 OGRE shader 资源，
    # 否则启动后立刻 SIGABRT（exit -6）：
    #   [Err] [RTShaderSystem.cc] Unable to find shader lib ...
    #         Your GAZEBO_RESOURCE_PATH is probably improperly set.
    # 只 source ROS 的 setup.bash 不会设置它，因此这里显式补上（存在才加）。
    # Gazebo Classic 的 master 端口也是全机共用的（默认 11345）：如果本机还有别的
    # Gazebo，gzclient / gz 命令会连到别人的世界上。本项目固定用 11355（env.sh 里设置），
    # 这里再兜一层，保证直接 ros2 launch 时也是同一个端口。
    gazebo_master = SetEnvironmentVariable(
        name='GAZEBO_MASTER_URI',
        value=os.environ.get('GAZEBO_MASTER_URI', 'http://127.0.0.1:11355'))
    gazebo_env_actions = [gazebo_model_path, gazebo_master]
    for _cand in ('/usr/share/gazebo-11', '/usr/share/gazebo'):
        if os.path.isdir(_cand):
            gazebo_env_actions.append(
                AppendEnvironmentVariable(name='GAZEBO_RESOURCE_PATH', value=_cand))
            break

    # 用 gazebo_ros 的 gzserver.launch.py：它会以 -s 方式加载
    #   libgazebo_ros_init.so / libgazebo_ros_factory.so / libgazebo_ros_force_system.so
    # 这三个是 SystemPlugin，**只能**从命令行加载，写进 world 的 <plugin> 会报
    # "incorrect plugin type" 且 /spawn_entity 不出现（实测踩过）。
    # 所以：ROS 仿真一律走这个 launch；world 里只放 WorldPlugin（如 state）。
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world, 'verbose': 'true'}.items(),
    )

    # ★ 不用 gazebo_ros 的 gzclient.launch.py：它会加
    #   --gui-client-plugin=libgazebo_ros_eol_gui.so（Gazebo Classic 停止维护的弹窗），
    #   这个模态对话框会顶住 GUI 事件循环，窗口看起来就是"卡死"、连 Ctrl-C 都要等 5 秒
    #   才被 SIGTERM 强杀（实测日志：failed to terminate '5' seconds after SIGINT）。
    #   这里直接起原生 gzclient，干净且响应快。
    gzclient = ExecuteProcess(
        cmd=['gzclient'],
        output='screen',
        condition=IfCondition(gui),
    )

    # 等 gzserver 与 factory 插件就绪后再生成机器人
    spawn_entity = TimerAction(
        period=4.0,
        actions=[Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_omni4',
            output='screen',
            arguments=['-entity', robot_name,
                       '-topic', 'robot_description',
                       '-x', spawn_x, '-y', spawn_y, '-z', spawn_z,
                       '-Y', spawn_yaw,
                       '-timeout', '120'],
        )],
    )

    # 本项目只保留 physics（真实轮速 + 重力 + 轮地摩擦）这一种驱动：
    # 原来那种 planar_move 运动学驱动会把 z 速度强制为 0，从高处下去会"飞"、
    # 台阶/坡道完全无效，已整体删除。
    spawn_jsb = Node(
        package='controller_manager', executable='spawner',
        name='spawn_joint_state_broadcaster', output='screen',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )
    spawn_wheel = Node(
        package='controller_manager', executable='spawner',
        name='spawn_wheel_velocity_controller', output='screen',
        arguments=['wheel_velocity_controller', '--controller-manager', '/controller_manager'],
    )
    omni_drive = Node(
        package='rc_field_nav', executable='omni_drive.py', name='omni_drive',
        output='screen',
        # ★ publish_odom_tf / odom_topic 必须跟着上层传进来的值走：
        #   与 FAST-LIO 同跑时（lio_map.launch.py 传 publish_odom_tf:=false、
        #   odom_topic:=odom_gz），轮速里程计就不能再发 odom->base_link，
        #   否则和 LIO 抢同一个 TF 父节点，RViz/Nav2 会看到位姿来回跳。
        # 轮系几何必须与 urdf/omni4_body.xacro 一致（560x560 车体、200 mm 全向轮、X 型 45°）
        parameters=[{'use_sim_time': use_sim_time,
                     'wheel_radius': 0.10, 'wheel_x': 0.20, 'wheel_y': 0.20,
                     'publish_odom': True,
                     'publish_odom_tf': publish_odom_tf,
                     'odom_topic': odom_topic}],
    )

    # ------------------------- 感知链路 -------------------------
    # 3D 点云 -> 2D 激光扫描：slam_toolbox / Nav2 的 costmap 与 AMCL 都吃 LaserScan
    pointcloud_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        output='screen',
        remappings=[('cloud_in', '/livox/lidar/pointcloud'),
                    ('scan', '/scan')],
        parameters=[os.path.join(pkg_share, 'config', 'pointcloud_to_laserscan.yaml'),
                    {'use_sim_time': use_sim_time}],
    )

    # ------------------------- RViz -------------------------
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        # 输出到终端（而不是只写日志文件）：RViz 起不来时能立刻看到原因
        output='screen',
        condition=IfCondition(rviz),
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'field_nav.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    ld = LaunchDescription()
    for a in declare_args:
        ld.add_action(a)
    for _a in gazebo_env_actions:
        ld.add_action(_a)
    ld.add_action(robot_state_publisher)
    ld.add_action(gzserver)
    ld.add_action(gzclient)
    ld.add_action(spawn_entity)
    # 控制器/驱动节点要等机器人与 gazebo_ros2_control 插件就绪
    ld.add_action(TimerAction(period=10.0, actions=[spawn_jsb, spawn_wheel]))
    ld.add_action(TimerAction(period=16.0, actions=[omni_drive]))
    ld.add_action(pointcloud_to_laserscan)
    ld.add_action(rviz_node)
    return ld
