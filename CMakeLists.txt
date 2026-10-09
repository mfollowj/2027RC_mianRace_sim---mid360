cmake_minimum_required(VERSION 3.8)
project(rc_field_nav)

find_package(ament_cmake REQUIRED)

# 资源：只保留两条工作流需要的（仿真：建图/存图/导航/键盘；真机在仓库 real/ 下独立）
install(DIRECTORY launch urdf config rviz meshes
        DESTINATION share/${PROJECT_NAME})

# 建图产出的地图目录（先建空目录，保证 install 后路径存在）
install(DIRECTORY maps
        DESTINATION share/${PROJECT_NAME}
        OPTIONAL)

install(PROGRAMS
        # ---- 仿真底盘：physics 轮速驱动 + 键盘遥控 ----
        scripts/omni_drive.py
        scripts/omni_teleop.py
        # ---- 建图与存图 ----
        scripts/record_trajectory.py
        scripts/save_pointcloud_pcd.py
        scripts/save_cloud_pcd.py
        scripts/map3d_to_costmap.py
        scripts/map_sanity_check.py
        scripts/save_prior_map.sh
        # ---- 先验地图导航（导航中持续重定位 -> map->odom） ----
        scripts/prior_map_localizer.py
        scripts/nav2_supervisor.py
        # ---- 清场 ----
        scripts/kill_sim.sh
        DESTINATION lib/${PROJECT_NAME}
        OPTIONAL)

ament_package()
