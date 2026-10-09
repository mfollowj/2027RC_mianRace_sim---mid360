#!/usr/bin/env bash
# ============================================================================
#  安全清场：只杀"本项目私有 DDS 域里"的残留进程
# ============================================================================
#  为什么这么写：同一台机器/同一网络里可能还有**别人的** ROS 2 仿真在跑
#  （实测存在：nav_tutorial_ws / pb_rm_simulation，同样用 gzserver、rviz2、
#   Fast-LIO）。如果按进程名无差别 kill，会把别人的仿真一起杀掉。
#  判据：读 /proc/<pid>/environ，看里面的 ROS_DOMAIN_ID 是否等于我们自己的
#  （env.sh 里默认 42）。别人的进程在自己域里，绝不会被误杀。
#
#  用法：
#     ros2 run rc_field_nav kill_sim.sh          # 清理
#     ros2 run rc_field_nav kill_sim.sh --list   # 只看有什么在跑
# ============================================================================
set +u
LIST_ONLY=0
[ "${1:-}" = "--list" ] && LIST_ONLY=1

DOMAIN="${ROS_DOMAIN_ID:-42}"
echo "[清场] 只处理 ROS_DOMAIN_ID=${DOMAIN} 的进程（别人的域不动）"

# 本脚本自己 + 所有祖先进程要跳过（否则会把调用它的 shell 一起 SIGKILL）
SELF=$$
SKIP=" $SELF "
p=$SELF
for _ in $(seq 1 20); do
    ppid=$(awk '{print $4}' /proc/$p/stat 2>/dev/null)
    [ -z "$ppid" ] && break
    [ "$ppid" = "0" ] && break
    SKIP="$SKIP$ppid "
    p=$ppid
done

found=0
for pid in $(pgrep -f 'gzserver|gzclient|rviz2|fastlio_mapping|fastlivo_mapping|octomap_server_node|spawn_entity.py|robot_state_publisher|controller_manager|camera_container|livox' 2>/dev/null)            $(pgrep -f 'rc_field_nav|omni_drive|omni_teleop|prior_map_localizer|record_trajectory|save_cloud_pcd|traj_recorder|livox_preprocess|nav2_supervisor|nav2_watchdog|reloc_pose_to_tf|map_server|lifecycle_manager|planner_server|controller_server|bt_navigator|behavior_server|waypoint_follower|velocity_smoother|smoother_server' 2>/dev/null); do
    case "$SKIP" in *" $pid "*) continue ;; esac
    [ -r "/proc/$pid/environ" ] || continue
    # ★ 只认"我们域里"的进程
    grep -qz "ROS_DOMAIN_ID=${DOMAIN}" "/proc/$pid/environ" 2>/dev/null || continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
    case "$cmd" in *kill_sim*) continue ;; esac
    found=1
    echo "  [$pid] $(echo $cmd | cut -c1-80)"
    [ "$LIST_ONLY" = "0" ] && kill -9 "$pid" 2>/dev/null
done

[ "$found" = "0" ] && echo "  没有发现本项目的残留进程（干净）"
[ "$LIST_ONLY" = "0" ] && { sleep 2; echo "清理完成。"; }
exit 0

# ---------------------------------------------------------------------------
#  补充清理：进程级残留（launch 被杀后子进程会变孤儿，ROS 图里看不到但 pgrep 到）
#  典型：fastlio_mapping / save_cloud_pcd.py / record_trajectory.py / omni_drive.py
#  这就是"check_lidar 说没驱动、real_map.sh 说已有驱动在跑"的根因。
# ---------------------------------------------------------------------------
for pat in 'fastlio_mappin[g]' 'fastlivo_mappin[g]' 'livox_ros_driver2_nod[e]' \
           'save_cloud_pcd\.py' 'record_trajector\.y' 'prior_map_localize[r]\.py' \
           'omni_driv[e]\.py' 'omni_teleop\.py' 'pointcloud_to_lasersca[n]'; do
    pkill -f "${pat}" 2>/dev/null && echo "  已清理: ${pat}"
done
sleep 1
# 孤儿 gzserver/gzclient（只清本项目 GAZEBO_MASTER_URI 对应的那两个）
pkill -f 'gzserve[r]' 2>/dev/null && echo "  已清理: gzserver"
pkill -f 'gzclien[t]' 2>/dev/null && echo "  已清理: gzclient"
