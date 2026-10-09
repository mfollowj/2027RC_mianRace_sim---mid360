#!/usr/bin/env bash
# ============================================================================
#  存图（仿真侧）：把当前建图结果落盘成"先验地图三件套"
# ============================================================================
#  前提：lio_map.launch.py 还在跑（脚本要触发内存里的点云累积器），并且车已经开过场地。
#
#  ★ 点云来源：**全程累积的 /cloud_registered**（每个 5 cm 格子有十几个点），
#    而不是 LIO 内部按 filter_size_map 降采样后的 /Laser_map
#    （实测后者只有 ~77 点/m²、73% 的 5 cm 格子只有 1 个点 —— 做 2D 代价地图必然
#     "墙是虚线、空地随机出点"）。累积器在 lio_map.launch.py 里随建图启动。
#
#  产物（默认写到包内 maps/）：
#      <名字>_map.pcd           先验 3D 点云（稠密）
#      <名字>_traj.txt          建图轨迹
#      <名字>_costmap.pgm/.yaml Nav2 map_server 用的 2D 占据栅格
#      <名字>_costmap.meta.json ground_z / imu_height / 分辨率 + terrain 统计
#      <名字>_costmap_elev.npy  当地地面高度场（方案A 中间产物）
#      <名字>_costmap_cost.pgm  代价栅格（方案甲产物，给将来的 costmap layer 用）
#      <名字>_costmap_traversability.png  可通行性预览
#
#  用法：
#      ros2 run rc_field_nav save_prior_map.sh              # 默认名字 prior
#      ros2 run rc_field_nav save_prior_map.sh mymap        # 自定义名字
#      RC_MAPS_DIR=/tmp/m ros2 run rc_field_nav save_prior_map.sh
#      其余参数原样透传给 map3d_to_costmap.py（如 --min-pts 2 / --max-slope 20）
# ============================================================================
set +u

NAME="prior"
if [ $# -gt 0 ] && [ "${1#-}" = "${1}" ]; then
    NAME="$1"; shift
fi
EXTRA_ARGS=("$@")

PKG_PREFIX="$(ros2 pkg prefix rc_field_nav)"
LIB="${PKG_PREFIX}/lib/rc_field_nav"
MAPS="${RC_MAPS_DIR:-${PKG_PREFIX}/share/rc_field_nav/maps}"
TRAJ_SRC="${RC_TRAJ_FILE:-$HOME/rc_field_traj.txt}"
mkdir -p "${MAPS}"

echo "=============================================================="
echo "[1/4] 触发点云累积器落盘（/save_prior_map）"
rm -f "${MAPS}/accum.pcd"
if ros2 topic pub --once /save_prior_map std_msgs/msg/Empty {} >/dev/null 2>&1; then
    for _ in $(seq 1 30); do
        [ -s "${MAPS}/accum.pcd" ] && break
        sleep 1
    done
fi
if [ -s "${MAPS}/accum.pcd" ]; then
    cp "${MAPS}/accum.pcd" "${MAPS}/${NAME}_map.pcd"
    echo "      稠密点云 -> ${MAPS}/${NAME}_map.pcd ($(du -h "${MAPS}/${NAME}_map.pcd" | cut -f1))"
else
    echo "      [警告] 没拿到累积点云 —— 这次建图没有起的累积器（比如你直接用 nav 模式？）"
    echo "             退回抓 /Laser_map 快照，密度会低很多（墙会变虚线）"
    python3 "${LIB}/save_pointcloud_pcd.py" -t /Laser_map -o "${MAPS}/${NAME}_map.pcd" --timeout 30 || exit 1
fi

echo "[2/4] 轨迹 -> ${MAPS}/${NAME}_traj.txt"
if [ -s "${TRAJ_SRC}" ]; then
    cp "${TRAJ_SRC}" "${MAPS}/${NAME}_traj.txt"
    echo "      已复制 ${TRAJ_SRC} ($(wc -l < "${MAPS}/${NAME}_traj.txt") 行)"
else
    echo "      [警告] 找不到轨迹文件 ${TRAJ_SRC}；将只用点云外包围盒估算地面"
fi

echo "[3/4] 点云密度自检（决定 2D 图能不能信）"
python3 - "${MAPS}/${NAME}_map.pcd" <<'PY'
import sys, numpy as np
p = sys.argv[1]
raw = open(p, 'rb').read()
hdr = raw.split(b'DATA binary\n')[0].decode('ascii', 'replace') if b'DATA binary\n' in raw else raw[:400].decode('ascii','replace')
import re
m = re.search(r'FIELDS\s+(.*)', hdr); f = (m.group(1).split() if m else [])
m2 = re.search(r'POINTS\s+(\d+)', hdr); n = int(m2.group(1)) if m2 else 0
body = raw.split(b'DATA binary\n', 1)[1] if b'DATA binary\n' in raw else b''
if not body or n == 0:
    print('      (读不了二进制体，跳过)'); sys.exit(0)
step = len(f) * 4
a = np.frombuffer(body[:n*step], dtype=np.float32).reshape(-1, len(f))
xyz = a[:, [f.index(c) for c in ('x','y','z')]] if all(c in f for c in ('x','y','z')) else a[:, :3]
res = 0.05
ix = ((xyz[:,0]-xyz[:,0].min())/res).astype(np.int64)
iy = ((xyz[:,1]-xyz[:,1].min())/res).astype(np.int64)
cnt = np.bincount(iy*(ix.max()+1)+ix)
cnt = cnt[cnt>0]
print('      点数 %d；5 cm 格子 %d 个；每格点数 中位 %d / 均值 %.1f；只有 1 个点的格 %.0f%%'
      % (n, len(cnt), int(np.median(cnt)), cnt.mean(), 100*(cnt<=1).mean()))
if np.median(cnt) < 3:
    print('      [警告] 每格中位点数 <3 —— 数据太稀，2D 图会出现"虚线墙/空地出点"；'
          '请确认这次建图用的是 ./sim_map.sh（带累积器）')
else:
    print('      [OK] 密度够做 5 cm 2D 图（墙应为实线）')
PY

echo "[4/4] 3D 点云 -> 2D 代价地图（方案A/甲）"
ARGS=(--cloud "${MAPS}/${NAME}_map.pcd" -o "${MAPS}/${NAME}_costmap"
      --imu-height "${RC_IMU_HEIGHT:-0.30}" --h-min 0.15 --h-max 2.0 --resolution 0.05)
[ -s "${MAPS}/${NAME}_traj.txt" ] && ARGS+=(--trajectory "${MAPS}/${NAME}_traj.txt")
ARGS+=("${EXTRA_ARGS[@]}")
python3 "${LIB}/map3d_to_costmap.py" "${ARGS[@]}" || exit 1

python3 "${LIB}/map_sanity_check.py" "${MAPS}/${NAME}_costmap.yaml" || true

echo "--------------------------------------------------------------"
ls -la "${MAPS}" | grep "${NAME}" || true
echo "下一步（导航）："
if [ "${NAME}" = "prior" ]; then
    echo "  ./sim_nav.sh                       # 默认就用 prior_map.pcd / prior_costmap.yaml"
else
    echo "  ./sim_nav.sh prior_map:=${MAPS}/${NAME}_map.pcd map:=${MAPS}/${NAME}_costmap.yaml"
fi
