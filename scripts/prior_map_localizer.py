#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""先验地图定位 / 重定位（真实雷达版）——把"当前点云 vs 先验地图"变成 Nav2 能用的 map->odom。

算法来源
--------
* **重定位（全局、无初值）**：借用本项目 rc_field_nav/relocalize_search.py 的思路 ——
  用"体素重叠度"当代价，在 (x, y, yaw) 上做**多分辨率穷举**，先 0.30 m/10° 扫全图，
  再 0.10 m/4°、0.02 m/1° 精化。为什么不用现成的 TEASER++/FPFH 链路：
  方厅+中间方块这种高度对称的室内场景，FPFH 特征稀疏时容易掉进 2 m 外的局部极小；
  穷举在室内尺度（几十米）完全跑得动，而且**保证全局最优**。
* **定位（连续跟踪）**：拿到 bootstrap 之后只在当前估计附近开窗搜索
  （±1.2 m / ±12°），单次 0.5 s 量级；配合门限、平滑、死区，把 LIO 的漂移持续拉回来。

和仿真版的区别（都是真机才会遇到的）
------------------------------------
1. 数据源默认是 FAST-LIVO2 的话题：`/cloud_registered`（odom 系当前帧）+ `/aft_mapped_to_init`；
2. 帧名默认 `map / odom / base_link`，可用参数改；
3. 地面高度不再手填：从建图阶段写出的 `prior_map.meta.json` 里读 `ground_z`
   （没有 meta 时用 `--imu-height` 推算：odom 原点就是开机时的 IMU 位置，所以
   地面 z = -IMU 离地高度）；
4. 搜索范围默认按先验地图的点云包围盒来，并且会**先把当前帧里落在搜索范围外的点剔掉**，
   避免真机上偶发的远处噪点把代价函数带偏；
5. 累积用的是 odom 系点云（`/cloud_registered` 本身就是），所以车在动的时候叠帧不会糊；
6. 搜索跑在独立线程并 `nice(10)`，不会饿死 TF 广播和 ROS 回调。

输出
----
* TF  `map -> odom`（Nav2 的 map 系定位就靠它）
* `/localization_pose`      机器人(base_link)在 map 系的位姿，给 RViz / 其它节点看
* `/relocalization_result`  雷达在 map 系的位姿；**FAST-LIVO2 的 locate_in_prior_map
                            模式订阅的就是这个话题**，所以这条链路可以直接驱动
                            FAST-LIVO2 的先验地图连续跟踪
* `/reloc_score`            Float64 重叠度，调参/诊断用

    python3 prior_map_localizer.py --prior-map maps/prior_map.pcd
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from std_msgs.msg import Float64, String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformBroadcaster
import sensor_msgs_py.point_cloud2 as pc2


# ----------------------------------------------------------------------------
# 启动前自检：先验点云在不在
# ----------------------------------------------------------------------------
def check_prior_map(path):
    """先验点云存在性检查。

    为什么要单独查：直接 open() 会抛 FileNotFoundError 栈，在 launch 日志里
    只留下"process has died"一行，看不出是"地图名不对"（实测踩过：
    存图用了自定义名 map1，导航仍按默认 prior_map.pcd 找，定位器 0.8 s 就死）。
    """
    if os.path.isfile(path):
        return
    d = os.path.dirname(os.path.abspath(path))
    cands = []
    if os.path.isdir(d):
        cands = sorted(f for f in os.listdir(d) if f.endswith('.pcd'))
    print('[错误] 找不到先验点云：%s' % path)
    if cands:
        print('       该目录下现有 .pcd：%s' % ', '.join(cands))
        print('       直接用它试试：--prior-map %s' % os.path.join(d, cands[0]))
        print('       导航时也可以： ros2 launch rc_field_nav nav_prior_map.launch.py \\')
        print('           prior_map:=%s \\' % os.path.join(d, cands[0]))
        print('           map:=%s' % os.path.join(
            d, cands[0].replace('_map.pcd', '_costmap.yaml')))
    else:
        print('       该目录下没有任何 .pcd —— 还没存过图？')
        print('       仿真： ./sim_map.sh 建图 -> ./sim_save_map.sh 存图')
        print('       真机： ./real_map.sh 建图 -> ./real_save_map.sh 存图')
    print('       ★ 存图用的名字与导航要用的名字必须一致；'
          '2D 图与 3D 点云要来自同一次存图。')
    sys.exit(2)


# ----------------------------------------------------------------------------
# PCD 读取（binary/ascii，字段名大小写都吃）
# ----------------------------------------------------------------------------
def read_pcd(path):
    with open(path, 'rb') as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError('PCD header truncated: %s' % path)
            txt = line.decode('ascii', 'replace').strip()
            if not txt or txt.startswith('#'):
                continue
            p = txt.split()
            header[p[0].upper()] = p[1:]
            if p[0].upper() == 'DATA':
                break
        fields = [s.lower() for s in header.get('FIELDS', [])]
        sizes = [int(s) for s in header.get('SIZE', [])]
        types = [s.upper() for s in header.get('TYPE', [])]
        npts = int(header.get('POINTS', ['0'])[0])
        dtype = header['DATA'][0].lower()
        tmap = {'F': 'f', 'U': 'u', 'I': 'i'}
        dt = np.dtype([(fields[i], '<' + tmap.get(types[i], 'f') + str(sizes[i]))
                       for i in range(len(fields))])
        if dtype == 'binary':
            arr = np.frombuffer(f.read(dt.itemsize * npts), dtype=dt, count=npts)
        elif dtype == 'ascii':
            arr = np.loadtxt(f, dtype=dt, max_rows=npts)
        else:
            raise RuntimeError('unsupported PCD DATA type: %s' % dtype)
    names = arr.dtype.names or ()

    def col(w):
        for n in names:
            if n.lower() == w:
                return arr[n].astype(np.float64)
        raise RuntimeError('missing field %s in %s' % (w, names))
    pts = np.stack([col('x'), col('y'), col('z')], axis=1)
    return pts[np.isfinite(pts).all(axis=1)]


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def voxel_dedup(arr, res):
    if arr.shape[0] == 0 or res <= 0:
        return arr
    keys = np.floor(arr / res).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return arr[np.sort(idx)]


# ----------------------------------------------------------------------------
# 体素重叠度代价 + 搜索
# ----------------------------------------------------------------------------
def make_voxel_map(pts, res, margin=3.0):
    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    dims = np.maximum(np.ceil((hi - lo) / res).astype(int) + 1, 1)
    occ = np.zeros(tuple(dims), dtype=bool)
    idx = np.clip(np.floor((pts - lo) / res).astype(np.int64), 0, dims - 1)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    dil = occ.copy()
    for a in (-1, 0, 1):                       # 1 个体素容差
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                if a == b == c == 0:
                    continue
                dil |= np.roll(np.roll(np.roll(occ, a, 0), b, 1), c, 2)
    return dil, lo, res


def score_pose(src, dil, lo, res, dx, dy, yaw_deg, zoff):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    x = c * src[:, 0] - s * src[:, 1] + dx
    y = s * src[:, 0] + c * src[:, 1] + dy
    z = src[:, 2] + zoff
    idx = np.floor((np.column_stack([x, y, z]) - lo) / res).astype(np.int64)
    d = np.array(dil.shape)
    inside = ((idx >= 0) & (idx < d)).all(axis=1)
    if not inside.any():
        return 0.0
    idx = np.clip(idx, 0, d - 1)
    return float(dil[idx[:, 0], idx[:, 1], idx[:, 2]].mean())


def search_full(src, dil, lo, res, zoff, bounds, subsample=800):
    x0, x1, y0, y1 = bounds
    coarse = src[::max(1, len(src) // subsample)]
    best = (-1.0, 0.0, 0.0, 0.0)
    for yaw in range(-180, 180, 10):
        for dx in np.arange(x0, x1, 0.30):
            for dy in np.arange(y0, y1, 0.30):
                v = score_pose(coarse, dil, lo, res, dx, dy, yaw, zoff)
                if v > best[0]:
                    best = (v, dx, dy, float(yaw))
    v0, bx, by, byaw = best
    for step, yr in ((0.10, 4.0), (0.02, 1.0)):
        best = (-1.0, bx, by, byaw)
        for yaw in np.arange(byaw - 2 * yr, byaw + 2 * yr + 1e-9, yr):
            for dx in np.arange(bx - 4 * step, bx + 4 * step + 1e-9, step):
                for dy in np.arange(by - 4 * step, by + 4 * step + 1e-9, step):
                    v = score_pose(src, dil, lo, res, dx, dy, yaw, zoff)
                    if v > best[0]:
                        best = (v, dx, dy, float(yaw))
        v0, bx, by, byaw = best
    return best


def argv_fix(argv, opts=('--initial-pose', '--bounds', '--ground-z', '--z-offset')):
    """把 "--opt -4.3,1.5,0" 这类参数自动拼成 "--opt=-4.3,1.5,0"。

    argparse 看到以 - 开头的 token 会当成选项名（除非它长得像一个负数），
    所以坐标这种 "x,y,z" 字符串一旦以负号开头就会报 expected one argument。
    本工程在 --transform / --initial-pose 上各踩过一次。最保险的写法始终是
    "--opt=值"，这个函数只是让人少踩一次。
    """
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in opts and i + 1 < len(argv) and argv[i + 1].startswith('-'):
            out.append('%s=%s' % (a, argv[i + 1]))
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def symmetry_margin(src, dil, lo, res, zoff, bounds, best, radius=2.0, yaw_sep=30.0,
                    subsample=500):
    """全图搜索的"歧义度"体检：最好的解之外，还有没有别的**位置上明显不同**的
    解分数也很高。

    为什么需要：场地/场馆常常是**中心对称**的（本工程的 ROBOCON 场就是：红蓝两半
    的 U 形走廊一模一样）。这种场景下"转 180°"的镜像解分数只比正解低一点点，
    纯穷举打分无法区分，可能锁到镜像解上 —— 表现是"定位分数挺高，但地图里车的
    位置整体跑到了场地另一半"。

    返回值 = 最佳分数 / 次佳分数（次佳解取"离最佳解 2 m 或 30° 以外"的最优）。
    1.0 表示完全对称（无法区分），越大越安全。只跑粗网格，开销约等于全图搜索的第一步。
    """
    bv, bx, by, byaw = best
    x0, x1, y0, y1 = bounds
    coarse = src[::max(1, len(src) // subsample)]
    second = -1.0
    for yaw in range(-180, 180, 10):
        for dx in np.arange(x0, x1, 0.30):
            for dy in np.arange(y0, y1, 0.30):
                if math.hypot(dx - bx, dy - by) < radius and \
                        abs(wrap(math.radians(yaw - byaw))) < math.radians(yaw_sep):
                    continue
                v = score_pose(coarse, dil, lo, res, dx, dy, yaw, zoff)
                if v > second:
                    second = v
    if second <= 0.0:
        return 99.0
    return bv / second


def search_window(src, dil, lo, res, zoff, cx, cy, cyaw, span_xy, span_yaw,
                  step_xy=0.10, step_yaw=3.0, subsample=500):
    coarse = src[::max(1, len(src) // subsample)]
    best = (-1.0, cx, cy, cyaw)
    for yaw in np.arange(cyaw - span_yaw, cyaw + span_yaw + 1e-9, step_yaw):
        for dx in np.arange(cx - span_xy, cx + span_xy + 1e-9, step_xy):
            for dy in np.arange(cy - span_xy, cy + span_xy + 1e-9, step_xy):
                v = score_pose(coarse, dil, lo, res, dx, dy, yaw, zoff)
                if v > best[0]:
                    best = (v, dx, dy, float(yaw))
    v0, bx, by, byaw = best
    best = (-1.0, bx, by, byaw)
    for yaw in np.arange(byaw - 2.0, byaw + 2.0 + 1e-9, 1.0):
        for dx in np.arange(bx - 0.16, bx + 0.16 + 1e-9, 0.04):
            for dy in np.arange(by - 0.16, by + 0.16 + 1e-9, 0.04):
                v = score_pose(src, dil, lo, res, dx, dy, yaw, zoff)
                if v > best[0]:
                    best = (v, dx, dy, float(yaw))
    return best


def polish_z(src, dil, lo, res, x, y, yaw, zc, span=0.20, step=0.025):
    best = (-1.0, zc)
    for zo in np.arange(zc - span, zc + span + 1e-9, step):
        v = score_pose(src, dil, lo, res, x, y, yaw, zo)
        if v > best[0]:
            best = (v, zo)
    return best


# ----------------------------------------------------------------------------
class PriorMapLocalizer(Node):
    def __init__(self, a):
        super().__init__('prior_map_localizer')
        self.a = a
        self.lock = threading.RLock()
        self.buf = []
        self.odom = None
        self.est = None
        self.zoff = a.z_offset
        self.score = 0.0
        self.n_ok = 0
        self.n_skip = 0
        self.n_rej = 0
        self.reject_streak = []
        self.busy = False
        self.mode = 'idle'
        self.force_full = False       # 连续低分后强制做一次全图重定位
        self.lost = 0                 # 连续低分计数
        self.margin = 99.0            # 全图搜索的歧义度（best/second，越大越安全）
        self.last_search = 0.0
        self._last_tf_ns = None

        # ---- 先验地图 ----
        pts = read_pcd(a.prior_map)
        gz = a.ground_z
        if gz is None and a.meta and os.path.isfile(a.meta):
            try:
                meta = json.load(open(a.meta))
                gz = float(meta.get('ground_z', None)) if meta.get('ground_z') is not None else None
                if gz is not None:
                    self.get_logger().info('从 %s 读到 ground_z = %.3f' % (a.meta, gz))
            except Exception as exc:                       # noqa: BLE001
                self.get_logger().warn('读 meta 失败：%s' % exc)
        if gz is None:
            gz = -abs(a.imu_height)
            self.get_logger().warn(
                '没有 meta/ground_z，按 IMU 离地高度 %.2f m 推算 ground_z = %.3f'
                '（不对的话请显式给 --ground-z）' % (a.imu_height, gz))
        self.ground_z = gz
        zrel = pts[:, 2] - gz
        band = pts[(zrel > a.h_min) & (zrel < a.h_max)]
        if len(band) < 200:
            self.get_logger().warn('障碍带内只有 %d 点，改用全部点' % len(band))
            band = pts
        self.dil, self.lo, self.res = make_voxel_map(band, a.voxel)
        x0, x1 = band[:, 0].min() - self.res, band[:, 0].max() + self.res
        y0, y1 = band[:, 1].min() - self.res, band[:, 1].max() + self.res
        if a.bounds:
            x0, x1, y0, y1 = [float(v) for v in a.bounds.split(',')]
        self.bounds = (x0, x1, y0, y1)
        self.get_logger().info(
            '先验地图 %s：%d 点（障碍带 %d，ground_z=%.3f），体素 %s @ %.2f m，'
            '搜索范围 x[%.2f,%.2f] y[%.2f,%.2f]'
            % (a.prior_map, len(pts), len(band), gz, self.dil.shape, self.res,
               x0, x1, y0, y1))

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(PointCloud2, a.cloud_topic, self.on_cloud, qos)
        self.create_subscription(Odometry, a.odom_topic, self.on_odom, 10)
        self.pub_reloc = self.create_publisher(PoseWithCovarianceStamped, a.reloc_topic, 10)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, '/localization_pose', 10)
        self.pub_score = self.create_publisher(Float64, '/reloc_score', 10)
        self.pub_stat = self.create_publisher(String, '/reloc_status', 5)
        # --no-tf：只发 /relocalization_result 和 /localization_pose，不发 map->odom。
        # 用于 FAST-LIVO2 locate_in_prior_map 模式 —— 那条路径下 TF map->odom 由
        # reloc_pose_to_tf.py 负责，两个都发会打两套 TF 互相打架。
        self.tf_br = None if a.no_tf else TransformBroadcaster(self)
        if not a.no_tf:
            self.create_timer(1.0 / a.tf_rate, self.publish_tf)
        self.create_timer(1.0 / a.pose_rate, self.publish_pose)
        self.create_timer(0.5, self.maybe_search)

        if a.initial_pose:
            v = [float(t) for t in a.initial_pose.split(',')]
            self.est = (v[0], v[1], math.radians(v[2] if len(v) > 2 else 0.0))
            self.mode = 'SEEDED'
            self.get_logger().info('用 --initial-pose 直接给定 map->odom = (%.3f, %.3f, %.2f deg)，'
                                   '跳过全图搜索' % (v[0], v[1], v[2] if len(v) > 2 else 0.0))
        self.get_logger().info('等待 %s（odom 系点云）与 %s；输出 TF %s->%s'
                               % (a.cloud_topic, a.odom_topic, a.map_frame, a.odom_frame))

    # ---------------- 回调 ----------------
    def on_odom(self, m):
        p = m.pose.pose
        with self.lock:
            self.odom = (p.position.x, p.position.y, p.position.z, yaw_of(p.orientation))

    def on_cloud(self, m):
        arr = np.array([[q[0], q[1], q[2]] for q in
                        pc2.read_points(m, field_names=('x', 'y', 'z'), skip_nans=True)],
                       dtype=np.float64)
        if arr.shape[0] < 50:
            return
        arr = arr[np.isfinite(arr).all(axis=1)]
        arr = voxel_dedup(arr, self.a.acc_voxel)
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with self.lock:
            self.buf.append((t, arr))
            while len(self.buf) > self.a.window or (
                    self.buf and t - self.buf[0][0] > self.a.window_sec):
                self.buf.pop(0)

    # ---------------- 主流程 ----------------
    def maybe_search(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.busy or (now - self.last_search) < self.a.period:
            return
        with self.lock:
            if self.odom is None or len(self.buf) < 5:
                return
            src = np.concatenate([b[1] for b in self.buf], axis=0)
            est = self.est
        if src.shape[0] < self.a.min_points:
            return
        # 剔掉明显落在搜索范围外的点（真机偶发远处噪点）
        x0, x1, y0, y1 = self.bounds
        m = ((src[:, 0] > x0 - 5) & (src[:, 0] < x1 + 5) &
             (src[:, 1] > y0 - 5) & (src[:, 1] < y1 + 5))
        if m.sum() < self.a.min_points:
            return
        src = src[m]
        self.busy = True
        self.last_search = now
        threading.Thread(target=self._worker, args=(src, est), daemon=True).start()

    def _worker(self, src, est):
        try:
            os.nice(10)           # 搜索是 CPU 密集型的，别饿死 ROS 回调和 Nav2
        except Exception:         # noqa: BLE001
            pass
        try:
            t0 = time.time()
            with self.lock:
                force = self.force_full                 # 连续低分后强制全图重定位
            if est is None or force:
                v, x, y, yaw = search_full(src, self.dil, self.lo, self.res,
                                           self.zoff, self.bounds)
                mode = 'FULL' if est is None else 'FULL(REINIT)'
                # 全图搜索额外算一次"歧义度"：对称场地里镜像解可能同样高分
                margin = symmetry_margin(src, self.dil, self.lo, self.res,
                                         self.zoff, self.bounds, (v, x, y, yaw))
            else:
                v, x, y, yaw = search_window(src, self.dil, self.lo, self.res, self.zoff,
                                             est[0], est[1], math.degrees(est[2]),
                                             self.a.track_xy, self.a.track_yaw)
                mode = 'WIN'
                margin = 0.0
            v, zo = polish_z(src, self.dil, self.lo, self.res, x, y, yaw, self.zoff)
            self.accept(v, x, y, math.radians(yaw), zo, mode, time.time() - t0,
                        src.shape[0], force, margin)
        except Exception as exc:                       # noqa: BLE001
            self.get_logger().error('重定位线程异常：%s' % exc)
        finally:
            self.busy = False

    def accept(self, v, x, y, yaw, zoff, mode, dt, npts, force=False, margin=0.0):
        with self.lock:
            est = self.est
            self.score = v
            if margin:
                self.margin = margin
            if v < self.a.min_score:
                self.n_rej += 1
                self.mode = 'LOW_SCORE(%s)' % mode
                self.get_logger().warn('%s 重叠度 %.3f < %.2f，丢弃（%d 点, %.1fs）'
                                       % (mode, v, self.a.min_score, npts, dt))
                # ★ 低分不能只靠窗口搜索硬撑：窗口只有 ±track_xy，一旦 LIO 漂出去
                #   （或被搬走）就永远找不到。连续若干次低分就强制做一次全图搜索，
                #   全图搜索拿到合格分数后直接接管（不再走跳变门限，否则永远被挡）。
                if force:
                    self.force_full = False          # 全图也没找到，下次再说
                else:
                    self.lost += 1
                    if self.lost >= self.a.reinit_after:
                        self.force_full = True
                        self.lost = 0
                        self.mode = 'REINIT_SEARCH'
                        self.get_logger().warn(
                            '连续 %d 次低分 -> 触发全图重定位（map->odom 交给下一次全图结果）'
                            % self.a.reinit_after)
                self._maybe_reinit_locked()
                return
            if est is None or force:
                first = est is None
                self.est = (x, y, yaw)
                self.zoff = zoff
                self.n_ok += 1
                self.lost = 0
                self.reject_streak = []
                self.force_full = False
                self.mode = 'LOCKED' if first else 'RELOCKED(%s)' % mode
                self.get_logger().info(
                    '%s map->odom：x=%.3f y=%.3f yaw=%.2f deg (zoff=%.3f) '
                    '重叠度 %.3f  歧义度 %.2f  %d 点  %.1fs'
                    % ('★ 初次锁定' if first else '★ 全图重新锁定', x, y,
                       math.degrees(yaw), zoff, v, self.margin, npts, dt))
                if self.margin < 1.05:
                    self.get_logger().warn(
                        '★ 场景近似中心对称：次佳解只比最佳解低 %.1f%%（歧义度 %.2f），'
                        '当前锁定可能是镜像解！建议给 --initial-pose 指定大致初始位姿，'
                        '或确认场地不是对称的' % (100.0 * (1.0 - 1.0 / max(self.margin, 1e-6)),
                                            self.margin))
                self._publish_locked()
                return
            dx, dy = x - est[0], y - est[1]
            dyaw = math.degrees(wrap(yaw - est[2]))
            jump = math.hypot(dx, dy)
            if jump > self.a.max_jump or abs(dyaw) > self.a.max_yaw_jump:
                self.n_rej += 1
                self.reject_streak.append((x, y, yaw))
                self.reject_streak = self.reject_streak[-self.a.reinit_after:]
                self.mode = 'GATED(%s)' % mode
                self.get_logger().warn('候选跳变 %.2f m / %.1f deg 超门限，丢弃'
                                       '（重叠度 %.3f，连续 %d 次）'
                                       % (jump, dyaw, v, len(self.reject_streak)))
                self._maybe_reinit_locked()
                return
            if jump <= self.a.deadband and abs(dyaw) <= self.a.deadband_yaw:
                self.n_skip += 1
                self.mode = 'TRACK(deadband)'
                return
            a = self.a.alpha
            self.est = (est[0] + a * dx, est[1] + a * dy, est[2] + a * math.radians(dyaw))
            self.zoff = zoff
            self.n_ok += 1
            self.reject_streak = []
            self.lost = 0
            self.mode = 'TRACK(%s)' % mode
            self.get_logger().info('跟踪更新：残差 %.3f m / %.1f deg -> map->odom '
                                   'x=%.3f y=%.3f yaw=%.2f deg 重叠度 %.3f  %.1fs'
                                   % (jump, dyaw, self.est[0], self.est[1],
                                      math.degrees(self.est[2]), v, dt))
            self._publish_locked()

    def _maybe_reinit_locked(self):
        rs = self.reject_streak
        if len(rs) < self.a.reinit_after:
            return
        xs = np.array([r[0] for r in rs])
        ys = np.array([r[1] for r in rs])
        yaws = np.array([r[2] for r in rs])
        if xs.ptp() > self.a.max_jump or ys.ptp() > self.a.max_jump:
            return
        if np.abs([math.degrees(wrap(yy - yaws[0])) for yy in yaws]).max() > self.a.max_yaw_jump:
            return
        self.est = (float(xs.mean()), float(ys.mean()), float(yaws.mean()))
        self.reject_streak = []
        self.mode = 'REINIT'
        self.get_logger().warn('连续 %d 次一致的新解 -> 重新锁定 map->odom '
                               'x=%.3f y=%.3f yaw=%.2f deg'
                               % (len(rs), self.est[0], self.est[1], math.degrees(self.est[2])))

    # ---------------- 输出 ----------------
    def _publish_locked(self):
        now = self.get_clock().now().to_msg()
        with self.lock:
            est, odom = self.est, self.odom
        if est is None:
            return
        if odom is not None:
            c, s = math.cos(est[2]), math.sin(est[2])
            bx = c * odom[0] - s * odom[1] + est[0]
            by = s * odom[0] + c * odom[1] + est[1]
            byaw = wrap(est[2] + odom[3])
            m = PoseWithCovarianceStamped()
            m.header.stamp = now
            m.header.frame_id = self.a.map_frame
            m.pose.pose.position.x = float(bx)
            m.pose.pose.position.y = float(by)
            m.pose.pose.orientation.z = math.sin(byaw / 2.0)
            m.pose.pose.orientation.w = math.cos(byaw / 2.0)
            self.pub_pose.publish(m)
            r = PoseWithCovarianceStamped()
            r.header.stamp = now
            r.header.frame_id = self.a.map_frame
            r.pose.pose.position.x = float(bx)
            r.pose.pose.position.y = float(by)
            r.pose.pose.position.z = float(self.a.z_lidar)
            r.pose.pose.orientation.z = math.sin(byaw / 2.0)
            r.pose.pose.orientation.w = math.cos(byaw / 2.0)
            self.pub_reloc.publish(r)          # FAST-LIVO2 locate_in_prior_map 就用这个
        self.pub_score.publish(Float64(data=float(self.score)))
        self.pub_stat.publish(String(data='mode=%s score=%.3f 歧义度=%.2f ok=%d skip=%d rej=%d'
                                     % (self.mode, self.score, self.margin,
                                        self.n_ok, self.n_skip, self.n_rej)))

    def publish_pose(self):
        self._publish_locked()

    def publish_tf(self):
        if self.tf_br is None:
            return
        with self.lock:
            est = self.est
        if est is None:
            return
        now = self.get_clock().now()
        # tf2 只接受时间戳严格递增的同一帧对；仿真/真机的时钟都是分片更新的，
        # 两次回调可能读到同一时刻，直接发会刷屏 TF_OLD_DATA
        if self._last_tf_ns is not None and now.nanoseconds <= self._last_tf_ns:
            return
        self._last_tf_ns = now.nanoseconds
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = self.a.map_frame
        t.child_frame_id = self.a.odom_frame
        t.transform.translation.x = float(est[0])
        t.transform.translation.y = float(est[1])
        t.transform.translation.z = float(self.a.tf_z)
        t.transform.rotation.z = math.sin(est[2] / 2.0)
        t.transform.rotation.w = math.cos(est[2] / 2.0)
        self.tf_br.sendTransform(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior-map', required=True, help='先验 3D 地图 .pcd（map 系）')
    ap.add_argument('--meta', default='',
                    help='建图阶段写出的 prior_map.meta.json（读 ground_z 等）')
    ap.add_argument('--cloud-topic', default='/cloud_registered',
                    help='odom 系的当前帧点云（FAST-LIVO2 的 /cloud_registered）')
    ap.add_argument('--odom-topic', default='/aft_mapped_to_init',
                    help='FAST-LIVO2 的里程计话题（用 FAST-LIO 时是 /Odometry）')
    ap.add_argument('--reloc-topic', default='/relocalization_result')
    ap.add_argument('--map-frame', default='map')
    ap.add_argument('--odom-frame', default='odom')
    ap.add_argument('--tf-z', type=float, default=0.0)
    ap.add_argument('--z-lidar', type=float, default=0.0,
                    help='雷达在 map 系的高度（只影响 /relocalization_result 的 z）')
    ap.add_argument('--initial-pose', default='',
                    help='可选 "x,y,yaw_deg"：已知大致位姿时直接给定，跳过全图搜索')
    # 先验地图 / 搜索
    ap.add_argument('--ground-z', type=float, default=None,
                    help='地面在 map 系的 z。不给则从 meta 读，再不行用 -imu_height')
    ap.add_argument('--imu-height', type=float, default=0.30,
                    help='IMU/雷达离地高度（odom 原点就是开机时 IMU 的位置）')
    ap.add_argument('--h-min', type=float, default=0.15)
    ap.add_argument('--h-max', type=float, default=2.0)
    ap.add_argument('--voxel', type=float, default=0.10)
    ap.add_argument('--z-offset', type=float, default=0.0)
    ap.add_argument('--bounds', default='')
    ap.add_argument('--min-score', type=float, default=0.55)
    # 累积
    ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--window-sec', type=float, default=5.0)
    ap.add_argument('--acc-voxel', type=float, default=0.03)
    ap.add_argument('--min-points', type=int, default=800)
    ap.add_argument('--period', type=float, default=3.0, help='重定位周期（秒）')
    ap.add_argument('--tf-rate', type=float, default=30.0)
    ap.add_argument('--no-tf', action='store_true',
                    help='不发 map->odom TF；给 FAST-LIVO2 locate_in_prior_map 当「喂位姿」的实例用')
    ap.add_argument('--pose-rate', type=float, default=5.0)
    # 跟踪 / 门限
    ap.add_argument('--track-xy', type=float, default=1.2)
    ap.add_argument('--track-yaw', type=float, default=12.0)
    ap.add_argument('--max-jump', type=float, default=1.0)
    ap.add_argument('--max-yaw-jump', type=float, default=15.0)
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--deadband', type=float, default=0.06)
    ap.add_argument('--deadband-yaw', type=float, default=1.5)
    ap.add_argument('--reinit-after', type=int, default=3)
    args, _ = ap.parse_known_args(argv_fix(sys.argv[1:]))

    check_prior_map(args.prior_map)

    rclpy.init()
    n = PriorMapLocalizer(args)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.get_logger().info('统计：接受 %d（死区内跳过 %d），拒绝 %d，最终模式 %s'
                            % (n.n_ok, n.n_skip, n.n_rej, n.mode))
        n.destroy_node()
        if rclpy.ok():                      # SIGTERM 时 rclpy 可能已经自己关过了
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
