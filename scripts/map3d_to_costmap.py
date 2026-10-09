#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3D 点云地图 -> 2D 占据栅格（Nav2 map_server 直接能用），真机版。

流程
----
1. 读 .pcd（可多片，每片可自带 "路径,tx,ty,yaw_deg" 变换，便于多趟建图合并）；
2. 定地面高度 ground_z：
   * 优先用 --ground-z；
   * 否则用 --trajectory 的**开头**几个点：LIO 的 odom 原点就是开机时的 IMU 位置，
     所以 地面 z = z(轨迹起点) - IMU离地高度（对同一套 LIO，这个是正确的绝对基准，
     且不受"跑完之后原地停放导致 z 漂移"影响）；
   * 再不行才退回 z 的 2% 分位数。
   **不要用 z 直方图众数** —— 竖直墙面点数远多于地面点，众数会落在墙面中段。
3. 占据：z 落在 [ground_z+h_min, ground_z+h_max] 的点所在格子；
4. 空闲：从轨迹每个采样点做射线投射（撞到占据格之前的格子算空闲），
   这样"没有观测到的地方"保持未知，不会被误当成可通行；
5. 形态学闭/开去毛刺，写 .pgm/.yaml，并顺手写一份 .meta.json
   （记录 ground_z / imu_height / 分辨率等，供第二阶段的定位节点直接读）。

    python3 map3d_to_costmap.py maps/prior_map.pcd \
        --trajectory maps/prior_traj.txt --imu-height 0.30 \
        --out maps/prior_costmap
"""
import argparse
import json
import math
import os
import sys

import numpy as np


def argv_fix(argv, opts=('--transform', '--ground-z')):
    """把 "--opt -4.3,1.5,0" 这类参数自动拼成 "--opt=-4.3,1.5,0"。

    argparse 看到以 - 开头的 token 会当成选项名（除非它长得像一个负数），
    所以坐标这种 "x,y,z" 字符串一旦以负号开头就会报 expected one argument。
    最保险的写法始终是 "--opt=值"，这个函数只是让人少踩一次。
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


def read_traj(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            v = [float(t) for t in line.replace(',', ' ').split()]
        except ValueError:
            continue
        if len(v) >= 3:
            rows.append(v[:3])
    return np.array(rows) if rows else np.zeros((0, 3))


def tf2d(arr, tx, ty, yaw_deg):
    if arr.size == 0 or (tx == 0 and ty == 0 and yaw_deg == 0):
        return arr
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    xy = arr[:, :2].copy()
    arr[:, 0] = c * xy[:, 0] - s * xy[:, 1] + tx
    arr[:, 1] = s * xy[:, 0] + c * xy[:, 1] + ty
    return arr


def cast_free(occ, traj, origin_xy, res, max_range, min_range, step_deg, chunk=48):
    """从每个轨迹点向四周打射线，撞到占据格之前的格子标为空闲。

    一次性构造 (轨迹点 x 方向 x 步数) 的索引矩阵很吃内存
    （几万条射线 x 400 步 x 4 字节 x 3 个数组就上百 MB），所以按轨迹点分批处理。
    """
    h, w = occ.shape
    free = np.zeros_like(occ)
    angs = np.deg2rad(np.arange(0.0, 360.0, step_deg))
    dirs = np.stack([np.cos(angs), np.sin(angs)], axis=1)
    steps = np.arange(min_range, max_range, res)
    n = steps.size
    if n == 0 or len(traj) == 0:
        return free
    for c0 in range(0, len(traj), chunk):
        pts = traj[c0:c0 + chunk]
        ox = (pts[:, 0] - origin_xy[0])[:, None, None]
        oy = (pts[:, 1] - origin_xy[1])[:, None, None]
        cx = np.floor((ox + dirs[None, :, 0:1] * steps[None, None, :]) / res).astype(np.int32)
        cy = np.floor((oy + dirs[None, :, 1:2] * steps[None, None, :]) / res).astype(np.int32)
        inside = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        if not inside.any():
            continue
        cxs = np.clip(cx, 0, w - 1)
        cys = np.clip(cy, 0, h - 1)
        hit = occ[cys, cxs] & inside
        first = np.where(hit.any(axis=2), hit.argmax(axis=2), n)   # 每条射线首次命中的步数
        idx = (np.arange(n)[None, None, :] < first[:, :, None]) & inside
        free[cys[idx], cxs[idx]] = True
        del cx, cy, inside, cxs, cys, hit, idx
    # 相邻射线在远处会留下 1 格的缝，补一格让空闲区连成片；
    # 占据格最后会重新盖回去，所以不会把障碍抹掉。
    fill = free.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        fill |= np.roll(np.roll(free, dy, 0), dx, 1)
    return fill


def stamp_corridor(free, traj, origin_xy, res, radius):
    """把轨迹点周围 radius 米内的格子也标成空闲。

    物理依据：车实际走到过的地方一定是可通行的。只靠射线投射的话，远处相邻
    射线之间会留下缝（1° 射线在 20 m 处间隔 0.35 m = 7 格），空闲区会碎成
    几十块（实测：15 m 的图碎成 48 块、最大块只占 1/3），Nav2 会因此找不到路。
    """
    if radius <= 0 or len(traj) == 0:
        return free, 0
    h, w = free.shape
    r = int(math.ceil(radius / res))
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
            if dy * dy + dx * dx <= r * r]
    ix = ((traj[:, 0] - origin_xy[0]) / res).astype(np.int32)
    iy = ((traj[:, 1] - origin_xy[1]) / res).astype(np.int32)
    for dy, dx in offs:
        yy = iy + dy
        xx = ix + dx
        ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        free[yy[ok], xx[ok]] = True
    return free, len(offs)


def morph(mask, times, mode):
    def dil(m):
        out = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out |= np.roll(np.roll(m, dy, 0), dx, 1)
        return out

    def ero(m):
        return ~dil(~m)
    m = mask
    for _ in range(times):
        m = ero(dil(m)) if mode == 'close' else dil(ero(m))
    return m


def despeckle(occ, res, min_area, protect=None):
    """删掉面积小于 min_area 的孤立占据块（8 邻域连通），返回 (occ, 删除格数)。

    protect: 可选 bool 栅格；只要一个连通块里含 protect 标记的格，就整块保留。
    ★ 用途：地形模式算出来的"台阶/陡坡边缘"是 1 格宽的细长护栏，面积很小，
      会被 min_area 当噪点整条删掉 —— 实测删掉后立刻出现 10 对"相邻空闲格高差
      > 0.25 m"（等于把高台边缘打开了）。这些边缘来自高度场、不是单点毛刺，
      必须保护。

    稀疏点云投到 5 cm 格子上，远处单点回波会留下孤立小格：它们既不是真障碍
    （虚警），又让地图看着毛刺。真障碍（墙/柱/围栏/台阶）连通面积都远大于阈值。
    ★ 必须在"射线投射空闲区"之前做 —— 否则这些假障碍还会挡住射线、把空闲区打碎。
    """
    if min_area <= 0:
        return occ, 0
    h, w = occ.shape
    seen = np.zeros_like(occ)
    min_cells = max(1, int(round(min_area / (res * res))))
    removed = 0
    for sy in range(h):
        for sx in range(w):
            if not occ[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and occ[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True
                            stack.append((yy, xx))
            if len(comp) < min_cells:
                keep = protect is not None and any(protect[y, x] for y, x in comp)
                if not keep:
                    for y, x in comp:
                        occ[y, x] = False
                    removed += len(comp)
    return occ, removed


# ---------------------------------------------------------------------------
# 方案 A：当地地面高度场 + 坡度/高差判据（目的是把坡道变成可通行区）
# ---------------------------------------------------------------------------
def raster_index(pts, xmin, ymin, res, w, h):
    """点 -> 栅格索引；返回 (iy, ix, z)，只保留落在地图内的点。"""
    ix = ((pts[:, 0] - xmin) / res).astype(np.int64)
    iy = ((pts[:, 1] - ymin) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    return iy[ok], ix[ok], pts[ok, 2]


def build_elevation(pts, traj, xmin, ymin, res, w, h, imu_h, seed_r, max_step, smooth):
    """建"当地地面高度场" elev(h,w)；NaN = 该格**不属于"从轨迹能走到的那张面"**。

    为什么需要它：固定高度带（相对**最低地面** 0.15~2.0 m）表达不了"坡道/高台"这种
    可通行的倾斜面/高层面 —— 坡道上段和高台面必然落进高度带而被判成障碍。

    关键点（都是实测踩出来的）：
      * 每格取点云 z 的最低点当候选表面，再做 **3x3 中值平滑**去掉单格离群；
      * 轨迹处用 轨迹z − IMU离地高度 直接覆盖（车开过 = 权威）；
      * ★ 从轨迹种子做**连通泛洪**：只有与当前地面相差 <= max_step 的相邻格才被接受为
        "同一张可走面"。**竖直墙面爬不上去**，于是墙格的表面不可达 —— 墙不会再被当成
        "平台"（老版本会把墙腰当地面，结果墙上的点"不算高"，墙从图上消失，导航直接撞墙）。
    """
    iy, ix, z = raster_index(pts, xmin, ymin, res, w, h)
    flat = iy * w + ix
    cnt = np.bincount(flat, minlength=h * w)
    zmin = np.full(h * w, np.inf)
    np.minimum.at(zmin, flat, z)
    zs = np.where(cnt > 0, zmin, np.nan).reshape(h, w)
    if smooth > 0:
        stack = np.stack([np.roll(np.roll(zs, dy, 0), dx, 1)
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
        with np.errstate(all='ignore'):
            med = np.nanmedian(stack, axis=0)
        zs = np.where(np.isfinite(zs), med, zs)

    elev = np.full((h, w), np.nan)
    reached = np.zeros((h, w), dtype=bool)
    if traj.shape[0] == 0:
        return zs, (iy, ix, z), 0
    # ---- 轨迹种子 ----
    r = max(1, int(round(seed_r / res)))
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
            if dy * dy + dx * dx <= r * r]
    ti = ((traj[:, 0] - xmin) / res).astype(np.int64)
    tj = ((traj[:, 1] - ymin) / res).astype(np.int64)
    seed = np.full((h, w), np.inf)
    for dy, dx in offs:
        yy = tj + dy
        xx = ti + dx
        ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        np.minimum.at(seed, (yy[ok], xx[ok]), (traj[:, 2] - imu_h)[ok])
    sm = np.isfinite(seed)
    elev[sm] = seed[sm]
    reached[sm] = True
    n_seed = int(sm.sum())
    # ---- 连通泛洪：|Δ| <= max_step 才认作同一张可走面 ----
    for _ in range(128):
        grew = 0
        for axis in (0, 1):
            e = elev if axis == 0 else elev.T
            nz = zs if axis == 0 else zs.T
            for sgn in (1, -1):
                src = np.roll(e, sgn, axis=0)
                dst = np.roll(nz, sgn, axis=0)
                cand = src                      # 候选高度 = 来源格的地面
                ok = np.isfinite(src) & np.isfinite(dst) & (np.abs(dst - src) <= max_step)
                if axis == 1:
                    ok = ok.T
                    cand = cand.T
                newm = ok & ~reached
                if newm.any():
                    elev[newm] = cand[newm]
                    reached[newm] = True
                    grew += int(newm.sum())
        if grew == 0:
            break
    # ---- 未连通的观测面：与**最近的已知地面**同高的 -> 也算地面 ----
    #   为什么需要：泛洪只覆盖"从轨迹走得到"的那张面，墙另一侧的平地（另一间屋子、
    #   被分隔墙隔开的半场）与轨迹不连通，若一律当障碍就会"把空地划成障碍"（实测踩过）。
    #   判据：把已知地面向外无约束地"最近邻铺开"当参考，未连通格的候选表面与参考
    #   相差 <= 2*max_step 的视为同一层地面；相差大的（墙/柱/结构）才留给上层标障碍。
    grow = elev.copy()
    for _ in range(max(h, w)):        # 铺到收敛：参考地面要覆盖全图，否则远处平地
                                      # 找不到参考高度、会被误判成"结构"
        prev = grow
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                sh = np.roll(np.roll(grow, dy, 0), dx, 1)
                grow = np.where(np.isfinite(grow), grow, sh)
        if np.array_equal(np.isfinite(grow), np.isfinite(prev)):
            break
    close = (~reached) & np.isfinite(zs) & np.isfinite(grow) & (np.abs(zs - grow) <= 2 * max_step)
    n_close = int(close.sum())
    if n_close:
        elev[close] = grow[close]
        reached |= close
    return elev, (iy, ix, z), n_seed


def terrain_occupancy(elev, idx, res, w, h, args):
    """★ 代价地图主逻辑（重构版）：**三件事 + 两条兜底**

    规则1 **特殊识别斜坡**：连续倾斜面（相邻格高差 <= max_step，坡度 > slope_flat
          且 <= max_slope）-> **可通行**，代价随坡度上升（写进 cost 栅格，不挡路）。
    规则2 **台阶 = 小幅度碰撞**：离散落差（|Δz| > max_step）且呈"多级"（窗口内落差
          格数 >= stair_min）-> 致命，但**只膨胀 stair_dilate**（默认 0.10 m，刻意小），
          既挡住又不会把周围自由区吃掉、也不会像膨胀层那样把窄坡道掐死。
    规则3 **过于低矮的物体 -> 可通过**：格内点高出当地地面 <= pass_h（默认 80 mm，
          即车能骑过去的高度）的突起**直接忽略**；介于 pass_h 与 obstacle_h 之间的
          给一个**中等代价**（--mid-cost）而不是判成障碍。

    兜底A 高出当地地面 > obstacle_h 的点、以及"从轨迹泛洪走不到的面" -> 障碍
          （墙/柱/结构侧面；后者专治"竖直墙面被当成平台、墙从图上消失"）。
    兜底B 没有任何地面依据 -> 未知（allow_unknown=false 时挡住规划）。

    返回 (occ, occ_obs, occ_step, occ_slope, info)：
      occ_obs  = 墙/结构（致命）
      occ_step = 台阶（致命 + 小膨胀；**不参与轨迹走廊优先权**）
      occ_slope= 其余落差边缘（坡道路肩/高台边缘 -> 写成 unknown：挡规划但不被膨胀层放大）
    """
    iy, ix, z = idx
    valid = np.isfinite(elev)
    slope = slope_field(elev, res, args.slope_base)
    step = step_mask(elev, res, args.max_step)
    has_pts = np.zeros((h, w), dtype=bool)
    has_pts[iy, ix] = True
    e_pt = np.where(valid[iy, ix], elev[iy, ix], np.nan)
    dz = z - e_pt
    fin = np.isfinite(e_pt)
    flat = iy * w + ix

    # ---- 规则3：低矮物体可通过 / 中等高度给代价 / 再高才是障碍 ----
    lab = np.full(h * w, -1, dtype=np.int8)
    m_low = fin & (dz > 0) & (dz <= args.pass_h) & (dz <= args.h_max)
    m_mid = fin & (dz > args.pass_h) & (dz <= args.min_obstacle_h) & (dz <= args.h_max)
    m_obs = fin & (dz > args.min_obstacle_h) & (dz <= args.h_max)
    for m, v in ((m_low, 0), (m_mid, 1), (m_obs, 2)):
        if m.any():
            lab[flat[m]] = np.maximum(lab[flat[m]], v)
    lab = lab.reshape(h, w)
    walls = lab == 2
    if args.min_pts > 1 and m_obs.any():
        cnt_obs = np.bincount(flat[m_obs], minlength=h * w).reshape(h, w)
        walls &= cnt_obs >= args.min_pts
    mid = (lab == 1) & ~walls
    low = (lab == 0) & ~walls
    unreached = has_pts & ~valid           # 兜底A：泛洪到不了的面 = 墙/柱/结构
    walls = walls | unreached
    mid &= ~walls

    # ---- 规则2：台阶 = 离散落差 + 多级 -> 致命 + 小膨胀 ----
    stairs_raw = np.zeros((h, w), dtype=bool)
    n_clusters = 0
    if step.any():
        R = max(1, int(round(args.stair_window / res)))
        dens = np.zeros((h, w), dtype=np.int32)
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                if dy * dy + dx * dx > R * R:
                    continue
                dens += np.roll(np.roll(step, dy, 0), dx, 1).astype(np.int32)
        stairs_raw = step & (dens >= args.stair_min)
        n_clusters = int(stairs_raw.sum())
    stairs = grow(stairs_raw, int(round(args.stair_dilate / res))) if args.stair_dilate > 0 else stairs_raw
    stairs = stairs & ~walls

    # ---- 规则1：斜坡（连续、可通行、代价随坡度）----
    ramp = valid & ~step & np.isfinite(slope) & (slope > args.slope_flat) & (slope <= args.max_slope)

    edge = step & ~stairs                    # 其余落差（坡道路肩/高台边缘）-> unknown
    occ = walls | stairs | edge
    info = {
        "elev_cells": int(valid.sum()),
        "obstacle_cells": int(walls.sum()),
        "unreached_cells": int(unreached.sum()),
        "stair_edge_cells": int(stairs_raw.sum()),
        "stair_zone_cells": int(stairs.sum()),
        "slope_edge_cells": int(edge.sum()),
        "mid_cost_cells": int(mid.sum()),
        "low_pass_cells": int(low.sum()),
        "ramp_cells": int(ramp.sum()),
        "occupied_cells": int(occ.sum()),
        "stairs": stairs,
        "stairs_raw": stairs_raw,
        "mid": mid,
        "low": low,
        "ramp": ramp,
        "slope_field": slope,
    }
    return occ, walls, stairs, edge, info


def corridor_mask(traj, xmin, ymin, res, w, h, radius):
    """轨迹走廊掩码：距轨迹任意采样点 <= radius 的格子。"""
    out = np.zeros((h, w), dtype=bool)
    if traj.shape[0] == 0:
        return out
    r = max(1, int(round(radius / res)))
    ti = ((traj[:, 0] - xmin) / res).astype(np.int64)
    tj = ((traj[:, 1] - ymin) / res).astype(np.int64)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx > r * r:
                continue
            yy = tj + dy
            xx = ti + dx
            ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            out[yy[ok], xx[ok]] = True
    return out


def corridor_hits(occ, traj, xmin, ymin, res, w, h, radius):
    """轨迹走廊内被判成占据的格数（用来对比"改前/改后"）。"""
    if traj.shape[0] == 0:
        return 0, 0
    r = max(1, int(round(radius / res)))
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
            if dy * dy + dx * dx <= r * r]
    ti = ((traj[:, 0] - xmin) / res).astype(np.int64)
    tj = ((traj[:, 1] - ymin) / res).astype(np.int64)
    seen = np.zeros((h, w), dtype=bool)
    for dy, dx in offs:
        yy = tj + dy
        xx = ti + dx
        ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        seen[yy[ok], xx[ok]] = True
    return int((seen & occ).sum()), int(seen.sum())


def fill_and_reclaim(elev, traj, xmin, ymin, res, w, h, imu_h, max_step, corridor_r, fill_cells):
    """把"已知高度"向外填几格，并把轨迹走廊里被未知包住的格子认回来。

    动机（实测）：坡道只有 1 m 宽，两侧很快就是雷达看不到的未知区；
    只按"已知高度"给空闲的话，坡道自由通道中位只有 0.60 m —— 车宽 0.56 m，
    Nav2 的膨胀层（内切半径 0.28 m）会把整条坡道判成致命区，规划器根本不敢走。
    所以：① 已知高度向外填 fill_cells 格（只补 NaN）；② 轨迹走廊内、填充值与
    "轨迹当地地面"相差不超过 max_step 的格子，认成已知地面。车开过的地方本来
    就是已知可通行的，这一步把自由通道恢复到 ≈ 走廊宽度。
    """
    ef = elev.copy()
    for _ in range(max(0, fill_cells)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                sh = np.roll(np.roll(ef, dy, 0), dx, 1)
                ef = np.where(np.isfinite(ef), ef, sh)
    tg = np.full((h, w), np.inf)
    if traj.shape[0]:
        r = max(1, int(round((corridor_r + 2 * res) / res)))
        offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                if dy * dy + dx * dx <= r * r]
        ti = ((traj[:, 0] - xmin) / res).astype(np.int64)
        tj = ((traj[:, 1] - ymin) / res).astype(np.int64)
        g = traj[:, 2] - imu_h
        for dy, dx in offs:
            yy = tj + dy
            xx = ti + dx
            ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            np.minimum.at(tg, (yy[ok], xx[ok]), g[ok])
    m = (~np.isfinite(elev)) & np.isfinite(ef) & np.isfinite(tg) & (np.abs(ef - tg) <= max_step)
    n = int(m.sum())
    if n:
        elev = np.where(m, ef, elev)
    return elev, n


def slope_field(elev, res, base_m):
    """用 base_m 基线的高差估坡度（度）；NaN 传播，返回 (h,w) 浮点。

    为什么用"最大高差/基线"而不是拟合平面：实现简单、对墙面一样的陡峭面不敏感于
    点云厚度；实测坡道 11.7°、高台 1.3°、台阶/墙 p90 >60°，足以分档。
    """
    h, w = elev.shape
    s = max(1, int(round(base_m / res)))
    out = np.full((h, w), np.nan)
    for axis in (0, 1):
        e = elev if axis == 0 else elev.T
        a, b = e[:-s, :], e[s:, :]
        both = np.isfinite(a) & np.isfinite(b)
        d = np.where(both, np.abs(a - b) / (s * res), np.nan)
        t = np.full(e.shape, np.nan)
        t[:-s, :] = d
        t[s:, :] = np.fmax(t[s:, :], d)
        if axis == 1:
            t = t.T
        out = np.fmax(out, t) if axis == 1 else t
    return np.degrees(np.arctan(out))


def step_mask(elev, res, max_step):
    """相邻 1 格高差 > max_step 的格（两侧都标）。"""
    h, w = elev.shape
    out = np.zeros((h, w), dtype=bool)
    for axis in (0, 1):
        e = elev if axis == 0 else elev.T
        both = np.isfinite(e[:-1, :]) & np.isfinite(e[1:, :])
        bad = both & (np.abs(e[:-1, :] - e[1:, :]) > max_step)
        t = np.zeros(e.shape, dtype=bool)
        t[:-1, :] |= bad
        t[1:, :] |= bad
        out |= (t if axis == 0 else t.T)
    return out


def detect_stairs(step, res, window_m, min_count):
    """从"相邻高差 > max_step"的掩码里找出**连续多级的楼梯区**。

    依据：楼梯会在一小片区域里出现很多级落差（实测台阶每级 150 mm，
    0.6 m 邻域内很容易累计到 ≥6 格）；而坡道路肩、高台边缘是孤立的一处落差，
    邻域内级数很少。两者分开处理，才能"楼梯设禁区、坡道照样能走"。
    """
    h, w = step.shape
    R = max(1, int(round(window_m / res)))
    dens = np.zeros((h, w), dtype=np.int32)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if dy * dy + dx * dx > R * R:
                continue
            dens += np.roll(np.roll(step, dy, 0), dx, 1).astype(np.int32)
    return dens >= min_count


def grow(mask, cells):
    """形态学膨胀 cells 格（8 邻域）。"""
    out = mask.copy()
    for _ in range(max(0, cells)):
        d = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                d |= np.roll(np.roll(out, dy, 0), dx, 1)
        out = d
    return out


def stair_zone_from_traj(traj, step, xmin, ymin, res, w, h, stair_slope, window_m, dilate_m):
    """从**轨迹**里识别楼梯：相邻采样点 "|Δz|/Δxy > stair_slope" 的路段（离散落差），
    取该处 window_m 邻域内的"台阶边"格子，再膨胀 dilate_m。

    为什么不用"台阶边密度"：实测场地上墙脚、坡道路肩的台阶边也很密，
    密度法会把 25 m^2 都判成楼梯（把坡道也堵了）。而车真正爬上/爬下的那条
    楼梯，在轨迹上表现为连续几段陡于 0.35 的落差 —— 用它定位最准、范围最小。
    """
    zone = np.zeros((h, w), dtype=bool)
    if traj.shape[0] < 3 or not step.any():
        return zone, 0
    dd = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1)
    dz = np.abs(np.diff(traj[:, 2]))
    with np.errstate(divide='ignore', invalid='ignore'):
        sl = np.where(dd > 1e-3, dz / np.maximum(dd, 1e-9), 0.0)
    seg = np.concatenate([[False], (sl > stair_slope) & (dz > 0.05)])
    n_seg = int(seg.sum())
    if n_seg == 0:
        return zone, 0
    R = max(1, int(round(window_m / res)))
    ti = ((traj[seg, 0] - xmin) / res).astype(np.int64)
    tj = ((traj[seg, 1] - ymin) / res).astype(np.int64)
    near = np.zeros((h, w), dtype=bool)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if dy * dy + dx * dx > R * R:
                continue
            yy = tj + dy
            xx = ti + dx
            ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            near[yy[ok], xx[ok]] = True
    cells = step & near
    zone = grow(cells, int(round(dilate_m / res))) if dilate_m > 0 else cells
    return zone, n_seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pcd', nargs='*')
    ap.add_argument('--cloud', action='append', default=[],
                    help='额外点云，可重复；格式 PATH 或 PATH,tx,ty,yaw_deg')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--trajectory', action='append', default=[],
                    help='轨迹文件，可重复；格式 PATH 或 PATH,tx,ty,yaw_deg')
    ap.add_argument('--resolution', type=float, default=0.05)
    ap.add_argument('--h-min', type=float, default=0.15,
                    help='障碍高度下限（相对地面）。要略大于 LIO 的高度漂移')
    ap.add_argument('--h-max', type=float, default=2.0)
    ap.add_argument('--ground-z', type=float, default=None)
    ap.add_argument('--imu-height', type=float, default=0.30,
                    help='IMU/雷达离地高度（用于从轨迹起点推地面 z）')
    ap.add_argument('--max-range', type=float, default=20.0, help='空闲区射线最大长度')
    ap.add_argument('--min-range', type=float, default=0.45)
    ap.add_argument('--step-deg', type=float, default=1.0, help='射线角分辨率（越小空闲区越完整、越慢）')
    ap.add_argument('--traj-stride', type=int, default=2, help='轨迹抽稀步长（1=全部用上）')
    ap.add_argument('--pad', type=float, default=0.30)
    ap.add_argument('--margin', type=float, default=3.0,
                    help='地图范围 = 轨迹包围盒 + 本余量（米）；默认 3 m')
    ap.add_argument('--free-radius', type=float, default=0.5,
                    help='轨迹走廊半径（米）：车走过的地方一定可通行，'
                         '直接标空闲，保证空闲区沿行车路线连通')
    ap.add_argument('--bounds', default='auto',
                    help="地图范围：auto(默认=轨迹+margin) / full(按点云极值) / "
                         "'xmin,ymin,xmax,ymax'")
    ap.add_argument('--min-pts', type=int, default=1,
                    help='一格至少落进 N 个点才算障碍（默认 1）。稀疏点云毛刺多时可加到 2~3，'
                         '但细杆/远墙也会被一起滤掉，慎用')
    ap.add_argument('--min-area', type=float, default=0.05,
                    help='小于该面积(m^2)的孤立占据块整块删掉（默认 0.05，即 20 格；0=关）')
    ap.add_argument('--close', type=int, default=1)
    ap.add_argument('--open', type=int, default=0,
                    help='对占据集合做 N 次开运算（清掉比 2N+1 格更细的结构）。'
                         '本场地 98%% 的占据格都是 1 格宽，开了会把真墙也清掉，'
                         '只在你确认某片是毛刺时用')
    ap.add_argument('--transform', default='0,0,0')
    ap.add_argument('--no-free-space', action='store_true')
    # ---- 方案 A：当地地面高度场 + 坡度/高差判据（把坡道/高台变成可通行区）----
    ap.add_argument('--terrain', type=int, default=1,
                    help='1(默认)=用当地地面高度场判障碍，坡道/高台可通行；'
                         '0=老做法（相对最低地面 0.15~2.0 m 的高度带）')
    ap.add_argument('--max-slope', type=float, default=15.0,
                    help='可通行最大坡度(度)。9.7 度坡道 -> 15 度留约 50%% 余量')
    ap.add_argument('--max-step', type=float, default=0.08,
                    help='相邻格允许的最大高差(米)。150 mm 台阶 = 0.15 > 0.08 -> 挡住')
    ap.add_argument('--min-obstacle-h', type=float, default=0.12,
                    help='高出**当地地面**多少算障碍(米)。默认 0.12：80 mm 围栏可骑过，'
                         '150 mm 台阶挡住')
    ap.add_argument('--elev-seed-r', type=float, default=0.30,
                    help='轨迹处地面高度种子的作用半径(米)')
    ap.add_argument('--elev-smooth', type=int, default=1,
                    help='高度场规整次数(默认 1)')
    ap.add_argument('--pass-h', type=float, default=0.08,
                    help='★规则3：高出当地地面不超过该值(米)的突起算"过于低矮" -> 可通过')
    ap.add_argument('--slope-flat', type=float, default=5.0,
                    help='★规则1：坡度超过该值(度)才算斜坡（平地=0，坡道 9.7）')
    ap.add_argument('--stair-min', type=int, default=12,
                    help='★规则2：窗口内落差格数 >= 该值才算"多级台阶"（否则按单处落差）')
    ap.add_argument('--mid-cost', type=int, default=60,
                    help='★规则3：介于 pass-h 与 min-obstacle-h 之间的突起给的代价(0~99)')
    ap.add_argument('--stair-dilate', type=float, default=0.10,
                    help='楼梯禁区在台阶边之外再膨胀多少米（默认 0.10，刻意取小，'
                         '避免吃掉过多自由区；0=不膨胀）')
    ap.add_argument('--stair-window', type=float, default=0.60,
                    help='判定"连续多级楼梯"的邻域半径（米）')
    ap.add_argument('--stair-slope', type=float, default=0.35,
                    help='轨迹里 |Δz|/Δxy 超过该值视为"离散落差路段"（楼梯），'
                         '9.7° 坡道是 0.17，150 mm/300 mm 台阶是 0.5')
    ap.add_argument('--cost-free-slope', type=float, default=5.0,
                    help='代价栅格里"完全免费"的坡度上限（度）')
    ap.add_argument('--cost-max-slope', type=float, default=30.0,
                    help='代价栅格里"最贵但仍然可通行"的坡度（度）')
    ap.add_argument('--edge-mode', choices=('unknown', 'occ'), default='unknown',
                    help='落差边缘（台阶/悬崖/路肩）在图上写成 unknown(默认) 还是 occupied。'
                         'unknown：挡规划但不被膨胀层放大 —— 1 m 宽的坡道只有这样才能过；'
                         'occupied：老行为，膨胀层会把窄通道掐死')
    ap.add_argument('--corridor-r', type=float, default=0.35,
                    help='轨迹走廊优先权半径(米，默认 0.32 = 车半宽 0.28 + 余量)：'
                         '车实际开过的地方强制可通行（台阶/陡坡边缘除外），'
                         '否则 Nav2 膨胀后会把 1 m 宽的坡道掐死')
    ap.add_argument('--elev-fill', type=int, default=4,
                    help='把已知地面高度向外填充几格（默认 4 格 = 0.2 m），'
                         '把坡道/走廊被"未知"包住的自由通道补出来')
    ap.add_argument('--slope-base', type=float, default=0.15,
                    help='算坡度用的基线长度(米)')
    ap.add_argument('--no-height', action='store_true',
                    help='不写 <out>_elev.npy（高度场，调试/诊断用）')
    args = ap.parse_args(argv_fix(sys.argv[1:]))

    clouds = list(args.cloud) + list(args.pcd)
    if not clouds:
        print('need at least one .pcd (positional or --cloud)')
        return 2
    gtx, gty, gtyaw = [float(v) for v in args.transform.split(',')]

    pts_all, traj_all = [], []
    for spec in clouds:
        p = spec.split(',')
        a = read_pcd(p[0])
        if len(p) >= 4:
            a = tf2d(a, float(p[1]), float(p[2]), float(p[3]))
            print('  %s  %d 点（自带变换 %s）' % (p[0], len(a), p[1:4]))
        else:
            a = tf2d(a, gtx, gty, gtyaw)
            print('  %s  %d 点' % (p[0], len(a)))
        pts_all.append(a)
    for spec in args.trajectory:
        p = spec.split(',')
        t = read_traj(p[0])
        t = tf2d(t, *([float(p[1]), float(p[2]), float(p[3])] if len(p) >= 4 else [gtx, gty, gtyaw]))
        print('  轨迹 %s  %d 点' % (p[0], len(t)))
        traj_all.append(t)
    pts = np.concatenate(pts_all, axis=0)
    traj = np.concatenate(traj_all, axis=0) if traj_all else np.zeros((0, 3))
    print('合计 %d 点' % len(pts))

    # ---- ground_z ----
    gz = args.ground_z
    if gz is None and traj.shape[0] >= 3:
        n0 = max(3, min(20, traj.shape[0] // 10))
        z0 = float(np.median(traj[:n0, 2]))
        gz = z0 - args.imu_height
        print('ground_z 由轨迹起点推算：z(start)=%.3f - imu_height %.2f = %.3f'
              % (z0, args.imu_height, gz))
    if gz is None:
        sel = pts[:, 2]
        gz = float(np.percentile(sel, 2.0))
        print('ground_z 退回 z 的 2%% 分位数 = %.3f（建议显式给 --ground-z）' % gz)

    zr = pts[:, 2] - gz
    print('ground_z = %.3f m；点云相对地面 z ∈ [%.2f, %.2f]' % (gz, zr.min(), zr.max()))

    # ★ 默认按"轨迹包围盒 + margin"裁剪，而不是按点云极值：
    #   雷达能看 17 m，累积点云的极值可能横跨 40 m（整个场馆），按它建图会得到
    #   一张 40x40 m、90%+ 未知的代价地图（实测踩过：Nav2 基本没法用）。
    #   导航真正需要的只是"车走过、看到过"的区域。
    #   注意：3D 点云本身（prior_map.pcd）仍是**全量**的，重定位用全量特征，
    #         不受这里裁剪影响；裁的只是交给 map_server 的 2D 占据栅格。
    if args.bounds and args.bounds != 'auto':
        if args.bounds == 'full':
            xmin, ymin = pts[:, 0].min() - args.pad, pts[:, 1].min() - args.pad
            xmax, ymax = pts[:, 0].max() + args.pad, pts[:, 1].max() + args.pad
            print('地图范围：点云极值 + pad %.2f m -> %.1f x %.1f m'
                  % (args.pad, xmax - xmin, ymax - ymin))
        else:
            xmin, ymin, xmax, ymax = [float(v) for v in args.bounds.split(',')]
            print('地图范围：显式给定 [%.2f, %.2f, %.2f, %.2f]' % (xmin, ymin, xmax, ymax))
    elif traj.shape[0] >= 3:
        m = args.margin
        xmin, ymin = traj[:, 0].min() - m, traj[:, 1].min() - m
        xmax, ymax = traj[:, 0].max() + m, traj[:, 1].max() + m
        print('地图范围：轨迹包围盒 + margin %.2f m -> [%.2f, %.2f]~[%.2f, %.2f]（%.1f x %.1f m）'
              % (m, xmin, ymin, xmax, ymax, xmax - xmin, ymax - ymin))
    else:
        xmin, ymin = pts[:, 0].min() - args.pad, pts[:, 1].min() - args.pad
        xmax, ymax = pts[:, 0].max() + args.pad, pts[:, 1].max() + args.pad
        print('地图范围：没有轨迹，退回点云极值 + pad %.2f m' % args.pad)
    w = int(math.ceil((xmax - xmin) / args.resolution))
    h = int(math.ceil((ymax - ymin) / args.resolution))
    # ---- 老做法（固定高度带）：两种模式都算一遍，用于对比 + --terrain 0 时直接用 ----
    occ_legacy = np.zeros((h, w), dtype=bool)
    sel = pts[(zr >= args.h_min) & (zr <= args.h_max)]
    if sel.size:
        lix = ((sel[:, 0] - xmin) / args.resolution).astype(np.int64)
        liy = ((sel[:, 1] - ymin) / args.resolution).astype(np.int64)
        lok = (lix >= 0) & (lix < w) & (liy >= 0) & (liy < h)
        if args.min_pts <= 1:
            occ_legacy[liy[lok], lix[lok]] = True
        else:
            # 密度阈值：一格至少落进 N 个点才算障碍（稀疏单点回波不当障碍）
            flat = np.bincount(liy[lok] * w + lix[lok], minlength=h * w).reshape(h, w)
            occ_legacy = flat >= args.min_pts
            print('  密度阈值 --min-pts %d：有回波的格 %d，达到阈值 %d，'
                  '被丢弃 %d 格' % (args.min_pts, int((flat > 0).sum()), int(occ_legacy.sum()),
                                    int((flat > 0).sum() - occ_legacy.sum())))

    tinfo = None
    n_seed = 0
    if args.terrain:
        # ---- 方案 A：先建当地地面高度场，再按"高出当地地面多少"判障碍 ----
        elev, idx, n_seed = build_elevation(
            pts, traj, xmin, ymin, args.resolution, w, h, args.imu_height,
            args.elev_seed_r, args.max_step, args.elev_smooth)
        n_reclaim = 0   # 泛洪代替了 已知高度外填+回填（回填会把墙重新认成地面）
        occ, occ_obs, occ_step, occ_slope, tinfo = terrain_occupancy(
            elev, idx, args.resolution, w, h, args)
        print('地形模式：高度场 %d 格（轨迹种子 %d 格，走廊回填 %d 格），坡度上限 %.1f°，'
              '相邻高差上限 %.0f mm，障碍阈值 +%.0f mm'
              % (tinfo['elev_cells'], n_seed, n_reclaim, args.max_slope,
                 1000 * args.max_step, 1000 * args.min_obstacle_h))
        print('  ★ 三规则分类：斜坡(可通行) %d 格 | 台阶(小碰撞 %.2f m) %d 格'
              ' | 低矮可通过 %d 格 | 中等代价 %d 格'
              % (tinfo['ramp_cells'], args.stair_dilate, tinfo['stair_zone_cells'],
                 tinfo['low_pass_cells'], tinfo['mid_cost_cells']))
        print('  障碍(墙/结构) %d 格（其中泛洪到不了的面 %d 格）| 其余落差边缘 %d 格'
              % (tinfo['obstacle_cells'], tinfo['unreached_cells'], tinfo['slope_edge_cells']))
        # ---- 方案甲：楼梯区检测 + 小幅膨胀成禁区 ----
        slope_deg = slope_field(elev, args.resolution, args.slope_base)
        stair_zone = tinfo['stairs']        # 规则2：台阶（小幅度碰撞，已按 stair_dilate 膨胀）
        stair_cells = tinfo['stairs_raw']
        n_seg = int(tinfo['stair_zone_cells'])
        tinfo['slope_free_slope_deg'] = args.cost_free_slope
        tinfo['slope_max_slope_deg'] = args.cost_max_slope
        tinfo['stair_cells'] = int(stair_cells.sum())
        tinfo['stair_traj_segments'] = int(n_seg)
        tinfo['stair_zone_cells'] = int(stair_zone.sum())
        tinfo['stair_dilate_m'] = args.stair_dilate
        # 台阶已由 terrain_occupancy 的规则2 处理（上面已打印）
        if not args.no_height:
            np.save(args.out + '_elev.npy', elev)
    else:
        occ = occ_legacy
        slope_deg = None
        stair_zone = np.zeros((h, w), dtype=bool)
        print('（--terrain 0：固定高度带老做法）')

    n_old, n_tot = corridor_hits(occ_legacy, traj, xmin, ymin, args.resolution, w, h,
                                 args.free_radius)
    n_new, _ = corridor_hits(occ, traj, xmin, ymin, args.resolution, w, h, args.free_radius)
    if n_tot:
        print('轨迹走廊(半径 %.2f m)内占据格：老做法 %d/%d (%.1f%%) -> 本次 %d (%.1f%%)'
              % (args.free_radius, n_old, n_tot, 100.0 * n_old / n_tot, n_new,
                 100.0 * n_new / n_tot))
    print('栅格 %d x %d = %.1f x %.1f m，原点 (%.2f, %.2f)，占据格 %d (%.2f%%)'
          % (w, h, w * args.resolution, h * args.resolution, xmin, ymin,
             occ.sum(), 100.0 * occ.mean()))
    if args.open:
        # ★ 这里必须作用在"占据集合"上：老写法是 grid[morph(grid==0,..,'open')]=0，
        #   而开运算是原集合的**子集**，等于什么都没删（实测 --open 1/2 输出完全一样）。
        #   开运算会清掉比 (2N+1) 格更细的结构 —— 本场地 98.6% 的占据格都是 1 格宽，
        #   所以这个参数很凶，默认 0 关闭，只在"毛刺确实盖住整条走廊"时才用。
        occ = morph(occ, args.open, 'open')
        print('  形态学开运算 --open %d：占据格 -> %d（清掉比 %d 格更细的结构，慎用）'
              % (args.open, int(occ.sum()), 2 * args.open + 1))
    # 真实障碍（高出当地地面的点）去毛刺；落差边缘单独一层，不参与去毛刺
    edges = (occ_step | occ_slope) if args.terrain else np.zeros_like(occ)
    occ_pts = occ_obs if args.terrain else occ
    occ_pts, n_rm = despeckle(occ_pts, args.resolution, args.min_area)
    occ = occ_pts
    edges = edges & ~occ_pts
    if n_rm:
        print('  去毛刺 --min-area %.3f m^2：删掉 %d 个孤立占据格（占原占据 %.1f%%）'
              % (args.min_area, n_rm, 100.0 * n_rm / max(1, n_rm + int(occ.sum()))))
        print('  去毛刺后占据格 %d (%.2f%%)' % (int(occ.sum()), 100.0 * occ.mean()))

    grid = np.full((h, w), 205, dtype=np.uint8)
    if traj.shape[0] and not args.no_free_space:
        free = cast_free(occ, traj[::max(1, args.traj_stride)], (xmin, ymin),
                         args.resolution, args.max_range, args.min_range, args.step_deg)
        free, nstamp = stamp_corridor(free, traj[::max(1, args.traj_stride)], (xmin, ymin),
                                      args.resolution, args.free_radius)
        if args.terrain:
            # ★ 空闲区只保留“已知当地地面高度”的格子。
            #   否则坡道/高台边缘外侧那些**悬空**格子（雷达既没看到地面、也没看到墙）
            #   会被射线/走廊标成 free，Nav2 就可能规划出“从边缘开下去”的路径 ——
            #   实测过：10 对相邻空闲格高差 > 0.25 m。地面未知的地方应该是**未知**。
            known = np.isfinite(elev)
            n_drop = int((free & ~known).sum())
            free &= known
            if n_drop:
                print('  ★ 剔除“地面未知”的空闲格 %d 个（避免边缘悬空处被当成可通行）' % n_drop)
        grid[free] = 254
        print('射线投射：%d 个轨迹点 x %d 条射线；轨迹走廊半径 %.2f m（%d 格/点）'
              % (traj[::max(1, args.traj_stride)].shape[0], int(360 / args.step_deg),
                 args.free_radius, nstamp))
    else:
        grid[~occ] = 254
        print('没有轨迹：所有非占据格算空闲')
    grid[occ] = 0
    if args.terrain and args.edge_mode == 'unknown' and edges.any():
        # ★ 落差边缘（台阶/悬崖/路肩）写成 unknown：allow_unknown=false 时它照样挡住
        #   规划，但**不会被膨胀层放大** —— 这是 1 m 宽坡道（车宽 0.56 m）唯一能过的
        #   写法。写成 occupied 的话，两侧路肩各膨胀 0.28 m（内切半径），通道全变致命区，
        #   车一走进去就是"Starting point in lethal space"（实测踩过）。
        grid[edges] = 205
        if stair_zone.any():
            # ★ 楼梯禁区：写**致命**（occupied），让 Nav2 的膨胀层再放大一点 ——
            #   这是"台阶及其附近不可通行"的实现，和坡道路肩（unknown，不膨胀）区分开。
            grid[stair_zone] = 0
    if args.terrain:
        # ★ 轨迹走廊优先权：车实际开过的地方一定可通行。
        #   为什么必须要有：坡道只有 1 m 宽、车宽 0.56 m，地形判据会把路肩（连续
        #   落差）标成占据，自由通道被压到 ~0.70 m；Nav2 的膨胀层（内切半径 0.28 m）
        #   会把它整条判成致命区 -> 规划器根本不敢走坡道。
        #   规则：走廊(默认 ±0.35 m)强制可通行，但**台阶边缘(occ_step)不override**——
        #   台阶/高台边缘是离散落差(> max_step)，必须继续挡住（楼梯、悬崖）。
        #   连续坡面在 1 格内的高差只有 8.6 mm，走的是 occ_slope，允许 override。
        force = corridor_mask(traj[::max(1, args.traj_stride)], xmin, ymin,
                              args.resolution, w, h, args.corridor_r)
        force &= ~occ_step & ~occ_obs
        n_force = int((force & (grid != 254)).sum())
        grid[force] = 254
        #   再沿走廊边界打一道"落差围栏"：走廊外一格的当地地面比走廊低超过 max_step
        #   的地方标占据，防止规划器从走廊侧向开出去（掉下路肩）。
        fence = np.zeros((h, w), dtype=bool)
        for axis in (0, 1):
            e = elev if axis == 0 else elev.T
            fr = force if axis == 0 else force.T
            a, b = e[:-1, :], e[1:, :]
            fa, fb = fr[:-1, :], fr[1:, :]
            both = np.isfinite(a) & np.isfinite(b)
            dzz = np.abs(a - b)
            m = both & (dzz > args.max_step)
            out_a = m & fa & ~fb & (a < b)      # 走廊内、外侧更低 -> 外侧加围栏
            out_b = m & fb & ~fa & (b < a)
            ta = (fence if axis == 0 else fence.T)[:-1, :]
            tb = (fence if axis == 0 else fence.T)[1:, :]
            ta |= out_a
            tb |= out_b
        n_fence = int((fence & ~force).sum())
        grid[fence & ~force] = 0
        if n_force or n_fence:
            print('  ★ 走廊优先权(±%.2f m)：%d 格恢复可通行；边界落差围栏 %d 格'
                  % (args.corridor_r, n_force, n_fence))
    if args.close:
        # 闭运算：把"虚线"接成实线、填掉凹口（默认 1 次）。它只会增加占据格，
        # 不会删，所以不会造成"漏检障碍"。
        grid[morph(grid == 0, args.close, 'close')] = 0

    if args.terrain:
        # ---- 方案甲产物：代价栅格（给将来的 terrain_cost_layer 插件用）+ 可通行性预览 ----
        cost = np.full((h, w), 255, dtype=np.uint8)          # 255 = 未知/无信息
        known = np.isfinite(elev)
        span = max(1e-6, args.cost_max_slope - args.cost_free_slope)
        cs = np.clip((slope_deg - args.cost_free_slope) / span * 90.0, 0, 99)
        cost[known] = cs[known].astype(np.uint8)
        cost[tinfo['mid']] = args.mid_cost      # ★规则3：低矮~中等高度的突起 = 可通过但贵一点
        cost[grid == 205] = 255
        cost[grid == 0] = 100                                # 致命（含楼梯禁区）
        with open(args.out + "_cost.pgm", "wb") as f:
            f.write(b"P5\n# cost: 0=free 1..99=graded 100=lethal 255=unknown\n%d %d\n255\n" % (w, h))
            f.write(cost[::-1, :].tobytes())
        try:
            from PIL import Image
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[grid == 205] = (150, 150, 150)               # 未知
            rgb[grid == 0] = (30, 30, 30)                    # 致命：障碍 / 楼梯禁区
            free = grid == 254
            sd = slope_deg
            fin = free & np.isfinite(sd)
            rgb[fin & (sd <= args.cost_free_slope)] = (60, 180, 75)
            rgb[fin & (sd > args.cost_free_slope) & (sd <= 15)] = (240, 200, 60)
            rgb[fin & (sd > 15) & (sd <= args.cost_max_slope)] = (240, 140, 40)
            rgb[fin & (sd > args.cost_max_slope)] = (200, 40, 40)
            Image.fromarray(rgb[::-1]).save(args.out + "_traversability.png")
        except ImportError:
            pass
        print("  代价栅格 -> %s_cost.pgm（0=免费 1~99=渐变 100=致命 255=未知）；"
              "可通行性预览 -> %s_traversability.png" % (args.out, args.out))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + '.pgm', 'wb') as f:
        f.write(b'P5\n# generated by map3d_to_costmap.py\n%d %d\n255\n' % (w, h))
        f.write(grid[::-1, :].tobytes())
    with open(args.out + '.yaml', 'w') as f:
        f.write('image: %s.pgm\n' % os.path.basename(args.out))
        f.write('mode: trinary\n')
        f.write('resolution: %.3f\n' % args.resolution)
        f.write('origin: [%.3f, %.3f, 0.0]\n' % (xmin, ymin))
        f.write('negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    meta = {'ground_z': gz, 'imu_height': args.imu_height, 'resolution': args.resolution,
            'origin': [xmin, ymin, 0.0], 'h_min': args.h_min, 'h_max': args.h_max,
            'size_m': [w * args.resolution, h * args.resolution],
            'points': int(len(pts)),
            'ratio': {'occupied': float((grid == 0).mean()),
                      'free': float((grid == 254).mean()),
                      'unknown': float((grid == 205).mean())}}
    if tinfo is not None:
        meta['terrain'] = {k: v for k, v in tinfo.items()
                           if isinstance(v, (int, float, str, bool, type(None)))}
        meta['terrain']['edge_cells_as'] = args.edge_mode
        meta['terrain']['cost_files'] = {
            'cost_pgm': os.path.basename(args.out) + '_cost.pgm',
            'traversability_png': os.path.basename(args.out) + '_traversability.png'}
        meta['terrain'].update({
            'max_slope_deg': args.max_slope,
            'max_step_m': args.max_step,
            'min_obstacle_h_m': args.min_obstacle_h,
            'elev_seed_cells': int(n_seed),
            'elev_file': (None if args.no_height
                          else os.path.basename(args.out) + '_elev.npy'),
        })
        meta['corridor_occupied'] = {'legacy_height_band': int(n_old),
                                     'terrain': int(n_new), 'corridor_cells': int(n_tot)}
    with open(args.out + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    try:
        from PIL import Image
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[grid == 254] = (245, 245, 245)
        rgb[grid == 205] = (150, 150, 150)
        rgb[grid == 0] = (0, 0, 0)
        Image.fromarray(rgb[::-1]).save(args.out + '_preview.png')
    except ImportError:
        pass
    print('写好：%s.pgm/.yaml/.meta.json' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
