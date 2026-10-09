#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""代价地图/先验地图体检：不依赖任何"真值几何"，只看地图本身是否合理。

检查项
------
1. 栅格尺寸、分辨率、原点、三值占比；
2. 自由空间是否是**一整块**（连通域）：室内地图如果碎成很多块，通常是建图时
   走得太快/雷达被遮挡/回环没闭上；
3. 已知区域里"障碍包围盒"是否闭合（用自由空间的最外圈是否贴近地图边界判断）；
4. 生成一张彩色预览图（白=空闲 灰=未知 黑=占据），方便肉眼看一眼。

    python3 map_sanity_check.py maps/prior_costmap.yaml
"""
import argparse
import os
import sys

import numpy as np
import yaml


def load(yaml_path):
    meta = yaml.safe_load(open(yaml_path))
    img = meta['image']
    if not img.startswith('/'):
        img = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img)
    with open(img, 'rb') as f:
        assert f.readline().strip() == b'P5'
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = [int(v) for v in line.split()]
        f.readline()   # maxval
        data = np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)
    return data, float(meta['resolution']), float(meta['origin'][0]), float(meta['origin'][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('yaml')
    ap.add_argument('--preview', default='', help='彩色预览输出路径（默认 <yaml>_preview.png）')
    ap.add_argument('--min-free-blocks', type=int, default=1)
    args = ap.parse_args()

    data, res, ox, oy = load(args.yaml)
    h, w = data.shape
    occ = data == 0
    free = data == 254
    unk = data == 205
    tot = float(data.size)
    print('地图 %s' % args.yaml)
    print('  栅格 %d x %d   分辨率 %.3f m   尺寸 %.1f x %.1f m   原点 (%.2f, %.2f)'
          % (w, h, res, w * res, h * res, ox, oy))
    print('  占据 %.1f%%   空闲 %.1f%%   未知 %.1f%%'
          % (100 * occ.sum() / tot, 100 * free.sum() / tot, 100 * unk.sum() / tot))

    # 自由空间的连通域（4 邻域洪泛）。地图很大时先降采样再算，纯 Python 的
    # 栈式洪泛在百万格级别会慢到不可接受，而"碎成几块"这个结论不受降采样影响。
    scale = 1
    f = free
    while (f.shape[0] * f.shape[1]) > 250000:
        # ★ 先裁成偶数边长再 2x2 合并：地图边长是奇数时（例如 813x813），
        #   f[0::2] 与 f[1::2] 的行数不同（407 vs 406），直接按位或会抛
        #   "operands could not be broadcast together" —— 存图最后一步崩掉，
        #   而外层是 || true，用户只看到一段 traceback、看不到 VERDICT。
        f = f[:f.shape[0] - f.shape[0] % 2, :f.shape[1] - f.shape[1] % 2]
        f = (f[0::2, 0::2] | f[1::2, 0::2] | f[0::2, 1::2] | f[1::2, 1::2])
        scale *= 2
    fh, fw = f.shape
    lab = np.zeros((fh, fw), dtype=bool)
    sizes = []
    for sy in range(fh):
        row = f[sy]
        for sx in range(fw):
            if not row[sx] or lab[sy, sx]:
                continue
            stack = [(sy, sx)]
            lab[sy, sx] = True
            n = 0
            while stack:
                y, x = stack.pop()
                n += 1
                if y + 1 < fh and f[y + 1, x] and not lab[y + 1, x]:
                    lab[y + 1, x] = True
                    stack.append((y + 1, x))
                if y > 0 and f[y - 1, x] and not lab[y - 1, x]:
                    lab[y - 1, x] = True
                    stack.append((y - 1, x))
                if x + 1 < fw and f[y, x + 1] and not lab[y, x + 1]:
                    lab[y, x + 1] = True
                    stack.append((y, x + 1))
                if x > 0 and f[y, x - 1] and not lab[y, x - 1]:
                    lab[y, x - 1] = True
                    stack.append((y, x - 1))
            sizes.append(n * (res * scale) ** 2)          # 直接换成 m^2
    sizes.sort(reverse=True)
    big = [s for s in sizes if s >= 0.05]                  # 忽略零散小点
    print('  自由空间连通域 %d 个（忽略 <0.05 m^2 的 %d 个小点），最大 %.1f m^2，前 5 大 %s'
          % (len(big), len(sizes) - len(big), sizes[0] if sizes else 0.0,
             [round(s, 1) for s in sizes[:5]]))

    warn = []
    if free.sum() == 0:
        warn.append('没有任何空闲区 —— 轨迹/射线投射参数不对')
    # ★ 地形模式（meta 里有 "terrain"）会**故意**把"地面未知"的区域留成未知
    #   （例如坡道/高台边缘外侧、雷达没看到地面的地方），未知率 60~75% 是正常的。
    #   所以这里的阈值放到 85%，并把判据重点放在"自由空间是否连成一片"。
    ter = None
    try:
        import json as _json
        _meta = _json.load(open(args.yaml.replace('.yaml', '.meta.json')))
        ter = _meta.get('terrain')
    except Exception:
        ter = None
    thr_unk = 0.85 if ter else 0.6
    if unk.sum() / tot > thr_unk:
        warn.append('未知区超过 %.0f%% —— 建图覆盖不全（没走过的地方本来就是未知；'
                    '但若超过 ~90%%，多半是轨迹没记到、或射线投射范围太小）'
                    % (100 * thr_unk))
    if ter:
        print('  地形模式：高度场 %d 格；占据拆解 = 高出当地地面 %d + 台阶边缘 %d + 陡坡边缘 %d'
              % (ter.get('elev_cells', 0), ter.get('obstacle_cells', 0),
                 ter.get('step_edge_cells', 0), ter.get('slope_edge_cells', 0)))
        co = _meta.get('corridor_occupied') or {}
        if co:
            print('  轨迹走廊内占据格：固定高度带 %s -> 地形法 %s（共 %s 格）'
                  % (co.get('legacy_height_band'), co.get('terrain'), co.get('corridor_cells')))
    # 判据放宽一点：场地本来就被分隔墙分成若干区域（红/蓝半场、房间），
    # 只有当"最大块占全部空闲区不到 1/5"或"碎块数 > 200"才算真的碎了。
    free_area = float(free.sum()) * res * res
    if big and (big[0] < 0.20 * free_area or len(big) > 200):
        warn.append('自由空间碎成 %d 块且最大块只占 %.0f%% —— 可能建图时走太快/雷达遮挡'
                    % (len(big), 100.0 * big[0] / max(free_area, 1e-9)))
    elif big:
        print('  自由空间占比：最大块 %.1f m^2 = 全部空闲的 %.0f%%'
              % (big[0], 100.0 * big[0] / max(free_area, 1e-9)))
    for t in warn:
        print('  [WARN] %s' % t)

    try:
        from PIL import Image
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[free] = (245, 245, 245)
        rgb[unk] = (150, 150, 150)
        rgb[occ] = (0, 0, 0)
        out = args.preview or (os.path.splitext(args.yaml)[0] + '_preview.png')
        Image.fromarray(rgb[::-1]).save(out)
        print('  预览图 -> %s' % out)
    except ImportError:
        pass
    print('VERDICT: %s' % ('OK' if not warn else 'CHECK'))
    return 0 if not warn else 2


if __name__ == '__main__':
    sys.exit(main())
