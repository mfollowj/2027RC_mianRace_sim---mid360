#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""点云落盘：三种用法

1) **一次性抓一帧**（用于 /Laser_map 这种"一帧就是整张地图"的话题）
       python3 save_cloud_pcd.py --topic /Laser_map -o maps/prior_map.pcd

2) **边建图边攒地图，收到触发再落盘**（真机推荐的用法）
       python3 save_cloud_pcd.py --topic /cloud_registered -o maps/prior_map.pcd \
           --accumulate --trigger-topic /save_prior_map
   它在整个建图过程中持续把 /cloud_registered 累加成一个**体素去重**的全局地图
   （内存只跟地图大小有关，不跟时间长短有关），收到 /save_prior_map 上的一条
   std_msgs/Empty 就写盘；Ctrl-C 也会写一次。
   ★ 为什么必须"边跑边攒"：-cloud_registered 只有当前视野，跑完再抓只能拿到
   最后一小片；而 /Laser_map 只在 FAST-LIVO2 的"先验图跟踪"模式下才发布。

3) 短时抓取（调试用）：--accumulate --duration 30
"""
import argparse
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2


def write_pcd(path, xyz, intensity=None):
    n = xyz.shape[0]
    fields = ['x', 'y', 'z'] + (['intensity'] if intensity is not None else [])
    hdr = ['# .PCD v0.7 - Point Cloud Data file format', 'VERSION 0.7',
           'FIELDS ' + ' '.join(fields),
           'SIZE ' + ' '.join(['4'] * len(fields)),
           'TYPE ' + ' '.join(['F'] * len(fields)),
           'COUNT ' + ' '.join(['1'] * len(fields)),
           'WIDTH %d' % n, 'HEIGHT 1', 'VIEWPOINT 0 0 0 1 0 0 0',
           'POINTS %d' % n, 'DATA binary']
    dt = np.dtype([(f, '<f4') for f in fields])
    arr = np.zeros(n, dtype=dt)
    arr['x'], arr['y'], arr['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if intensity is not None:
        arr['intensity'] = intensity
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(('\n'.join(hdr) + '\n').encode('ascii'))
        f.write(arr.tobytes())
    print('写好 %s：%d 点，字段 %s' % (path, n, ','.join(fields)))


class Saver(Node):
    def __init__(self, args):
        super().__init__('save_cloud_pcd')
        self.args = args
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
        self.last = None
        self.acc = []
        self.map = np.zeros((0, 4))     # 体素去重后的全局地图
        self.n_msg = 0
        self.n_pts = 0
        self.last_flush = time.time()
        self.sub = self.create_subscription(PointCloud2, args.topic, self.cb, qos)
        # 把累积中的地图实时发出去，RViz 里就能看着地图长大（否则只能看到当前视野）
        self.pub_map = None
        self.last_pub_map = 0.0
        if args.trigger_topic:
            from std_msgs.msg import Empty
            self.create_subscription(Empty, args.trigger_topic, self.on_trigger, 10)
            if args.pub_topic:
                latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
                self.pub_map = self.create_publisher(PointCloud2, args.pub_topic, latched)
            self.get_logger().info(
                '累积模式：%s -> %s；收到 %s 上的空消息就写盘（Ctrl-C 也会写）；'
                '地图实时发布到 %s'
                % (args.topic, args.out, args.trigger_topic, args.pub_topic or '(关闭)'))
        self.get_logger().info('等 %s 的点云 ...' % args.topic)

    # ------------------------------------------------------------------
    def publish_map(self, force=False):
        if self.pub_map is None or len(self.map) == 0:
            return
        now = time.time()
        if not force and now - self.last_pub_map < 1.0 / max(self.args.pub_rate, 0.1):
            return
        self.last_pub_map = now
        m = PointCloud2()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.args.map_frame
        m.height = 1
        m.width = int(len(self.map))
        m.fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)]
        m.is_bigendian = False
        m.point_step = 16
        m.row_step = 16 * m.width
        m.is_dense = True
        m.data = np.ascontiguousarray(self.map[:, :4], dtype=np.float32).tobytes()
        self.pub_map.publish(m)

    def on_trigger(self, _msg):
        self.get_logger().info('收到落盘触发，写 %s' % self.args.out)
        self.dump()

    def flush(self, force=False):
        """把缓冲区并进去重地图。分批合并，内存只随地图大小增长。"""
        if not self.acc:
            return
        pts = np.concatenate(self.acc, axis=0)
        self.acc = []
        allp = np.concatenate([self.map, pts], axis=0) if len(self.map) else pts
        vox = self.args.voxel if self.args.voxel > 0 else 0.05
        key = np.floor(allp[:, :3] / vox).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        self.map = allp[np.sort(idx)]
        self.last_flush = time.time()
        if self.args.verbose:
            self.get_logger().info('累积：%d 帧 / %d 点 -> 去重后 %d 点'
                                   % (self.n_msg, self.n_pts, len(self.map)))

    def cb(self, msg):
        names = [f.name for f in msg.fields]
        if not all(k in names for k in ('x', 'y', 'z')):
            return
        want = ['x', 'y', 'z'] + (['intensity'] if 'intensity' in names else [])
        pts = np.array([[float(p[0]), float(p[1]), float(p[2]), float(p[3]) if len(p) > 3 else 0.0]
                        for p in pc2.read_points(msg, field_names=want, skip_nans=True)],
                       dtype=np.float64)
        if pts.size == 0:
            return
        self.n_msg += 1
        self.n_pts += pts.shape[0]
        self.last = pts
        if self.args.accumulate:
            self.acc.append(pts)
            if time.time() - self.last_flush >= self.args.flush_sec:
                self.flush()
                self.publish_map()

    def dump(self):
        if self.args.accumulate:
            self.flush(force=True)
            pts = self.map                 # flush 里已经按体素去过重
            print('累积了 %d 帧 / %d 个原始点 -> 去重后 %d 点'
                  % (self.n_msg, self.n_pts, len(pts)))
        elif self.last is not None:
            pts = self.last
        else:
            print('没有收到任何点云', file=sys.stderr)
            return 1
        if len(pts) == 0:
            print('没有收到任何点云', file=sys.stderr)
            return 1
        if self.args.voxel > 0 and not self.args.accumulate:
            key = np.floor(pts[:, :3] / self.args.voxel).astype(np.int64)
            _, idx = np.unique(key, axis=0, return_index=True)
            pts = pts[np.sort(idx)]
            print('体素下采样 %.2f m -> %d 点' % (self.args.voxel, pts.shape[0]))
        write_pcd(self.args.out, pts[:, :3], pts[:, 3])
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/Laser_map')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--accumulate', action='store_true')
    ap.add_argument('--duration', type=float, default=5.0,
                    help='只收最后一帧时，收到首帧后再多等这么久就写盘')
    ap.add_argument('--voxel', type=float, default=0.05,
                    help='累积模式下的去重体素（默认 0.05 m）；非累积模式则当降采样用')
    ap.add_argument('--trigger-topic', default='',
                    help='累积模式下监听这个话题（std_msgs/Empty），收到就写盘')
    ap.add_argument('--flush-sec', type=float, default=5.0,
                    help='累积模式下每隔多久把缓冲区并进地图')
    ap.add_argument('--pub-topic', default='/prior_map_accum',
                    help='把累积中的地图实时发到这个话题（RViz 看建图过程），空则不发')
    ap.add_argument('--pub-rate', type=float, default=1.0)
    ap.add_argument('--map-frame', default='odom',
                    help='地图话题的 frame（第一阶段就是 odom）')
    ap.add_argument('--min-points', type=int, default=1000)
    ap.add_argument('--verbose', action='store_true')
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = Saver(args)

    if args.trigger_topic:
        # 累积模式：一直跑，等触发或 Ctrl-C
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.publish_map(force=True)
        rc = node.dump()
        node.destroy_node()
        rclpy.shutdown()
        return rc

    t0 = time.time()
    first = None
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.last is not None:
            if first is None:
                first = time.time()
            if node.n_msg >= args.min_points or time.time() - first > args.duration:
                break
        if not args.accumulate and time.time() - t0 > 60.0:
            break
    rc = node.dump()
    node.destroy_node()
    rclpy.shutdown()
    return rc


if __name__ == '__main__':
    sys.exit(main())
