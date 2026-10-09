#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把一帧 PointCloud2（默认取 3D 建图的累积地图 /Laser_map）存成 .pcd 文件。

不依赖 pcl_ros / python-pcl：直接按 PCD 规范写二进制文件（x y z intensity）。

用法
----
    ros2 run rc_field_nav save_pointcloud_pcd.py                    # /Laser_map -> field_map.pcd
    ros2 run rc_field_nav save_pointcloud_pcd.py -o /tmp/my.pcd
    ros2 run rc_field_nav save_pointcloud_pcd.py -t /cloud_registered
    ros2 run rc_field_nav save_pointcloud_pcd.py --ascii            # 存 ASCII 便于查看

提示：LIO 的 /Laser_map 是 **odom 帧下的累积地图**，点云已配准好，可直接给
CloudCompare / Open3D / pcl_viewer 打开。
"""
import argparse
import struct
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class Collector(Node):
    def __init__(self, topic, timeout):
        super().__init__('pcd_saver')
        self.msg = None
        self.biggest = 0
        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(PointCloud2, topic, self.cb, qos)
        self.deadline = time.time() + timeout

    def cb(self, msg):
        # 累积地图会不断长大，取收到的最大的那一帧
        n = msg.width * msg.height
        if n >= self.biggest:
            self.biggest = n
            self.msg = msg

    def spin_until_done(self):
        while rclpy.ok() and time.time() < self.deadline and self.msg is None:
            rclpy.spin_once(self, timeout_sec=0.2)
        # 再多收一会儿，尽量拿到更大的地图
        while rclpy.ok() and time.time() < self.deadline:
            rclpy.spin_once(self, timeout_sec=0.2)


def write_pcd(msg, path, ascii_mode=False):
    fields = {f.name: f for f in msg.fields}
    if 'x' not in fields or 'y' not in fields or 'z' not in fields:
        raise RuntimeError('点云里没有 x/y/z 字段，无法保存')
    has_i = 'intensity' in fields
    header = ['# .PCD v0.7 - Point Cloud Data file format',
              'VERSION 0.7',
              'FIELDS x y z' + (' intensity' if has_i else ''),
              'SIZE 4 4 4' + (' 4' if has_i else ''),
              'TYPE F F F' + (' F' if has_i else ''),
              'COUNT 1 1 1' + (' 1' if has_i else ''),
              'WIDTH %d' % (msg.width * msg.height),
              'HEIGHT 1',
              'VIEWPOINT 0 0 0 1 0 0 0',
              'POINTS %d' % (msg.width * msg.height),
              'DATA %s' % ('ascii' if ascii_mode else 'binary'),
              '']
    n = 0
    with open(path, 'wb') as f:
        f.write(('\n'.join(header)).encode('ascii'))
        for p in point_cloud2.read_points(msg, field_names=('x', 'y', 'z') +
                                          (('intensity',) if has_i else ()),
                                          skip_nans=True):
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            if ascii_mode:
                if has_i:
                    f.write(('%.4f %.4f %.4f %.4f\n' % (x, y, z, float(p[3]))).encode())
                else:
                    f.write(('%.4f %.4f %.4f\n' % (x, y, z)).encode())
            else:
                if has_i:
                    f.write(struct.pack('<ffff', x, y, z, float(p[3])))
                else:
                    f.write(struct.pack('<fff', x, y, z))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description='PointCloud2 -> .pcd')
    ap.add_argument('-t', '--topic', default='/Laser_map', help='点云话题（默认 LIO 累积地图）')
    ap.add_argument('-o', '--output', default='/home/zx/ITEM/RC_ground_sim/field_map.pcd')
    ap.add_argument('--timeout', type=float, default=25.0, help='等待点云的秒数')
    ap.add_argument('--ascii', action='store_true', help='存 ASCII 格式')
    args = ap.parse_args()

    rclpy.init()
    node = Collector(args.topic, args.timeout)
    print('[INFO] 等待点云 %s（最多 %.0fs，取点数最多的一帧）...' % (args.topic, args.timeout))
    node.spin_until_done()
    if node.msg is None:
        print('[ERROR] %.0fs 内没收到 %s 的任何点云。'
              '请确认 3D 建图在跑（lio_map.launch.py）且已经开动过。' % (args.timeout, args.topic))
        rclpy.shutdown()
        return 1
    n = write_pcd(node.msg, args.output, args.ascii)
    print('[INFO] 已保存 %d 个点 -> %s（frame: %s）' % (n, args.output, node.msg.header.frame_id))
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
