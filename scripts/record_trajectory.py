#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Record a trajectory (x y z per line) from an odometry topic.

Used to feed map3d_to_costmap.py so that the 2D occupancy grid gets a proper
"observed free space" region by ray casting along the sensor path.

    ros2 run rc_field_nav record_trajectory.py -o /path/traj.txt
    ros2 run rc_field_nav record_trajectory.py -t /Odometry -o /path/traj.txt --rate 5
"""
import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry


class TrajRecorder(Node):
    def __init__(self, topic, path, min_step, rate):
        super().__init__('record_trajectory')
        self.path = path
        self.min_step = min_step
        self.f = open(path, 'w')
        self.f.write('# x y z  (recorded from %s)\n' % topic)
        self.last = None
        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, topic, self.cb, qos)
        self.create_timer(1.0 / rate, self.flush)
        self.n = 0

    def cb(self, m):
        p = m.pose.pose.position
        if self.last is not None:
            d = ((p.x - self.last[0]) ** 2 + (p.y - self.last[1]) ** 2) ** 0.5
            if d < self.min_step:
                return
        self.last = (p.x, p.y, p.z)
        self.f.write('%.4f %.4f %.4f\n' % (p.x, p.y, p.z))
        self.n += 1

    def flush(self):
        self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--topic', default='/Odometry')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--min-step', type=float, default=0.02,
                    help='只记录位移超过该值的新点（米）')
    ap.add_argument('--rate', type=float, default=2.0, help='落盘频率 Hz')
    args, _ = ap.parse_known_args()

    rclpy.init()
    n = TrajRecorder(args.topic, args.out, args.min_step, args.rate)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.f.close()
        print('[record_trajectory] %d points -> %s' % (n.n, args.out))
        n.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
