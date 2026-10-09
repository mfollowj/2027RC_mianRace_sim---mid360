#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全向底盘键盘遥控（支持横向平移 + 脚本化回放）—— 建图时用它开车

为什么不用 teleop_twist_keyboard：那个节点只有 linear.x / angular.z，
**没有 linear.y**，全向底盘最关键的横向平移按不出来。本节点三轴齐全。

用法
----
交互遥控（在终端里按键，看的到实时状态）：
    ros2 run rc_field_nav omni_teleop.py
    ros2 run rc_field_nav omni_teleop.py --speed 0.4 --yaw-speed 0.8

脚本化回放（不需要终端，可用于自动化建图/自测）：
    ros2 run rc_field_nav omni_teleop.py --seq "w:3.0 d:2.0 w+q:2.0 stop:1.0"
    语法：  <按键组合>:<秒数>   组合用 + 连接，stop 表示全停、s+ 等不允许
            可用的键：w s a d q e（同交互模式的键位）

按键（交互模式，按下即生效并"锁存"，按空格急停）：
    w / s   前进 / 后退        (linear.x ±)
    a / d   左移 / 右移        (linear.y ±)  ← 全向独有
    q / e   左转 / 右转        (angular.z ±)
    空格    全部停止
    + / -   加减速度档（线性 0.05 m/s 一档，角速度 0.1 rad/s 一档）
    z       三轴归零并打印当前状态
    t       退出（等价 Ctrl-C）

安全提示：键盘遥控是"锁存"式的（松手不会停），请用空格急停；
         可用 --auto-stop 3.0 让它在 3 秒无按键后自动停下。
"""
import argparse
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist

HELP = """
================= 全向底盘键盘遥控 =================
  w / s : 前进 / 后退        a / d : 左移 / 右移
  q / e : 左转 / 右转        空格  : 急停（三轴清零）
  + / - : 加减速档           z     : 归零并显示状态
  t     : 退出
提示：按键为锁存式（松手不停），请用空格急停。
===================================================
"""

VX_KEYS = {'w': +1.0, 's': -1.0}
VY_KEYS = {'a': +1.0, 'd': -1.0}
WZ_KEYS = {'q': +1.0, 'e': -1.0}


class OmniTeleop(Node):
    def __init__(self, speed, yaw_speed, rate, topic):
        super().__init__('omni_teleop')
        self.pub = self.create_publisher(Twist, topic, 10)
        self.speed = speed
        self.yaw_speed = yaw_speed
        self.vx = self.vy = self.wz = 0.0
        self.last_key_time = time.time()
        self.create_timer(1.0 / rate, self.tick)

    def tick(self):
        t = Twist()
        t.linear.x = self.vx
        t.linear.y = self.vy
        t.angular.z = self.wz
        self.pub.publish(t)

    def stop(self):
        self.vx = self.vy = self.wz = 0.0

    def status(self):
        return ('vx=%+.2f vy=%+.2f wz=%+.2f  (档位 线%.2f 角%.2f)'
                % (self.vx, self.vy, self.wz, self.speed, self.yaw_speed))

    def apply_key(self, ch):
        """返回 False 表示要退出。"""
        if ch in VX_KEYS:
            self.vx = VX_KEYS[ch] * self.speed
        elif ch in VY_KEYS:
            self.vy = VY_KEYS[ch] * self.speed
        elif ch in WZ_KEYS:
            self.wz = WZ_KEYS[ch] * self.yaw_speed
        elif ch == ' ':
            self.stop()
        elif ch == '+' or ch == '=':
            self.speed = min(2.0, self.speed + 0.05)
            self.yaw_speed = min(3.0, self.yaw_speed + 0.1)
        elif ch == '-' or ch == '_':
            self.speed = max(0.05, self.speed - 0.05)
            self.yaw_speed = max(0.1, self.yaw_speed - 0.1)
        elif ch == 'z':
            self.stop()
        elif ch in ('t', '\x03'):     # t 或 Ctrl-C
            self.stop()
            return False
        self.last_key_time = time.time()
        return True


def run_sequence(node, seq, verbose=True):
    """按 "键组合:秒数" 脚本回放，返回 0。"""
    total = 0.0
    for token in seq.split():
        if ':' not in token:
            print('[WARN] 忽略无法解析的片段: %r' % token)
            continue
        keys, dur = token.split(':', 1)
        dur = float(dur)
        node.stop()
        if keys.lower() != 'stop':
            for ch in keys:
                if ch == '+':
                    node.apply_key('+')
                elif ch in VX_KEYS or ch in VY_KEYS or ch in WZ_KEYS:
                    node.apply_key(ch)
                else:
                    print('[WARN] 未知按键 %r（可用 w s a d q e）' % ch)
        if verbose:
            print('  [SEQ] %-12s %.1fs -> %s' % (token, dur, node.status()))
        end = time.time() + dur
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.02)
        total += dur
    node.stop()
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.02)
    if verbose:
        print('  [SEQ] 回放结束，共 %.1fs，已停车' % total)
    return 0


def run_keyboard(node, auto_stop):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(HELP)
    print('初始状态: %s' % node.status())
    try:
        tty.setcbreak(fd)
        while True:
            rclpy.spin_once(node, timeout_sec=0.02)
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                if ch == '':
                    # stdin 到达 EOF（例如被管道/脚本驱动）：直接退出，避免忙等
                    break
                if not node.apply_key(ch):
                    break
                sys.stdout.write('\r' + node.status() + '        ')
                sys.stdout.flush()
            if auto_stop > 0 and (time.time() - node.last_key_time) > auto_stop \
                    and (node.vx or node.vy or node.wz):
                node.stop()
                sys.stdout.write('\r[自动停车] ' + node.status() + '        ')
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.stop()
        for _ in range(30):
            rclpy.spin_once(node, timeout_sec=0.02)
        print('\n[INFO] 已退出，速度已清零')
    return 0


def main():
    ap = argparse.ArgumentParser(description='全向底盘键盘遥控')
    ap.add_argument('--speed', type=float, default=0.30, help='线速度档位 m/s（默认 0.30）')
    ap.add_argument('--yaw-speed', type=float, default=0.8, help='角速度档位 rad/s')
    ap.add_argument('--rate', type=float, default=20.0, help='发布频率 Hz')
    ap.add_argument('--topic', default='/cmd_vel', help='速度话题')
    ap.add_argument('--seq', default='', help='脚本回放，如 "w:3.0 d:2.0 stop:1.0"')
    ap.add_argument('--auto-stop', type=float, default=0.0,
                    help='无按键超过该秒数自动停车（0=关闭）')
    args = ap.parse_args()

    rclpy.init()
    node = OmniTeleop(args.speed, args.yaw_speed, args.rate, args.topic)
    try:
        if args.seq:
            print('[INFO] 脚本回放模式: %s' % args.seq)
            return run_sequence(node, args.seq)
        if not sys.stdin.isatty():
            print('[ERROR] 当前不是终端（无 tty），交互遥控需要真实终端；'
                  '自动化请用 --seq "w:3.0 ..."')
            return 2
        return run_keyboard(node, args.auto_stop)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
