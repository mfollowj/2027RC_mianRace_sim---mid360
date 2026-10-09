#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nav2 bringup 守护：起 navigation_launch.py，并保证它**真的**激活。

为什么需要它（实测踩的坑）：
  本机只有 Fast DDS（没装 CycloneDDS）。实测在"仿真 + FAST-LIO + 定位器"同时在跑的时候，
  nav2_lifecycle_manager 在连续切换 7~8 个节点的生命周期时，偶发某一次
  change_state 的**响应发不出去**：
      [velocity_smoother.rclcpp] failed to send response to
      /velocity_smoother/change_state (timeout): client will not receive response
  服务端写响应超时（Fast DDS 写阻塞 ~100 ms），客户端于是**永远等不到响应**，
  lifecycle_manager 就停在 "Configuring velocity_smoother" 不动了：
  bt_navigator 一直是 inactive，Nav2 表面"起来了"但根本不能用。
  这个故障是间歇的（同一套配置 4 次里成功 2 次），且 nav2_lifecycle_manager 自己
  不会重试（失败后只是抛异常结束，再调 STARTUP 也会因为部分节点已是 active 而失败）。

  所以这里做一个**进程级守护**：把 navigation_launch.py 当子进程起，
  轮询 bt_navigator 的生命周期状态；超时没激活就整轮杀掉重启，直到成功为止。
  成功之后继续盯着，节点掉线/被 kill 也会自动拉起来。

    ros2 run rc_field_nav nav2_supervisor.py --timeout 75 --retries 4 -- \
        ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true ...
"""
import argparse
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from lifecycle_msgs.srv import GetState


class Supervisor(Node):
    def __init__(self, cmd, node_name, timeout, retries, watch_timeout, retry_delay):
        super().__init__('nav2_supervisor')
        self.cmd = cmd
        self.node_name = node_name
        self.timeout = timeout
        self.retries = retries
        self.watch_timeout = watch_timeout
        self.retry_delay = retry_delay
        self.child = None
        self.cli = None
        self.done = False

    # ---------- 子进程 ----------
    def start_child(self, attempt):
        self.get_logger().info('第 %d 次启动 Nav2 bringup: %s' % (attempt, ' '.join(self.cmd)))
        self.child = subprocess.Popen(self.cmd)
        self.cli = self.create_client(GetState, '%s/get_state' % self.node_name)

    def stop_child(self):
        if self.child is None:
            return
        if self.child.poll() is None:
            self.get_logger().warn('停止未成功激活的 Nav2（pid=%d）' % self.child.pid)
            self.child.terminate()
            for _ in range(50):
                if self.child.poll() is not None:
                    break
                time.sleep(0.1)
            if self.child.poll() is None:
                self.child.kill()
                time.sleep(1.0)
        self.child = None

    # ---------- 状态查询 ----------
    def active(self):
        if self.cli is None:
            return False
        if not self.cli.wait_for_service(timeout_sec=1.0):
            return False
        fut = self.cli.call_async(GetState.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        res = fut.result()
        return res is not None and res.current_state.label == 'active'

    # ---------- 主流程 ----------
    def run(self):
        for attempt in range(1, self.retries + 1):
            self.start_child(attempt)
            t0 = time.time()
            while time.time() - t0 < self.timeout:
                if self.child.poll() is not None:
                    self.get_logger().error(
                        'Nav2 bringup 进程自己退出了（exit=%s），准备重启' % self.child.returncode)
                    break
                if self.active():
                    self.get_logger().info(
                        '★ Nav2 已激活（第 %d 次尝试，用时 %.1f s）' % (attempt, time.time() - t0))
                    return True
                time.sleep(2.0)
            else:
                self.get_logger().error(
                    '%.0f s 内 %s 没有激活 —— 判定本轮 bringup 卡死（多半是 '
                    'lifecycle change_state 丢响应），重启' % (self.timeout, self.node_name))
            self.stop_child()
            # 客户端句柄要跟着作废，否则会连到旧服务上
            if self.cli is not None:
                self.destroy_client(self.cli)
                self.cli = None
            time.sleep(self.retry_delay)

        self.get_logger().error('重试 %d 次仍未把 Nav2 拉起来，放弃。' % self.retries)
        return False

    def supervise(self):
        """激活之后继续盯着：节点掉线超过 watch_timeout 就整轮重启。"""
        if self.watch_timeout <= 0:
            return
        bad_since = None
        while not self.done:
            time.sleep(2.0)
            if self.child is not None and self.child.poll() is not None:
                self.get_logger().error('Nav2 进程退出（exit=%s），重启' % self.child.returncode)
                break
            if self.active():
                bad_since = None
            else:
                if bad_since is None:
                    bad_since = time.time()
                elif time.time() - bad_since > self.watch_timeout:
                    self.get_logger().error(
                        '%s 掉线超过 %.0f s，重启 Nav2 bringup' % (self.node_name, self.watch_timeout))
                    break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--node', default='bt_navigator')
    ap.add_argument('--timeout', type=float, default=75.0,
                    help='单轮等待激活的最长时间（秒），超时则整轮重启')
    ap.add_argument('--retries', type=int, default=4)
    ap.add_argument('--retry-delay', type=float, default=5.0)
    ap.add_argument('--watch-timeout', type=float, default=30.0,
                    help='激活后掉线多久就重启；0=激活后不再管')
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = [c for c in args.cmd if c != '--']
    if not cmd:
        print('用法：nav2_supervisor.py [选项] -- <要守护的命令...>')
        return 2

    rclpy.init()
    sup = Supervisor(cmd, args.node, args.timeout, args.retries,
                     args.watch_timeout, args.retry_delay)

    def on_signal(_sig, _frm):
        sup.done = True
        sup.stop_child()
        sup.get_logger().info('收到退出信号，守护结束')
        rclpy.shutdown()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    ok = False
    try:
        while not sup.done:
            ok = sup.run()
            if not ok or sup.watch_timeout <= 0:
                break
            sup.supervise()
            if sup.done:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sup.stop_child()
        if rclpy.ok():
            sup.destroy_node()
            rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
