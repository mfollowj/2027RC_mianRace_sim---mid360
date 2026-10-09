#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实轮速驱动节点（配合 drive_mode:=physics）

职责：
  1) 逆运动学：/cmd_vel (vx, vy, ω) → 4 个全向轮的角速度，发到
     /wheel_velocity_controller/commands (std_msgs/Float64MultiArray)；
  2) 正运动学：读 /joint_states 里 4 个轮子的实测角速度 → 最小二乘解出车体速度
     → 积分出 odom，发布 /odom 与 TF odom->base_link。

轮系（与 urdf 一致，X 型布局）：
    φ_i = atan2(±0.16, ±0.22) = ±36.03° / 180°∓36.03°
    r   = hypot(0.22, 0.16) = 0.2720 m
    v_i = -sin(φ_i)·vx + cos(φ_i)·vy + r·ω        ω_wheel_i = v_i / R_wheel

这样车体运动是"轮子转 + 轮-地摩擦"真实涌现出来的，而不是插件直接设定车体速度。
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster

WHEEL_JOINTS = ['base_to_wheel1', 'base_to_wheel2', 'base_to_wheel3', 'base_to_wheel4']


def _as_bool(v):
    # launch_ros 参数字典里写 'false'（字符串）时 rclpy 会原样存成字符串，
    # 而 bool('false') 在 Python 里是 True；这类假 false 很难查，统一在这里兜住。
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(v)


class OmniDrive(Node):
    def __init__(self):
        super().__init__('omni_drive')
        # ---- 参数 ----
        self.wheel_radius = self.declare_parameter('wheel_radius', 0.10).value
        wx = self.declare_parameter('wheel_x', 0.22).value
        wy = self.declare_parameter('wheel_y', 0.16).value
        rate = self.declare_parameter('rate', 20.0).value
        self.cmd_topic = self.declare_parameter('cmd_topic', '/cmd_vel').value
        self.wheel_cmd_topic = self.declare_parameter(
            'wheel_cmd_topic', '/wheel_velocity_controller/commands').value
        self.odom_topic = self.declare_parameter('odom_topic', '/odom').value
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        # launch 用字符串传进来的布尔参数（'false'/'true'）不能直接当 bool 用：
        #   Python 里非空字符串恒为 True，会导致 publish_odom_tf:=false 仍然发
        #   odom->base_link，与 FAST-LIO 抢同一个 TF 父节点（实测踩过）。
        self.publish_odom = _as_bool(self.declare_parameter('publish_odom', True).value)
        self.publish_odom_tf = _as_bool(
            self.declare_parameter('publish_odom_tf', True).value)
        self.max_wheel_speed = self.declare_parameter('max_wheel_speed', 40.0).value

        # ---- 轮系矩阵 ----
        self.phi = [math.atan2(wy, wx), math.atan2(wy, -wx),
                    math.atan2(-wy, -wx), math.atan2(-wy, wx)]
        self.r_center = math.hypot(wx, wy)
        self.A = np.array([[-math.sin(p), math.cos(p), self.r_center]
                           for p in self.phi])

        self.cmd = Twist()
        self.wheel_vel = [0.0] * 4          # 实测轮速 (rad/s)
        self.joint_index = {}
        self.x = self.y = self.yaw = 0.0
        self.last_stamp = None

        best_effort = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Twist, self.cmd_topic, self.on_cmd, 10)
        self.create_subscription(JointState, '/joint_states', self.on_joint, best_effort)
        self.wheel_pub = self.create_publisher(Float64MultiArray, self.wheel_cmd_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.tf_br = TransformBroadcaster(self)
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info(
            'omni_drive 启动：R=%.3f r=%.4f φ=%s | %s -> %s，odom_tf=%s'
            % (self.wheel_radius, self.r_center,
               ['%.1f°' % math.degrees(p) for p in self.phi],
               self.cmd_topic, self.wheel_cmd_topic, self.publish_odom_tf))

    # ---------------- 逆运动学：车体速度 -> 轮速 ----------------
    def inverse(self, vx, vy, wz):
        v_surface = self.A @ np.array([vx, vy, wz])
        w = v_surface / self.wheel_radius
        return np.clip(w, -self.max_wheel_speed, self.max_wheel_speed)

    def on_cmd(self, msg):
        self.cmd = msg

    def tick(self):
        w = self.inverse(self.cmd.linear.x, self.cmd.linear.y, self.cmd.angular.z)
        out = Float64MultiArray()
        out.data = [float(v) for v in w]
        self.wheel_pub.publish(out)

    # ---------------- 正运动学：轮速 -> odom ----------------
    def on_joint(self, msg):
        if not self.joint_index:
            for i, name in enumerate(WHEEL_JOINTS):
                if name in msg.name:
                    self.joint_index[i] = msg.name.index(name)
            if len(self.joint_index) != 4:
                return
        new_vel = list(self.wheel_vel)
        for i, idx in self.joint_index.items():
            if idx < len(msg.velocity):
                new_vel[i] = msg.velocity[idx]

        stamp = msg.header.stamp
        if self.last_stamp is not None:
            dt = (stamp.sec - self.last_stamp.sec) + (stamp.nanosec - self.last_stamp.nanosec) * 1e-9
            if 0.0 < dt < 0.5:
                v_surface = np.array(new_vel) * self.wheel_radius
                sol, *_ = np.linalg.lstsq(self.A, v_surface, rcond=None)
                vx, vy, wz = (float(sol[0]), float(sol[1]), float(sol[2]))
                # 车体系 -> odom 系
                cy, sy = math.cos(self.yaw), math.sin(self.yaw)
                self.x += (vx * cy - vy * sy) * dt
                self.y += (vx * sy + vy * cy) * dt
                self.yaw += wz * dt
                self.publish(stamp, vx, vy, wz)
        self.wheel_vel = new_vel
        self.last_stamp = stamp

    def publish(self, stamp, vx, vy, wz):
        qz, qw = math.sin(self.yaw / 2.0), math.cos(self.yaw / 2.0)
        if self.publish_odom:
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = self.odom_frame
            odom.child_frame_id = self.base_frame
            odom.pose.pose.position.x = self.x
            odom.pose.pose.position.y = self.y
            odom.pose.pose.orientation.z = qz
            odom.pose.pose.orientation.w = qw
            odom.twist.twist.linear.x = vx
            odom.twist.twist.linear.y = vy
            odom.twist.twist.angular.z = wz
            self.odom_pub.publish(odom)
        if self.publish_odom_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_br.sendTransform(t)


def main():
    rclpy.init()
    node = OmniDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 退出前把轮速清零，避免车继续跑
        try:
            out = Float64MultiArray()
            out.data = [0.0, 0.0, 0.0, 0.0]
            node.wheel_pub.publish(out)
        except Exception:                      # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
