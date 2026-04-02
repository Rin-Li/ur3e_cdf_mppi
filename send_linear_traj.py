"""
Linear joint interpolation sender using servoJ.
Sends full trajectory to /target/multi_joint_trajectory.

Run:
    source /opt/ros/humble/setup.bash
    cd /home/kklab-ur-robot/ur3e_cdf_mppi
    .venv/bin/python send_linear_traj.py
"""

import numpy as np
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# ── trajectory params ──────────────────────────────────────
START = np.array([ -1.57, -0.5,  1.2, -1.3, -1.57, 0.0], dtype=np.float64)
GOAL  = np.array([ 1.57, -0.5,  1.2, -1.3, -1.57, 0.0], dtype=np.float64)

DT          = 0.002   # seconds per point (must match servo_j node)
DURATION    = 5.0     # total motion time in seconds
# ───────────────────────────────────────────────────────────


def build_trajectory(start, goal, dt, duration):
    n_points = int(duration / dt)
    alphas = np.linspace(0.0, 1.0, n_points)

    msg = JointTrajectory()
    msg.joint_names = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']

    for i, a in enumerate(alphas):
        q = (1.0 - a) * start + a * goal
        pt = JointTrajectoryPoint()
        pt.positions = q.tolist()
        t_ns = int(i * dt * 1e9)
        pt.time_from_start = Duration(sec=t_ns // 10**9, nanosec=t_ns % 10**9)
        msg.points.append(pt)

    return msg


class TrajSender(Node):
    def __init__(self):
        super().__init__('traj_sender')
        self.pub = self.create_publisher(
            JointTrajectory,
            '/target/multi_joint_trajectory',
            10
        )

    def send(self, msg):
        # Wait for subscriber (servo_j node)
        self.get_logger().info('Waiting for servo_j_to_follower_node...')
        while self.count_subscribers('/target/multi_joint_trajectory') == 0:
            rclpy.spin_once(self, timeout_sec=0.5)

        self.get_logger().info(f'Sending {len(msg.points)} points (duration={DURATION}s, dt={DT}s)')
        self.pub.publish(msg)
        self.get_logger().info('Trajectory sent.')


def main():
    rclpy.init()
    node = TrajSender()

    traj = build_trajectory(START, GOAL, DT, DURATION)
    node.send(traj)

    rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
