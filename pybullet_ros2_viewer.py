"""
Real-time PyBullet viewer synchronized with real UR3e robot via ROS2.
Subscribes to /follower/joint_state and /obstacles/base_frame,
and visualizes both robot and obstacles in PyBullet.

Run with:
    source /opt/ros/humble/setup.bash
    cd /home/kklab-ur-robot/ur3e_cdf_mppi
    .venv/bin/python pybullet_ros2_viewer.py
"""

import sys
import os
import threading
import time
import numpy as np

# Add project root to path so ur3e package is found
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

from ur3e.ur3e_pybullet import UR3ePyBullet


OBSTACLE_TOPIC = '/obstacles/base_frame'
OBSTACLE_RADIUS = 0.05
OBSTACLE_UPDATE_PERIOD = 0.5
WS_MIN = np.array([-0.4, -0.4, 0.0], dtype=np.float32)
WS_MAX = np.array([0.4, 0.4, 0.4], dtype=np.float32)


def xyz_from_cloud_msg(msg: PointCloud2) -> np.ndarray:
    gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    pts = np.array(list(gen), dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 3)
    return pts


class PyBulletViewer(Node):
    def __init__(self, robot: UR3ePyBullet):
        super().__init__('pybullet_viewer')
        self.robot = robot
        self.latest_joints = None
        self.latest_obstacles = np.zeros((0, 3), dtype=np.float32)
        self.lock = threading.Lock()
        self.received_count = 0
        self.obstacle_count = 0

        self.sub = self.create_subscription(
            JointState,
            '/follower/joint_state',
            self._joint_state_cb,
            10
        )
        self.obs_sub = self.create_subscription(
            PointCloud2,
            OBSTACLE_TOPIC,
            self._obstacle_cb,
            10
        )
        self.get_logger().info('Subscribed to /follower/joint_state')
        self.get_logger().info(f'Subscribed to {OBSTACLE_TOPIC}')

    def _joint_state_cb(self, msg: JointState):
        if len(msg.position) < 6:
            self.get_logger().warn(f'Got {len(msg.position)} joints, expected 6')
            return
        with self.lock:
            self.latest_joints = np.array(msg.position[:6])
            self.received_count += 1

    def _obstacle_cb(self, msg: PointCloud2):
        pts = xyz_from_cloud_msg(msg)
        with self.lock:
            self.latest_obstacles = pts
            self.obstacle_count = len(pts)

    def get_latest_joints(self):
        with self.lock:
            return self.latest_joints.copy() if self.latest_joints is not None else None

    def get_latest_obstacles(self):
        with self.lock:
            return self.latest_obstacles.copy()


def main():
    print('=' * 60)
    print('UR3e PyBullet Real-time Viewer')
    print('Topics: /follower/joint_state, /obstacles/base_frame')
    print('Press Ctrl+C to exit')
    print('=' * 60)

    # Initialize PyBullet with GUI
    print('Initializing PyBullet GUI...')
    try:
        device = 'cuda'
        import torch
        if not torch.cuda.is_available():
            device = 'cpu'
            print('CUDA not available, using CPU')
        robot = UR3ePyBullet(device=device, gui=True)
    except Exception as e:
        print(f'Failed to create UR3ePyBullet: {e}')
        sys.exit(1)

    # Initialize ROS2
    rclpy.init()
    node = PyBulletViewer(robot)

    # Spin ROS2 in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print('Waiting for joint states...')

    import pybullet as p
    last_print_time = time.time()
    last_count = 0
    last_obstacle_update = 0.0
    obstacle_body_ids = []
    obstacle_shape_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=OBSTACLE_RADIUS,
        rgbaColor=[1.0, 0.2, 0.2, 0.35],
    )

    def draw_workspace():
        corners = np.array([
            [x, y, z]
            for x in [WS_MIN[0], WS_MAX[0]]
            for y in [WS_MIN[1], WS_MAX[1]]
            for z in [WS_MIN[2], WS_MAX[2]]
        ], dtype=np.float32)
        edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        ]
        for i, j in edges:
            p.addUserDebugLine(
                corners[i].tolist(),
                corners[j].tolist(),
                lineColorRGB=[0.2, 0.6, 1.0],
                lineWidth=2.0,
                lifeTime=0,
            )
        p.addUserDebugText(
            "CDF workspace",
            [WS_MAX[0], WS_MAX[1], WS_MAX[2] + 0.03],
            textColorRGB=[0.2, 0.6, 1.0],
            textSize=1.2,
            lifeTime=0,
        )

    draw_workspace()

    def redraw_obstacles(pts: np.ndarray):
        nonlocal obstacle_body_ids
        for body_id in obstacle_body_ids:
            try:
                p.removeBody(body_id)
            except Exception:
                pass
        obstacle_body_ids = []

        for pt in pts:
            body_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=obstacle_shape_id,
                basePosition=pt.tolist(),
            )
            obstacle_body_ids.append(body_id)

    try:
        while True:
            joints = node.get_latest_joints()

            if joints is not None:
                robot.set_joint_angles(joints)
                p.stepSimulation()

                # Print joint angles every 2 seconds
                now = time.time()
                if now - last_obstacle_update >= OBSTACLE_UPDATE_PERIOD:
                    obstacles = node.get_latest_obstacles()
                    redraw_obstacles(obstacles)
                    last_obstacle_update = now

                if now - last_print_time >= 2.0:
                    hz = (node.received_count - last_count) / (now - last_print_time)
                    np.set_printoptions(precision=4, suppress=True)
                    print(f'[{hz:.1f} Hz] joints (rad): {joints}')
                    print(f'[{len(obstacle_body_ids)} obstacles] topic: {OBSTACLE_TOPIC}')
                    print(f'workspace: X[{WS_MIN[0]}, {WS_MAX[0]}] Y[{WS_MIN[1]}, {WS_MAX[1]}] Z[{WS_MIN[2]}, {WS_MAX[2]}]')
                    last_print_time = now
                    last_count = node.received_count
            else:
                # Still waiting
                time.sleep(0.1)
                continue

            time.sleep(1.0 / 100.0)  # 100 Hz update loop

    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        robot.disconnect()


if __name__ == '__main__':
    main()
