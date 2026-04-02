"""
Live camera viewer for Orbbec Gemini 335L via ROS2.
Shows color and depth images in an OpenCV window.

Run:
    source /opt/ros/humble/setup.bash
    source ~/ros2_ws/install/setup.bash
    cd /home/kklab-ur-robot/ur3e_cdf_mppi
    .venv/bin/python visual/camera_viewer.py
"""

from __future__ import annotations

import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


COLOR_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"
WINDOW_NAME = "Orbbec Camera Viewer"
SYSTEM_DIST_PACKAGES = "/usr/lib/python3/dist-packages"


def import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        if SYSTEM_DIST_PACKAGES not in sys.path:
            sys.path.append(SYSTEM_DIST_PACKAGES)
        import cv2
        return cv2


cv2 = import_cv2()


def image_msg_to_bgr(msg: Image) -> np.ndarray | None:
    if msg.encoding not in ("rgb8", "bgr8", "mono8"):
        return None

    img = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == "mono8":
        img = img.reshape(msg.height, msg.width)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = img.reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def depth_msg_to_bgr(msg: Image) -> np.ndarray | None:
    if msg.encoding == "16UC1":
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        depth_m = depth.astype(np.float32) / 1000.0
    elif msg.encoding == "32FC1":
        depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    else:
        return None

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if not np.any(valid):
        return np.zeros((msg.height, msg.width, 3), dtype=np.uint8)

    depth_vis = depth_m.copy()
    lo = float(np.percentile(depth_vis[valid], 5))
    hi = float(np.percentile(depth_vis[valid], 95))
    if hi <= lo:
        hi = lo + 1e-3

    depth_vis = np.clip((depth_vis - lo) / (hi - lo), 0.0, 1.0)
    depth_u8 = (depth_vis * 255.0).astype(np.uint8)
    depth_u8[~valid] = 0
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (340, 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


class CameraViewer(Node):
    def __init__(self):
        super().__init__("camera_viewer")
        self._lock = threading.Lock()
        self._color = None
        self._depth = None

        self.create_subscription(Image, COLOR_TOPIC, self._color_cb, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self._depth_cb, 10)
        self.get_logger().info(f"Subscribed to {COLOR_TOPIC}")
        self.get_logger().info(f"Subscribed to {DEPTH_TOPIC}")

    def _color_cb(self, msg: Image):
        img = image_msg_to_bgr(msg)
        if img is None:
            self.get_logger().warn(f"Unsupported color encoding: {msg.encoding}", throttle_duration_sec=5.0)
            return
        with self._lock:
            self._color = img

    def _depth_cb(self, msg: Image):
        img = depth_msg_to_bgr(msg)
        if img is None:
            self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}", throttle_duration_sec=5.0)
            return
        with self._lock:
            self._depth = img

    def get_frames(self):
        with self._lock:
            color = None if self._color is None else self._color.copy()
            depth = None if self._depth is None else self._depth.copy()
        return color, depth


def main():
    rclpy.init()
    node = CameraViewer()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            color, depth = node.get_frames()

            if color is None and depth is None:
                blank = np.zeros((480, 960, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for camera topics...", (40, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                canvas = blank
            else:
                panels = []
                if color is not None:
                    panels.append(add_label(color, "Color"))
                if depth is not None:
                    if color is not None and depth.shape[:2] != color.shape[:2]:
                        depth = cv2.resize(depth, (color.shape[1], color.shape[0]))
                    panels.append(add_label(depth, "Depth"))
                if len(panels) == 1:
                    canvas = panels[0]
                else:
                    canvas = np.hstack(panels)

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
