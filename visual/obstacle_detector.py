"""Obstacle detector: RGBD color segmentation -> robot base frame point cloud.

Detects a colored cube obstacle using HSV color segmentation on the RGB image,
looks up depth for detected pixels, projects to 3D, transforms to robot base frame.

Setup (one-time):
  python3 visual/pick_color.py    # click on cube to save HSV range

Run:
  source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
  python3 visual/obstacle_detector.py
"""

import pathlib
import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge

_ROOT = pathlib.Path(__file__).parent.parent
CALIB_DIR = _ROOT / "visual" / "calib_data"
OBSTACLES_FILE = CALIB_DIR / "obstacles.npy"
HSV_FILE = CALIB_DIR / "cube_hsv.npz"

# Robot workspace limits (metres, base frame)
WS_X = (-0.4, 0.4)
WS_Y = (-0.4, 0.4)
WS_Z = (0.0,  0.4)

VOXEL_SIZE  = 0.02    # metres
DEPTH_SCALE = 0.001   # Orbbec depth in mm -> metres
MIN_PIXELS  = 50      # ignore detections smaller than this


def load_T_cam2base():
    npz_path = CALIB_DIR / "T_cam2base.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {npz_path}")
    d = np.load(npz_path)
    T = d["T_cam2base"].astype(np.float64)
    print(f"[obstacle_detector] Loaded T_cam2base")
    print(f"  Camera position (mm): {T[:3,3]*1000}")
    return T


def load_hsv_range():
    if not HSV_FILE.exists():
        print(f"[obstacle_detector] WARNING: {HSV_FILE} not found.")
        print("  Run python3 visual/pick_color.py first to calibrate cube color.")
        return None, None
    d = np.load(str(HSV_FILE))
    lo, hi = d["lo"], d["hi"]
    print(f"[obstacle_detector] Loaded HSV range: lo={lo}  hi={hi}")
    return lo, hi


def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(pts) == 0:
        return pts
    vox = np.floor(pts / voxel_size).astype(np.int32)
    _, idx = np.unique(vox, axis=0, return_index=True)
    return pts[idx]


def make_cloud_msg(pts: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    fields = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 12
    data = pts.astype(np.float32).tobytes()
    return PointCloud2(
        header=header, height=1, width=len(pts),
        is_dense=True, is_bigendian=False,
        fields=fields, point_step=point_step,
        row_step=point_step * len(pts), data=data,
    )


class ObstacleDetector(Node):
    def __init__(self, T_cam2base, hsv_lo, hsv_hi):
        super().__init__("obstacle_detector")
        self._R   = T_cam2base[:3, :3]
        self._t   = T_cam2base[:3, 3]
        self._bridge = CvBridge()
        self._lock = threading.Lock()

        self._hsv_lo = hsv_lo
        self._hsv_hi = hsv_hi
        self._K  = None   # 3x3 camera intrinsics matrix
        self._color_img = None
        self._depth_img = None
        self._last_pts  = None

        # Subscriptions
        self.create_subscription(CameraInfo, "/camera/depth/camera_info", self._info_cb, 1)
        self.create_subscription(Image, "/camera/color/image_raw",   self._color_cb, 10)
        self.create_subscription(Image, "/camera/depth/image_raw",   self._depth_cb, 10)

        # Publisher
        self._pub = self.create_publisher(PointCloud2, "/obstacles/base_frame", 10)

        # Processing timer — run at ~10 Hz
        self.create_timer(0.1, self._process)

        # Save timer — write obstacles.npy every second
        self.create_timer(1.0, self._save_cb)

        if hsv_lo is None:
            self.get_logger().warn("No HSV range loaded — will output empty obstacles until pick_color.py is run.")
        self.get_logger().info("Obstacle detector started. Waiting for images...")

    def _info_cb(self, msg: CameraInfo):
        if self._K is None:
            k = msg.k
            self._K = np.array(k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"Camera intrinsics loaded: fx={self._K[0,0]:.1f} fy={self._K[1,1]:.1f} cx={self._K[0,2]:.1f} cy={self._K[1,2]:.1f}")

    def _color_cb(self, msg: Image):
        with self._lock:
            self._color_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _depth_cb(self, msg: Image):
        with self._lock:
            self._depth_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def _process(self):
        if self._K is None or self._hsv_lo is None:
            return

        with self._lock:
            color = self._color_img
            depth = self._depth_img

        if color is None or depth is None:
            return

        # --- Color segmentation on RGB image ---
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        lo, hi = self._hsv_lo, self._hsv_hi
        if lo[0] <= hi[0]:
            mask = cv2.inRange(hsv, lo, hi)
        else:
            # Hue wraps around 0 (e.g. red/orange)
            mask1 = cv2.inRange(hsv, np.array([0,   lo[1], lo[2]], dtype=np.uint8), hi)
            mask2 = cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]], dtype=np.uint8))
            mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        n_px = int(mask.sum() / 255)
        if n_px < MIN_PIXELS:
            self.get_logger().warn(f"Too few pixels detected ({n_px}), cube not visible?",
                                   throttle_duration_sec=5.0)
            return

        # --- Map color pixels to depth image pixels ---
        # Color image may be different resolution from depth image
        color_h, color_w = color.shape[:2]
        depth_h, depth_w = depth.shape[:2]

        ys_color, xs_color = np.where(mask > 0)

        # Scale color pixel coords to depth image coords
        xs_depth = (xs_color * depth_w / color_w).astype(int)
        ys_depth = (ys_color * depth_h / color_h).astype(int)

        # Clamp to depth image bounds
        xs_depth = np.clip(xs_depth, 0, depth_w - 1)
        ys_depth = np.clip(ys_depth, 0, depth_h - 1)

        # Look up depth values
        depth_vals = depth[ys_depth, xs_depth].astype(np.float64) * DEPTH_SCALE  # metres

        # Filter invalid/zero depths
        valid = (depth_vals > 0.1) & (depth_vals < 2.0)
        xs_d = xs_depth[valid]
        ys_d = ys_depth[valid]
        zs   = depth_vals[valid]

        if len(zs) == 0:
            self.get_logger().warn("No valid depth values for detected pixels.",
                                   throttle_duration_sec=5.0)
            return

        # --- Backproject to 3D in camera frame ---
        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]

        # Use depth intrinsics for backprojection (depth image coordinates)
        xs_cam = (xs_d - cx) / fx * zs
        ys_cam = (ys_d - cy) / fy * zs
        pts_cam = np.stack([xs_cam, ys_cam, zs], axis=1)  # (N,3)

        # --- Transform to robot base frame ---
        pts_base = (self._R @ pts_cam.T).T + self._t
        # X and Y axes are flipped due to camera mounting orientation
        pts_base[:, 0] *= -1
        pts_base[:, 1] *= -1

        # --- Workspace filter ---
        mask_ws = (
            (pts_base[:, 0] >= WS_X[0]) & (pts_base[:, 0] <= WS_X[1]) &
            (pts_base[:, 1] >= WS_Y[0]) & (pts_base[:, 1] <= WS_Y[1]) &
            (pts_base[:, 2] >= WS_Z[0]) & (pts_base[:, 2] <= WS_Z[1])
        )
        pts_ws = pts_base[mask_ws]

        if len(pts_ws) == 0:
            self.get_logger().warn("Cube detected in image but outside robot workspace.",
                                   throttle_duration_sec=5.0)
            return

        # --- Voxel downsample ---
        pts_down = voxel_downsample(pts_ws, VOXEL_SIZE).astype(np.float32)

        with self._lock:
            self._last_pts = pts_down

        # Publish for RViz
        stamp = self.get_clock().now().to_msg()
        self._pub.publish(make_cloud_msg(pts_down, "base", stamp))

        self.get_logger().info(
            f"Cube: {n_px} px -> {len(pts_ws)} 3D pts -> {len(pts_down)} voxels",
            throttle_duration_sec=2.0,
        )

    def _save_cb(self):
        with self._lock:
            pts = self._last_pts
        if pts is None:
            return
        np.save(str(OBSTACLES_FILE), pts)
        self.get_logger().info(
            f"Saved {len(pts)} obstacle points -> {OBSTACLES_FILE}",
            throttle_duration_sec=5.0,
        )


def main():
    T_cam2base = load_T_cam2base()
    hsv_lo, hsv_hi = load_hsv_range()

    rclpy.init()
    node = ObstacleDetector(T_cam2base, hsv_lo, hsv_hi)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
