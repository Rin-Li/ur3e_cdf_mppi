"""Hand-eye calibration data collection (eye-to-hand).

ChArUco board is mounted on robot flange.
Camera is fixed in environment.

Controls:
  SPACE  — capture current pose (only if board detected)
  Q      — save data and quit
  D      — delete last capture

Run:
  source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
  python3 visual/collect_calib_poses.py
"""

import pathlib
import threading
import sys
import numpy as np
import cv2
import cv2.aruco as aruco
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import rtde_receive

_ROOT = pathlib.Path(__file__).parent.parent
OUTPUT_DIR = _ROOT / "visual" / "calib_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "poses.npz"

ROBOT_IP   = "192.168.1.102"
MIN_CORNERS = 10   # minimum ChArUco corners to accept a capture

# Board params — must match generate_charuco.py
COLS       = 7
ROWS       = 5
SQUARE_M   = 0.040
MARKER_M   = 0.030
ARUCO_DICT = aruco.DICT_5X5_50


class ImageReceiver(Node):
    def __init__(self):
        super().__init__("calib_collector")
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self.latest_frame = None
        self.camera_matrix = None
        self.dist_coeffs = None

        self.create_subscription(CameraInfo, "/camera/color/camera_info",
                                 self._info_cb, 10)
        self.create_subscription(Image, "/camera/color/image_raw",
                                 self._image_cb, 10)

    def _info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs   = np.array(msg.d, dtype=np.float64)
            self.get_logger().info(
                f"Camera intrinsics received: fx={self.camera_matrix[0,0]:.1f}")

    def _image_cb(self, msg):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self.latest_frame = frame

    def get_frame(self):
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None


def rotation_diff_deg(R1, R2):
    """Angle between two rotation matrices in degrees."""
    R_rel = R1.T @ R2
    cos_val = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))


def main():
    # ── Robot connection ──────────────────────────────────────────────────────
    print(f"Connecting to robot at {ROBOT_IP} ...")
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    except Exception as e:
        print(f"ERROR: Cannot connect to robot: {e}")
        sys.exit(1)
    print("Robot connected.")

    # ── ROS2 setup ────────────────────────────────────────────────────────────
    rclpy.init()
    node = ImageReceiver()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    print("Waiting for camera...")
    while node.camera_matrix is None:
        executor.spin_once(timeout_sec=0.5)
    print("Camera ready.")

    # ── ArUco / ChArUco setup ─────────────────────────────────────────────────
    dictionary = aruco.Dictionary_get(ARUCO_DICT)
    board      = aruco.CharucoBoard_create(COLS, ROWS, SQUARE_M, MARKER_M, dictionary)
    params     = aruco.DetectorParameters_create()

    camera_matrix = node.camera_matrix
    dist_coeffs   = node.dist_coeffs

    # ── Capture storage ───────────────────────────────────────────────────────
    captures = []   # list of dicts with R/t pairs

    print("\n=== Calibration data collection ===")
    print("  Put robot in FREEDRIVE mode on teach pendant")
    print("  Move to varied poses (different heights, tilts, distances)")
    print("  SPACE = capture  |  D = delete last  |  Q = save & quit")
    print("  Aim for 20+ captures with varied orientations\n")

    flash_counter = 0   # frames to show "CAPTURED!" message

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        executor.spin_once(timeout_sec=0.033)
        frame = node.get_frame()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)

        board_detected = False
        current_rvec = current_tvec = None
        n_corners = 0

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(display, corners, ids)
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            if ret >= MIN_CORNERS:
                aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
                ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, board,
                    camera_matrix, dist_coeffs, None, None)
                if ok:
                    aruco.drawAxis(display, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                    board_detected = True
                    current_rvec = rvec
                    current_tvec = tvec
                    n_corners = ret

        # Border colour: green = detected, red = not
        border_color = (0, 200, 0) if board_detected else (0, 0, 200)
        cv2.rectangle(display, (0, 0), (display.shape[1]-1, display.shape[0]-1),
                      border_color, 6)

        # Status text
        status = f"Board OK ({n_corners} corners)" if board_detected else "No board detected"
        status_color = (0, 220, 0) if board_detected else (0, 0, 220)
        cv2.putText(display, status, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)
        cv2.putText(display, f"Captures: {len(captures)}", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "SPACE:capture  D:delete  Q:save+quit",
                    (15, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        if flash_counter > 0:
            cv2.putText(display, "CAPTURED!", (display.shape[1]//2 - 120, display.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, cv2.LINE_AA)
            flash_counter -= 1

        cv2.imshow("Hand-Eye Calibration — ChArUco Collection", display)
        key = cv2.waitKey(1) & 0xFF

        # ── SPACE: capture ────────────────────────────────────────────────────
        if key == ord(" ") and board_detected:
            tcp = rtde_r.getActualTCPPose()   # [x,y,z, rx,ry,rz]
            rot_vec = np.array(tcp[3:6], dtype=np.float64)
            t_g2b   = np.array(tcp[0:3], dtype=np.float64).reshape(3, 1)
            R_g2b, _ = cv2.Rodrigues(rot_vec)

            R_t2c, _ = cv2.Rodrigues(current_rvec)
            t_t2c    = current_tvec.reshape(3, 1).astype(np.float64)

            # Warn if too similar to previous
            skip = False
            if captures:
                diff = rotation_diff_deg(captures[-1]["R_g2b"], R_g2b)
                if diff < 5.0:
                    print(f"  [WARN] Only {diff:.1f}° from previous pose — move robot more before capturing")
                    skip = True

            if not skip:
                captures.append({"R_g2b": R_g2b, "t_g2b": t_g2b,
                                  "R_t2c": R_t2c, "t_t2c": t_t2c})
                print(f"  Captured #{len(captures):2d} | TCP [mm]: "
                      f"[{tcp[0]*1000:6.1f}, {tcp[1]*1000:6.1f}, {tcp[2]*1000:6.1f}]")
                flash_counter = 15

        # ── D: delete last ────────────────────────────────────────────────────
        elif key == ord("d") and captures:
            captures.pop()
            print(f"  Deleted last capture. Total: {len(captures)}")

        # ── Q: quit ───────────────────────────────────────────────────────────
        elif key == ord("q"):
            if len(captures) < 15:
                print(f"  [WARN] Only {len(captures)} captures (recommend ≥15). Press Q again to force save.")
                # require second Q
                cv2.waitKey(0)
                k2 = cv2.waitKey(2000) & 0xFF
                if k2 != ord("q") and len(captures) < 8:
                    print("  Continuing capture...")
                    continue
            break

    cv2.destroyAllWindows()

    # ── Save ──────────────────────────────────────────────────────────────────
    if len(captures) == 0:
        print("No captures — nothing saved.")
    else:
        np.savez(str(OUTPUT_FILE),
                 R_gripper2base = np.stack([c["R_g2b"] for c in captures]),
                 t_gripper2base = np.stack([c["t_g2b"] for c in captures]),
                 R_target2cam   = np.stack([c["R_t2c"] for c in captures]),
                 t_target2cam   = np.stack([c["t_t2c"] for c in captures]),
                 camera_matrix  = camera_matrix,
                 dist_coeffs    = dist_coeffs,
                 n_captures     = np.array(len(captures)))
        print(f"\nSaved {len(captures)} poses → {OUTPUT_FILE}")
        print("Next step: python3 visual/solve_handeye.py")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
