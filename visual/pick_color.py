"""Color picker for cube obstacle detection.

Click on the cube in the live camera feed to sample its HSV color range.
Press 's' to save, 'q' to quit.

Run:
  source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
  python3 visual/pick_color.py
"""

import pathlib
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

_ROOT = pathlib.Path(__file__).parent.parent
CALIB_DIR = _ROOT / "visual" / "calib_data"
HSV_FILE = CALIB_DIR / "cube_hsv.npz"

HSV_MARGIN = np.array([15, 60, 60])   # ±margin around sampled HSV


class ColorPicker(Node):
    def __init__(self):
        super().__init__("color_picker")
        self._bridge = CvBridge()
        self._frame = None
        self._hsv_samples = []

        self._sub = self.create_subscription(
            Image, "/camera/color/image_raw", self._img_cb, 10
        )
        cv2.namedWindow("pick_color", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("pick_color", self._mouse_cb)
        self.get_logger().info(
            "Click on the cube to sample its color. Press 's' to save, 'q' to quit."
        )

    def _img_cb(self, msg):
        self._frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self._frame is not None:
            hsv = cv2.cvtColor(self._frame, cv2.COLOR_BGR2HSV)
            # Sample 5x5 patch around click
            patch = hsv[max(0,y-2):y+3, max(0,x-2):x+3]
            mean_hsv = patch.reshape(-1, 3).mean(axis=0)
            self._hsv_samples.append(mean_hsv)
            print(f"  Sampled HSV: {mean_hsv.astype(int)}  (total samples: {len(self._hsv_samples)})")

            # Draw circle on display
            cv2.circle(self._frame, (x, y), 8, (0, 255, 0), 2)

    def spin_once(self):
        rclpy.spin_once(self, timeout_sec=0.03)

    def get_hsv_range(self):
        if not self._hsv_samples:
            return None, None
        samples = np.array(self._hsv_samples)
        center = samples.mean(axis=0)
        lo = np.clip(center - HSV_MARGIN, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
        hi = np.clip(center + HSV_MARGIN, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
        return lo, hi


def main():
    rclpy.init()
    node = ColorPicker()

    print("\n=== Cube Color Picker ===")
    print("Click on the cube multiple times to sample its color.")
    print("Press 's' to save the color range.")
    print("Press 'r' to reset samples.")
    print("Press 'q' to quit without saving.\n")

    saved = False
    while rclpy.ok():
        node.spin_once()
        frame = node._frame
        if frame is None:
            continue

        display = frame.copy()

        # Show current mask if we have samples
        lo, hi = node.get_hsv_range()
        if lo is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Handle hue wrap-around (e.g. orange/red near H=0 or H=180)
            if lo[0] <= hi[0]:
                mask = cv2.inRange(hsv, lo, hi)
            else:
                # hue wraps around 0
                mask1 = cv2.inRange(hsv, np.array([0, lo[1], lo[2]]), hi)
                mask2 = cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]]))
                mask = cv2.bitwise_or(mask1, mask2)

            # Overlay mask in green
            display[mask > 0] = (display[mask > 0] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
            n_px = int(mask.sum() / 255)
            cv2.putText(display, f"Masked pixels: {n_px}  Samples: {len(node._hsv_samples)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"HSV lo: {lo}  hi: {hi}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display, "Click on the cube to sample color",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.imshow("pick_color", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and lo is not None:
            CALIB_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(str(HSV_FILE), lo=lo, hi=hi)
            print(f"\n[saved] HSV range saved to {HSV_FILE}")
            print(f"  lo = {lo}")
            print(f"  hi = {hi}")
            saved = True
            break
        elif key == ord('r'):
            node._hsv_samples.clear()
            print("[reset] Samples cleared.")
        elif key == ord('q'):
            print("[quit] Not saved.")
            break

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
    return saved


if __name__ == "__main__":
    main()
