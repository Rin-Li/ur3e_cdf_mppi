"""Record camera color stream to a video file.

Subscribes to /camera/color/image_raw and writes to an mp4 file.
Runs silently in the background — does not interfere with other nodes.

Usage:
    source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
    python3 visual/record_video.py                        # auto-named output
    python3 visual/record_video.py --out my_run.mp4       # custom filename
    Ctrl+C to stop recording
"""

import argparse
import sys
import pathlib
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

_ROOT = pathlib.Path(__file__).parent.parent


class VideoRecorder(Node):
    def __init__(self, out_path: str, fps: float = 30.0):
        super().__init__("video_recorder")
        self._bridge  = CvBridge()
        self._writer  = None
        self._fps     = fps
        self._out_path = out_path
        self._frame_count = 0

        self.create_subscription(Image, "/camera/color/image_raw", self._cb, 10)
        self.get_logger().info(f"Recording to: {out_path}  (Ctrl+C to stop)")

    def _cb(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._out_path, fourcc, self._fps, (w, h))
            self.get_logger().info(f"Stream: {w}x{h} @ {self._fps}fps")

        self._writer.write(frame)
        self._frame_count += 1
        if self._frame_count % 150 == 0:
            self.get_logger().info(f"  {self._frame_count} frames recorded...")

        cv2.imshow("Recording (press Q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

    def stop(self):
        cv2.destroyAllWindows()
        if self._writer is not None:
            self._writer.release()
            self.get_logger().info(
                f"Saved {self._frame_count} frames → {self._out_path}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="Output mp4 path")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _ROOT / "recordings"
        out_dir.mkdir(exist_ok=True)
        args.out = str(out_dir / f"rec_{ts}.mp4")

    rclpy.init()
    node = VideoRecorder(args.out, fps=args.fps)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
