"""
Read the current UR3e robot joint state and TCP pose via RTDE.

Run:
    cd /home/kklab-ur-robot/ur3e_cdf_mppi
    .venv/bin/python read_robot_state.py

Optional:
    .venv/bin/python read_robot_state.py 192.168.1.102
"""

from __future__ import annotations

import sys
import numpy as np


DEFAULT_ROBOT_IP = "192.168.1.102"
SYSTEM_DIST_PACKAGES = "/usr/lib/python3/dist-packages"


def import_rtde_receive():
    try:
        import rtde_receive
        return rtde_receive
    except ImportError:
        if SYSTEM_DIST_PACKAGES not in sys.path:
            sys.path.append(SYSTEM_DIST_PACKAGES)
        import rtde_receive
        return rtde_receive


def main() -> int:
    robot_ip = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROBOT_IP

    try:
        rtde_receive = import_rtde_receive()
    except ImportError as exc:
        print(f"Failed to import rtde_receive: {exc}")
        print("Tried both the current environment and /usr/lib/python3/dist-packages.")
        return 1

    try:
        receiver = rtde_receive.RTDEReceiveInterface(robot_ip)
        q = np.array(receiver.getActualQ(), dtype=np.float64)
        tcp = np.array(receiver.getActualTCPPose(), dtype=np.float64)
    except Exception as exc:
        print(f"Failed to read robot state from {robot_ip}: {exc}")
        return 1

    np.set_printoptions(precision=6, suppress=True)
    print(f"robot_ip = {robot_ip}")
    print("q (rad) =", q.tolist())
    print("tcp [x, y, z, rx, ry, rz] =", tcp.tolist())
    print(f"tcp position (m) = {tcp[:3].tolist()}")
    print(f"tcp rotation vector (rad) = {tcp[3:].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
