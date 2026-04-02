"""Visualize detected obstacle points + real robot pose in PyBullet.

- Loads obstacles.npy every second and renders as red spheres
- Reads current robot joint angles via RTDE and updates PyBullet in real-time
- Falls back to home pose if robot is not connected

Run:
  python3 visual/visualize_obstacles_pybullet.py
"""

import sys
import time
import pathlib
import numpy as np
import pybullet as p

_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from ur3e.ur3e_pybullet import UR3ePyBullet

OBS_FILE        = _ROOT / "visual" / "calib_data" / "obstacles.npy"
OBS_RADIUS      = 0.03    # display radius (metres)
RELOAD_INTERVAL = 1.0     # reload obstacles.npy every N seconds
ROBOT_IP        = "192.168.1.102"
HOME_Q          = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]


def connect_rtde(robot_ip):
    try:
        import rtde_receive
        r = rtde_receive.RTDEReceiveInterface(robot_ip)
        print(f"[viz] Connected to robot at {robot_ip}")
        return r
    except Exception:
        try:
            import sys as _sys
            _sys.path.append("/usr/lib/python3/dist-packages")
            import rtde_receive
            r = rtde_receive.RTDEReceiveInterface(robot_ip)
            print(f"[viz] Connected to robot at {robot_ip}")
            return r
        except Exception as e:
            print(f"[viz] WARNING: Cannot connect to robot ({e})")
            print("      Showing home pose instead.")
            return None


def main():
    print("[viz] Opening PyBullet...")
    robot_vis = UR3ePyBullet(gui=True)

    # Look from +Y side (same side as physical camera and user)
    # yaw=0 → viewer at +Y looking toward robot: cube between camera and robot appears in front
    p.resetDebugVisualizerCamera(
        cameraDistance=1.8,
        cameraYaw=0,
        cameraPitch=-35,
        cameraTargetPosition=[0, 0, 0.3]
    )

    # Draw coordinate axes at origin for reference
    axis_len = 0.3
    p.addUserDebugLine([0,0,0], [axis_len,0,0], [1,0,0], lineWidth=3)  # X red
    p.addUserDebugLine([0,0,0], [0,axis_len,0], [0,1,0], lineWidth=3)  # Y green
    p.addUserDebugLine([0,0,0], [0,0,axis_len], [0,0,1], lineWidth=3)  # Z blue
    p.addUserDebugText("X", [axis_len+0.02,0,0], [1,0,0], textSize=1.5)
    p.addUserDebugText("Y", [0,axis_len+0.02,0], [0,1,0], textSize=1.5)
    p.addUserDebugText("Z", [0,0,axis_len+0.02], [0,0,1], textSize=1.5)

    # Try to connect to real robot
    rtde_r = connect_rtde(ROBOT_IP)

    # Set initial pose
    if rtde_r is not None:
        q_init = rtde_r.getActualQ()
    else:
        q_init = HOME_Q
    robot_vis.set_joint_angles(q_init)
    p.stepSimulation()

    if not OBS_FILE.exists():
        print(f"[viz] WARNING: {OBS_FILE} not found — no obstacles shown until obstacle_detector.py is running.")

    print(f"\n[viz] Running. Press Ctrl+C to quit.")
    print(f"      Obstacles reload every {RELOAD_INTERVAL}s from {OBS_FILE}")
    if rtde_r:
        print(f"      Robot pose updates in real-time from {ROBOT_IP}\n")
    else:
        print(f"      Robot pose: fixed at home (no robot connection)\n")

    sphere_ids  = []
    last_reload = 0.0

    try:
        while True:
            now = time.time()

            # --- Update robot joint angles ---
            if rtde_r is not None:
                try:
                    q = rtde_r.getActualQ()
                    robot_vis.set_joint_angles(q)
                except Exception as e:
                    print(f"[viz] RTDE read error: {e}")
                    rtde_r = None

            # --- Reload obstacles ---
            if now - last_reload >= RELOAD_INTERVAL:
                for bid in sphere_ids:
                    try:
                        p.removeBody(bid)
                    except Exception:
                        pass
                sphere_ids.clear()

                if OBS_FILE.exists():
                    obs = np.load(str(OBS_FILE)).astype(np.float32)
                    if len(obs) > 0:
                        vis_shape = p.createVisualShape(
                            p.GEOM_SPHERE,
                            radius=OBS_RADIUS,
                            rgbaColor=[1.0, 0.2, 0.2, 0.6],
                        )
                        for pt in obs:
                            bid = p.createMultiBody(
                                baseMass=0,
                                baseVisualShapeIndex=vis_shape,
                                basePosition=pt.tolist(),
                            )
                            sphere_ids.append(bid)
                        c = obs.mean(axis=0)
                        print(f"[viz] {len(obs)} pts  centroid X={c[0]:.3f} Y={c[1]:.3f} Z={c[2]:.3f} m")

                last_reload = now

            p.stepSimulation()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[viz] Quit.")
    finally:
        robot_vis.disconnect()


if __name__ == "__main__":
    main()
