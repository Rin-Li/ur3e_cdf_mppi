"""Real-world MPPI planner for UR3e — with PyBullet preview + confirmation.

Flow:
  1. Read current robot joint angles as start pose
  2. Load obstacle point cloud from camera (obstacles.npy)
  3. MPPI planning + cubic spline interpolation
  4. PyBullet GUI trajectory animation (with obstacle spheres)
  5. Terminal confirmation: [s] send / [r] replay / [p] replan / [q] cancel

Run:
  source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
  .venv/bin/python planner/planner_ur3e_real.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ====== Internal imports ======================================================
CUR_PATH  = Path(__file__).resolve().parent
ROOT_PATH = CUR_PATH.parent
sys.path.insert(0, str(ROOT_PATH))

from cdf.mlp     import MLPRegression
from cdf.nn_cdf  import CDF
from ur3e.ur3e   import URRobot
from rdf.ur_rdf  import BPSDF
from planner.planner_ur3e import (
    MPPIUR3e, JOINT_LIMITS,
    DT, CONS_U, EPS_GOAL, MAX_STEPS,
)

# =============================================================================
# Configuration
# =============================================================================

REAL_OBS_RADIUS: float = 0.05   # obstacle sphere radius (meters)
ROBOT_IP:        str   = "192.168.1.102"
DEVICE:          str   = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH: Path = ROOT_PATH / "cdf" / "model" / "model_best.pt"
OBS_FILE:   Path = ROOT_PATH / "visual" / "calib_data" / "obstacles.npy"

# Goal joint angles (radians) — modify here to set the target configuration
GOAL_Q = np.array([-1.6504529158221644, -1.774635454217428, 0.9465034643756312, -1.9923249683775843, -1.3538959662066858, 0.5546590685844421], dtype=np.float32)

# Trajectory send parameters
EXEC_DT:       float = 0.002   # time interval per point (seconds), must match servo_j node
EXEC_DURATION: float = 8.0     # total motion duration (seconds)


# =============================================================================
# RTDE compatibility
# =============================================================================

def import_rtde_receive():
    try:
        import rtde_receive
        return rtde_receive
    except ImportError:
        system_dist_packages = "/usr/lib/python3/dist-packages"
        if system_dist_packages not in sys.path:
            sys.path.append(system_dist_packages)
        import rtde_receive
        return rtde_receive


# =============================================================================
# PyBullet preview + terminal confirmation
# =============================================================================

def preview_and_confirm(traj_np: np.ndarray, obs_np: np.ndarray):
    """Play planned trajectory in PyBullet; returns True / False / 'replan'."""
    import pybullet as p
    from ur3e.ur3e_pybullet import UR3ePyBullet

    print("\n[preview] Opening PyBullet preview window...")
    robot_vis = UR3ePyBullet(device=DEVICE, gui=True)

    # Draw obstacle spheres in PyBullet (red, semi-transparent)
    obs_sphere_ids = []
    if len(obs_np) > 0:
        vis_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=REAL_OBS_RADIUS,
            rgbaColor=[1.0, 0.2, 0.2, 0.4],
        )
        for pt in obs_np:
            bid = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis_shape,
                basePosition=pt.tolist(),
            )
            obs_sphere_ids.append(bid)
        print(f"[preview] Obstacle spheres: {len(obs_np)}")

    def replay():
        # Preview uses 200 frames spaced over EXEC_DURATION seconds (real-time speed)
        preview_idx = np.linspace(0, len(traj_np) - 1, 200).astype(int)
        preview_traj = traj_np[preview_idx]
        dt_preview = EXEC_DURATION / len(preview_traj)
        print(f"[preview] Playing trajectory ({len(preview_traj)} frames, duration≈{EXEC_DURATION:.1f}s)... Ctrl+C to skip")
        try:
            for q in preview_traj:
                robot_vis.set_joint_angles(q)
                p.stepSimulation()
                time.sleep(dt_preview)
            robot_vis.set_joint_angles(traj_np[-1])
            p.stepSimulation()
        except KeyboardInterrupt:
            print("[preview] Playback interrupted, returning to confirmation menu.")

    replay()

    # Terminal confirmation loop
    while True:
        print("\n" + "="*50)
        print("  [s] Confirm — send trajectory to robot")
        print("  [r] Replay PyBullet animation")
        print("  [p] Replan (re-read robot state and obstacles)")
        print("  [q] Cancel — do not execute")
        print("="*50)
        try:
            key = input("Enter s / r / p / q : ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            key = 'q'

        if key == 's':
            robot_vis.disconnect()
            return True
        elif key == 'r':
            # Reset to start pose then replay
            robot_vis.set_joint_angles(traj_np[0])
            p.stepSimulation()
            time.sleep(0.3)
            replay()
        elif key == 'p':
            print("[preview] Replanning...")
            robot_vis.disconnect()
            return 'replan'
        elif key == 'q':
            print("[preview] Cancelled, not sending to robot.")
            robot_vis.disconnect()
            return False
        else:
            print("  Invalid input, please enter s / r / p / q")


# =============================================================================
# Trajectory resampling
# =============================================================================

def resample_trajectory(traj_np: np.ndarray, n_pts: int) -> np.ndarray:
    """Resample raw MPPI trajectory to n_pts points using cubic spline."""
    from scipy.interpolate import CubicSpline
    n = len(traj_np)
    t_src = np.linspace(0.0, 1.0, n)
    t_dst = np.linspace(0.0, 1.0, n_pts)
    cs = CubicSpline(t_src, traj_np, axis=0)
    return cs(t_dst).astype(np.float32)


# =============================================================================
# ROS2 trajectory publisher
# =============================================================================

def publish_trajectory(traj_np: np.ndarray):
    """Publish planned trajectory to /target/multi_joint_trajectory."""
    import rclpy
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration as RosDuration

    msg = JointTrajectory()
    msg.joint_names = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
    for i, q in enumerate(traj_np):
        pt  = JointTrajectoryPoint()
        pt.positions = q.astype(float).tolist()
        t_ns = int(i * EXEC_DT * 1e9)
        pt.time_from_start = RosDuration(sec=t_ns // 10**9, nanosec=t_ns % 10**9)
        msg.points.append(pt)

    class _Sender(Node):
        def __init__(self):
            super().__init__("mppi_real_sender")
            self.pub = self.create_publisher(JointTrajectory, "/target/multi_joint_trajectory", 10)

        def send(self, msg):
            self.get_logger().info("Waiting for servo_j node to subscribe...")
            while self.count_subscribers("/target/multi_joint_trajectory") == 0:
                rclpy.spin_once(self, timeout_sec=0.5)
            self.get_logger().info(f"Sending trajectory ({len(msg.points)} points, duration={EXEC_DURATION}s)")
            self.pub.publish(msg)
            self.get_logger().info("Trajectory sent.")

    rclpy.init()
    node = _Sender()
    node.send(msg)
    rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device(DEVICE)
    np.set_printoptions(precision=4, suppress=True)

    # ------------------------------------------------------------------
    # 1. Load CDF model
    # ------------------------------------------------------------------
    cdf = CDF(device, load_data=False)
    net = MLPRegression(9, 1, [1024, 512, 256, 128, 128], skips=[], act_fn=nn.ReLU, nerf=True)
    state_obj = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(state_obj, dict) and all(isinstance(k, int) for k in state_obj.keys()):
        net.load_state_dict(state_obj[max(state_obj.keys())])
    else:
        net.load_state_dict(state_obj)
    net.to(device).eval()
    print(f"[real] CDF loaded: {MODEL_PATH}")

    # ------------------------------------------------------------------
    # 2. Load RDF model
    # ------------------------------------------------------------------
    robot     = URRobot(device)
    rdf       = BPSDF(8, -1.0, 1.0, robot, device)
    rdf_model = torch.load(ROOT_PATH / "rdf/models/BP_8.pt", map_location=device, weights_only=False)
    print("[real] RDF loaded")

    # ------------------------------------------------------------------
    # 3-7. Planning loop (supports replanning)
    # ------------------------------------------------------------------
    while True:
        # 3. Read start pose
        try:
            rtde_receive = import_rtde_receive()
            rtde_r  = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
            q_start = np.array(rtde_r.getActualQ(), dtype=np.float32)
            print(f"[real] Start pose (from robot): {q_start}")
        except Exception as e:
            print(f"[real] Warning: cannot connect to robot ({e}), using fallback start pose.")
            q_start = np.array([-1.57, -0.5, 1.2, -1.3, -1.57, 0.0], dtype=np.float32)

        q_goal = GOAL_Q.copy()
        print(f"[real] Goal: {q_goal}")

        # 4. Load obstacle point cloud
        if OBS_FILE.exists():
            obs_np = np.load(str(OBS_FILE)).astype(np.float32)
            print(f"[real] Loaded {len(obs_np)} obstacle points from: {OBS_FILE}")
        else:
            print(f"[real] Warning: {OBS_FILE} does not exist, obstacle list is empty.")
            print("[real] Please run visual/obstacle_detector.py first")
            obs_np = np.zeros((0, 3), dtype=np.float32)

        obs_pts = torch.from_numpy(
            obs_np if len(obs_np) > 0 else np.array([[5., 5., 5.]], dtype=np.float32)
        ).float().to(device)

        # 5. Check start/goal collision
        pose = torch.eye(4).unsqueeze(0).to(device).float()
        sdf_s, _ = rdf.get_whole_body_sdf_batch(obs_pts, pose,
            torch.from_numpy(q_start).float().to(device).unsqueeze(0), rdf_model, use_derivative=False)
        sdf_g, _ = rdf.get_whole_body_sdf_batch(obs_pts, pose,
            torch.from_numpy(q_goal).float().to(device).unsqueeze(0), rdf_model, use_derivative=False)
        print(f"[real] Start SDF: {sdf_s.min().item():.4f}   Goal SDF: {sdf_g.min().item():.4f}   (threshold: {REAL_OBS_RADIUS})")
        if sdf_s.min().item() - REAL_OBS_RADIUS < 0:
            print("[real] Warning: start pose may be in collision with obstacles!")
        if sdf_g.min().item() - REAL_OBS_RADIUS < 0:
            print("[real] Warning: goal pose may be in collision with obstacles!")

        # 6. MPPI planning
        solver = MPPIUR3e(
            start=torch.tensor(q_start).unsqueeze(0).to(device),
            goal=torch.tensor(q_goal).to(device),
            obs_pts=obs_pts, cdf=cdf, net=net, rdf=rdf, rdf_model=rdf_model,
            dt=DT, cons_u=CONS_U, obs_radius=REAL_OBS_RADIUS,
        )

        print("[real] Starting MPPI planning...")
        traj, success, path_len, steps, elapsed = solver.optimise(
            max_steps=MAX_STEPS, eps_goal=EPS_GOAL, verbose=True)
        print(f"[real] Planning done: success={success}, steps={steps}, "
              f"path_len={path_len:.3f}, time={elapsed:.2f}s")

        traj_np = np.stack(traj, axis=0).astype(np.float32)

        if len(traj_np) < 2:
            print(f"[real] Trajectory too short ({len(traj_np)} points), please replan.")
            input("Press Enter to replan...")
            continue

        # Resample to send resolution via cubic spline
        n_pts = int(EXEC_DURATION / EXEC_DT)
        traj_np = resample_trajectory(traj_np, n_pts)
        print(f"[real] Trajectory resampled: {n_pts} points")

        out_path = ROOT_PATH / "saved_traj_ur3e" / "real_traj.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path), q=traj_np, start=q_start, goal=q_goal, success=success)
        print(f"[real] Trajectory saved: {out_path}")

        # 7. PyBullet preview + user confirmation
        if not success:
            print("[real] Planning failed — previewing current trajectory in PyBullet, decide whether to send anyway.")

        result = preview_and_confirm(traj_np, obs_np)

        if result == 'replan':
            print("[real] Replanning...")
            continue
        elif result is True:
            if not success:
                print("[real] Warning: trajectory did not converge, sending anyway.")
            print("[real] User confirmed — sending trajectory to robot...")
            publish_trajectory(traj_np)
            print("[real] Done.")
            break
        else:
            print("[real] Cancelled, trajectory not sent.")
            break


if __name__ == "__main__":
    main()
