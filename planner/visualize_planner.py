"""
Visualize UR3e MPPI planning results.
Shows the straight-line path first, then replays the MPPI planned trajectory.
"""

import numpy as np
import pybullet as p
import pybullet_data
import time
import sys
import os
# Root = three_freedom/ (parent of planner/), so ur3e/ and RDF/ are importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from ur3e.ur3e_pybullet import UR3ePyBullet
from planner.obstacle_manager import CrossObstacleManager


def visualize_trajectory(traj_file, show_direct_path=True):
    data = np.load(traj_file)
    q_traj = data['q']
    q_start = data['start']
    q_goal = data['goal']

    print(f"\n{'='*60}")
    print(f"Trajectory file: {traj_file}")
    print(f"Start: {q_start}")
    print(f"Goal:  {q_goal}")
    print(f"Trajectory length: {len(q_traj)} steps")
    print(f"{'='*60}\n")

    robot = UR3ePyBullet(device='cuda', gui=True)

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
    except:
        pass

    # Obstacle — consistent with planner_ur3e.py and visualize_obstacles.py
    obs_manager = CrossObstacleManager(sphere_radius=0.05, device='cpu')
    centers = obs_manager.get_obstacle_points().numpy()
    r = obs_manager.sphere_radius
    for pos in centers:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[1.0, 0.15, 0.15, 0.80])
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
        p.createMultiBody(0, col, vis, pos.tolist())

    # Start (green) and goal (blue) EE markers
    robot.set_joint_angles(q_start)
    ee_start, _ = robot.get_end_effector_pose()
    p.createMultiBody(0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 0.8]), basePosition=ee_start)

    robot.set_joint_angles(q_goal)
    ee_goal, _ = robot.get_end_effector_pose()
    p.createMultiBody(0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 1, 0.8]), basePosition=ee_goal)

    # Direct (straight-line) path in red
    if show_direct_path:
        print("Showing direct path (red)...")
        ee_direct = []
        for i in range(30):
            q_interp = (1 - i/29) * q_start + (i/29) * q_goal
            robot.set_joint_angles(q_interp)
            ee_pos, _ = robot.get_end_effector_pose()
            ee_direct.append(ee_pos)
            time.sleep(0.05)
        for i in range(len(ee_direct)-1):
            p.addUserDebugLine(ee_direct[i], ee_direct[i+1], [1, 0, 0], 2)
        time.sleep(1)

    # MPPI path in green
    print("Showing MPPI path (green)...")
    ee_mppi = []
    for q in q_traj:
        robot.set_joint_angles(q)
        ee_pos, _ = robot.get_end_effector_pose()
        ee_mppi.append(ee_pos)

    for i in range(len(ee_mppi)-1):
        p.addUserDebugLine(ee_mppi[i], ee_mppi[i+1], [0, 1, 0], 4)

    for i, q in enumerate(q_traj):
        robot.set_joint_angles(q)
        time.sleep(0.03)

    print("Done. Red = direct path, Green = MPPI path")
    print("Press Ctrl+C to exit...")

    try:
        for _ in range(300):  # auto-exit after 30 seconds
            p.stepSimulation()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    p.disconnect()


if __name__ == "__main__":
    import argparse, glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None,
                        help='Single trajectory file; if omitted, all trial_*.npz files in --dir are played')
    parser.add_argument('--dir', type=str,
                        default=os.path.join(_ROOT, 'saved_traj_ur3e'),
                        help='Trajectory directory (used when --file is not specified)')
    args = parser.parse_args()

    if args.file:
        visualize_trajectory(args.file)
    else:
        files = sorted(glob.glob(os.path.join(args.dir, 'trial_*.npz')))
        if not files:
            print(f"No trajectory files found: {args.dir}/trial_*.npz")
        else:
            print(f"Found {len(files)} trajectories, playing sequentially...")
            for f in files:
                visualize_trajectory(f)
