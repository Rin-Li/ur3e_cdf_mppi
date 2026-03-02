import sys, os
# Root = three_freedom/ (parent of planner/), so ur3e/ and RDF/ are importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import pybullet as p
import time
from ur3e.ur3e_pybullet import UR3ePyBullet
from planner.obstacle_manager import CrossObstacleManager

WS_MIN = np.array([-0.4, -0.4, 0.0])
WS_MAX = np.array([ 0.4,  0.4, 0.4])

# Start / goal configs — only joint[0] differs by π rad
Q0 = np.array([-1.57, -0.5, 1.2, -1.3, -1.57, 0.0], dtype=np.float32)
QF = np.array([ 1.57, -0.5, 1.2, -1.3, -1.57, 0.0], dtype=np.float32)



def rot_to_quat(R: np.ndarray):
    """Rotation matrix → PyBullet quaternion [x, y, z, w]"""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [x, y, z, w]


def load_ghost_links(mesh_dir: str, rgba):
    """Load a set of semi-transparent ghost robot links with a uniform color."""
    link_names = ['base', 'shoulder', 'upperarm', 'forearm', 'wrist1', 'wrist2', 'wrist3']
    ids = []
    for name in link_names:
        stl = os.path.join(mesh_dir, f'{name}.stl')
        if not os.path.exists(stl):
            continue
        col = p.createCollisionShape(p.GEOM_MESH, fileName=stl, meshScale=[1, 1, 1])
        vis = p.createVisualShape(p.GEOM_MESH, fileName=stl, meshScale=[1, 1, 1],
                                  rgbaColor=rgba)
        ids.append(p.createMultiBody(0, col, vis, [0, 0, 0]))
    return ids


def apply_fk(body_ids, joint_angles, ur3e_kin, device: str):
    """Update PyBullet body poses from joint angles via forward kinematics."""
    dev = torch.device(device)
    theta = torch.from_numpy(
        np.asarray(joint_angles, dtype=np.float32)
    ).unsqueeze(0).to(dev)
    pose = torch.eye(4).unsqueeze(0).to(dev).float()
    trans_list = ur3e_kin.get_transformations_each_link(pose, theta)
    for i, bid in enumerate(body_ids):
        if i >= len(trans_list):
            break
        T = trans_list[i].squeeze().cpu().numpy()
        p.resetBasePositionAndOrientation(bid, T[:3, 3].tolist(),
                                          rot_to_quat(T[:3, :3]))


def get_ee_pos(joint_angles, ur3e_kin, device: str) -> np.ndarray:
    """Return the end-effector (wrist3) position for the given joint angles."""
    dev = torch.device(device)
    theta = torch.from_numpy(
        np.asarray(joint_angles, dtype=np.float32)
    ).unsqueeze(0).to(dev)
    pose = torch.eye(4).unsqueeze(0).to(dev).float()
    trans_list = ur3e_kin.get_transformations_each_link(pose, theta)
    return trans_list[-1].squeeze().cpu().numpy()[:3, 3]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Launch PyBullet GUI with the main robot
    robot = UR3ePyBullet(device=device, gui=True)
    mesh_dir = os.path.join(_ROOT, 'ur3e', 'model')

    # 2. Start robot — recolor green, set to q0
    for bid in robot.link_body_ids:
        p.changeVisualShape(bid, -1, rgbaColor=[0.15, 0.85, 0.15, 1.0])
    robot.set_joint_angles(Q0)

    # 3. Goal robot — semi-transparent blue ghost at qf
    ur3e_kin = robot.ur3e
    goal_ids = load_ghost_links(mesh_dir, rgba=[0.2, 0.4, 1.0, 0.50])
    apply_fk(goal_ids, QF, ur3e_kin, device)

    # 4. Cross-shaped obstacle (red spheres)
    obs_manager = CrossObstacleManager(sphere_radius=0.05, device='cpu')
    centers = obs_manager.get_obstacle_points().numpy()
    r = obs_manager.sphere_radius
    for pos in centers:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=r,
                                  rgbaColor=[1.0, 0.15, 0.15, 0.80])
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
        p.createMultiBody(0, col, vis, pos.tolist())

    # 5. Workspace bounding box wireframe (blue)
    corners = np.array([[x, y, z]
                        for x in [WS_MIN[0], WS_MAX[0]]
                        for y in [WS_MIN[1], WS_MAX[1]]
                        for z in [WS_MIN[2], WS_MAX[2]]])
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    for a, b in edges:
        p.addUserDebugLine(corners[a].tolist(), corners[b].tolist(),
                           lineColorRGB=[0, 0.5, 1], lineWidth=1.5)

    # 6. EE position markers and text labels
    ee0 = robot.get_end_effector_pose()[0]
    eef = get_ee_pos(QF, ur3e_kin, device)

    p.createMultiBody(0,
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_SPHERE, radius=0.025, rgbaColor=[0.0, 1.0, 0.0, 1.0]),
        basePosition=ee0.tolist())
    p.createMultiBody(0,
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_SPHERE, radius=0.025, rgbaColor=[0.2, 0.4, 1.0, 1.0]),
        basePosition=eef.tolist())

    offset = np.array([0.0, 0.0, 0.06])
    p.addUserDebugText("Start q0", (ee0 + offset).tolist(),
                       textColorRGB=[0.0, 1.0, 0.0], textSize=1.3)
    p.addUserDebugText("Goal  qf", (eef + offset).tolist(),
                       textColorRGB=[0.2, 0.4, 1.0], textSize=1.3)
    p.addUserDebugText("Cross Obstacle", [0.15, 0.0, 0.33],
                       textColorRGB=[1.0, 0.2, 0.2], textSize=1.3)

    # 7. Console summary
    delta_deg = np.degrees(QF[0] - Q0[0])
    print("=" * 60)
    print(f"  Green robot  = start q0 = {np.round(Q0, 3)}")
    print(f"  Blue robot   = goal  qf = {np.round(QF, 3)}")
    print(f"  Only joint[0] differs: {Q0[0]:.2f} -> {QF[0]:.2f} rad  (delta={delta_deg:.0f} deg)")
    print(f"  Start EE position: {np.round(ee0, 3)}")
    print(f"  Goal  EE position: {np.round(eef, 3)}")
    print(f"  Cross obstacle: center[0.15,0.0,0.2]  r={r}m  {len(centers)} spheres")
    print(f"  Workspace: X[-0.4,0.4]  Y[-0.4,0.4]  Z[0,0.4]")
    print("  Press Ctrl+C to exit...")
    print("=" * 60)

    try:
        while True:
            p.stepSimulation()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass

    p.disconnect()


if __name__ == '__main__':
    main()

