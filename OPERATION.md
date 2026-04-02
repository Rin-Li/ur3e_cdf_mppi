# UR3e Operation Guide

## Environment Overview

| Component | Details |
|-----------|---------|
| Robot IP | `192.168.1.102` |
| ROS2 | Humble (`/opt/ros/humble`) |
| Python environment | `.venv` (Python 3.10, located at `ur3e_cdf_mppi/.venv`) |
| Camera ROS2 workspace | `~/ros2_ws` |
| Robot ROS2 workspace | `~/ur3e_cdf_mppi/ros2_ws` |
| Camera | Orbbec Gemini 335L (USB 3.0) |

---

## 1. One-Time Environment Setup

### 1.1 Install librtde to system path

```bash
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/ur-rtde.conf
sudo ldconfig
```

### 1.2 Build ROS2 workspace

```bash
cd ~/ur3e_cdf_mppi/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select custom_msgs receiver_from_follower sender_to_follower
source install/setup.bash
```

### 1.3 Create Python virtual environment (if `.venv` does not exist)

```bash
cd /home/kklab-ur-robot/ur3e_cdf_mppi
python3 -m venv .venv
.venv/bin/pip install pybullet trimesh numpy torch scipy opencv-python
.venv/bin/pip install scikit-image mesh-to-sdf
.venv/bin/pip install "git+https://github.com/rfeinman/pytorch-minimize.git"

# Expose ROS2 Python packages to .venv
cat > .venv/lib/python3.10/site-packages/ros2_humble.pth << 'EOF'
/opt/ros/humble/local/lib/python3.10/dist-packages
/opt/ros/humble/lib/python3.10/site-packages
EOF

# Expose system-level rtde_receive to .venv
echo "/usr/lib/python3/dist-packages" > .venv/lib/python3.10/site-packages/system_extras.pth
```

---

## 2. Hand-Eye Calibration (redo only if camera is physically moved)

ChArUco board (7×5, square=40mm, marker=30mm, DICT_5X5_50) is mounted on the robot flange during calibration only.

### Step 1: Generate calibration board

```bash
cd /home/kklab-ur-robot/ur3e_cdf_mppi
python3 visual/generate_charuco.py
# → visual/charuco_board.png  (print at 100% scale, A4 landscape, mount on flange)
```

### Step 2: Start camera

```bash
bash visual/start_camera.sh
```

### Step 3: Collect calibration poses

Switch robot to Freedrive mode, move to 20+ varied poses, press SPACE to capture each pose, Q to quit.

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd /home/kklab-ur-robot/ur3e_cdf_mppi
python3 visual/collect_calib_poses.py
# → visual/calib_data/poses.npz
```

> Collection tips: avoid wrist rotations >60°, vary shoulder/elbow more (30–70° tilts), collect 20+ poses.

### Step 4: Solve calibration

```bash
python3 visual/solve_handeye.py
# → visual/calib_data/T_cam2base.npz  (used by code)
# → visual/calib_data/T_cam2base.yaml (human-readable)
```

**Calibration quality checks (confirm in output):**
- `det(R)` ≈ 1.000 (±0.001)
- Mean translation residual < 20mm
- Camera position should be ~0.8–1.0m directly in front of the robot

> Note: OpenCV methods fail when rotation diversity is insufficient (residual >500mm); the script has a built-in scipy optimization fallback.

---

## 3. Obstacle Color Calibration (redo when cube color changes)

Calibrate the cube's color so the detector can identify it.

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd /home/kklab-ur-robot/ur3e_cdf_mppi
python3 visual/pick_color.py
```

Instructions:
- A live camera window appears
- **Click on the cube surface** (click multiple spots under different lighting)
- Green mask covers the cube → color sampling succeeded
- Press `s` to save, `r` to reset, `q` to quit without saving

Result: `visual/calib_data/cube_hsv.npz`

---

## 4. Experiment Launch Procedure (each run)

Requires **4–5 terminals**, launched in order.

---

### Terminal 1 — Camera

```bash
cd /home/kklab-ur-robot/ur3e_cdf_mppi
bash visual/start_camera.sh
```

> Expected output: `color` / `depth` topics publishing

---

### Terminal 2 — Obstacle Detection

**Prerequisite:** `visual/calib_data/cube_hsv.npz` exists (see Section 3)

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd /home/kklab-ur-robot/ur3e_cdf_mppi
python3 visual/obstacle_detector.py
```

Expected output example:
```
Cube: 25000 px -> 300 3D pts -> 18 voxels
Saved 18 obstacle points -> .../obstacles.npy
```

If you see `Cube detected in image but outside robot workspace`, the cube is outside the workspace (X[-0.4,0.4] Y[-0.4,0.4] Z[0,0.4]) — move it closer to the robot.

---

### Terminal 3 — receiver (reads real robot joint angles)

```bash
source /opt/ros/humble/setup.bash
source ~/ur3e_cdf_mppi/ros2_ws/install/setup.bash
ros2 run receiver_from_follower receiver_from_follower_node --ros-args -p robot_ip:=192.168.1.102
```

> Expected output: `ReceiverFromFollowerNode started with IP: 192.168.1.102`

---

### Terminal 4 — PyBullet visualization (robot pose + obstacles)

```bash
cd /home/kklab-ur-robot/ur3e_cdf_mppi
python3 visual/visualize_obstacles_pybullet.py
```

PyBullet window shows:
- **Live robot pose** (read from RTDE every 0.05s, connects directly to robot)
- **Red spheres** = detected cube obstacle points (refreshed every 1s from obstacles.npy)

> If robot is not connected, robot shows home pose.

---

### Terminal 5 — servo_j node (only needed when executing a trajectory)

```bash
source /opt/ros/humble/setup.bash
source ~/ur3e_cdf_mppi/ros2_ws/install/setup.bash
ros2 run sender_to_follower servo_j_to_follower_node --ros-args -p robot_ip:=192.168.1.102
```

> Expected output: `Connected to UR robot at 192.168.1.102`

---

## 5. MPPI Planning and Execution

**Prerequisites:** Terminals 1–3 are running normally and `obstacles.npy` is being updated continuously.

First set the goal joint angles:

```python
# Top of planner/planner_ur3e_real.py
GOAL_Q = np.array([...], dtype=np.float32)   # goal configuration (radians)
```

Run the planner:

```bash
cd /home/kklab-ur-robot/ur3e_cdf_mppi
python3 planner/planner_ur3e_real.py
```

The script automatically:
1. Reads current joint angles from robot as start pose
2. Loads `obstacles.npy`
3. Runs MPPI planning
4. Previews trajectory in PyBullet (with obstacle spheres)
5. Waits for terminal confirmation

Terminal input:
- `s`: confirm — publish to `/target/multi_joint_trajectory` (robot executes)
- `r`: replay preview (Ctrl+C to interrupt early, returns to menu)
- `p`: replan (re-reads robot joint angles + latest obstacles.npy)
- `q`: cancel, do not send

> Planning failure does not exit — still enters preview so you can decide whether to force-send or replan.
> Trajectory is cubic-spline interpolated to 4000 points before sending; motion should be smooth.

---

## 6. Other Send Methods

### Method A: Linear joint interpolation

Edit parameters in `send_linear_traj.py` then run:

```bash
cd /home/kklab-ur-robot/ur3e_cdf_mppi
python3 send_linear_traj.py
```

### Method B: Single-point moveJ

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --once /target/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
  points: [{
    positions: [-1.57, -0.5, 1.2, -1.3, -1.57, 0.0],
    time_from_start: {sec: 0, nanosec: 0}
  }]
}"
```

---

## 7. Reading Current Robot State

### Method A: Directly via RTDE (recommended, no ROS2 dependency)

```bash
cd /home/kklab-ur-robot/ur3e_cdf_mppi
python3 read_robot_state.py
```

Output:
- `q`: 6 joint angles (radians)
- `tcp`: end-effector pose `[x, y, z, rx, ry, rz]`

### Method B: Via ROS2 topic (requires receiver to be running)

```bash
source /opt/ros/humble/setup.bash
source ~/ur3e_cdf_mppi/ros2_ws/install/setup.bash
ros2 topic echo /follower/joint_state --once
```

---

## 8. Topic Overview

| Topic | Direction | Message Type | Description |
|-------|-----------|-------------|-------------|
| `/camera/color/image_raw` | camera → subscribe | `sensor_msgs/Image` | RGB image (1280×720) |
| `/camera/depth/image_raw` | camera → subscribe | `sensor_msgs/Image` | depth image (848×480) |
| `/camera/depth/camera_info` | camera → subscribe | `sensor_msgs/CameraInfo` | depth camera intrinsics |
| `/obstacles/base_frame` | publish → RViz | `sensor_msgs/PointCloud2` | obstacle point cloud (base frame) |
| `/target/joint_trajectory` | publish → robot | `trajectory_msgs/JointTrajectory` | moveJ single-point target |
| `/target/multi_joint_trajectory` | publish → robot | `trajectory_msgs/JointTrajectory` | servoJ full trajectory |
| `/follower/joint_state` | robot → subscribe | `sensor_msgs/JointState` | real joint angles |

Joint order (index 0–5): `base, shoulder, elbow, wrist1, wrist2, wrist3`

---

## 9. Script Reference

| File | Description |
|------|-------------|
| `visual/generate_charuco.py` | Generate ChArUco calibration board image |
| `visual/collect_calib_poses.py` | Collect hand-eye calibration poses |
| `visual/solve_handeye.py` | Solve T_cam2base (scipy optimization) |
| `visual/pick_color.py` | Interactive cube color sampling, saves HSV range |
| `visual/obstacle_detector.py` | RGBD color segmentation → 3D obstacle point cloud → obstacles.npy |
| `visual/visualize_obstacles_pybullet.py` | PyBullet live display of robot pose + obstacles |
| `visual/record_video.py` | Record camera color stream to mp4 with live preview (Q to stop) |
| `pybullet_ros2_viewer.py` | PyBullet live visualization (subscribes to ROS2 joint_state) |
| `send_linear_traj.py` | Linear joint interpolation, sends to servoJ |
| `read_robot_state.py` | Read current joint angles and TCP directly via RTDE |
| `planner/planner_ur3e.py` | MPPI planner (simulation, uses CrossObstacleManager) |
| `planner/planner_ur3e_real.py` | Real-robot MPPI planner with PyBullet preview and confirmation |

---

## 10. Troubleshooting

**rviz2 crashes with Qt version conflict**
→ CoppeliaSim libraries are no longer exported globally; launch CoppeliaSim with the `coppeliasim` shell function (already fixed in `.bashrc`)

**obstacle_detector shows `No HSV range loaded`**
→ Run `python3 visual/pick_color.py` first to complete color calibration

**obstacle_detector shows `outside robot workspace`**
→ Cube is outside the workspace (X[-0.4,0.4] Y[-0.4,0.4] Z[0,0.4]); move cube closer to the robot

**PyBullet shows home pose and does not move**
→ Robot not connected; check network or switch to `pybullet_ros2_viewer.py` (subscribes to ROS2)

**Planner reports no obstacles**
→ Confirm `obstacle_detector.py` is running; check that `visual/calib_data/obstacles.npy` is being updated

**`librtde.so.1.6: cannot open shared object file`**
→ Run Section 1.1 again: `sudo ldconfig`

**`Package not found` (ROS2 node)**
→ Must `source ~/ur3e_cdf_mppi/ros2_ws/install/setup.bash` first

**Obstacle X/Y coordinates are flipped**
→ `visual/obstacle_detector.py` already applies `pts_base[:, :2] *= -1` after transform; if coordinates flip again after re-calibration, check that line

**`sequence size exceeds remaining buffer` (many terminals)**
→ Known ROS2 FastRTPS bug with large point cloud messages; noise log only, does not affect actual data transfer, can be ignored

**Launch all terminals at once**
→ `bash start_all_terminals.sh` (three modes: full / obs / calib); all Python processes use `.venv/bin/python`
