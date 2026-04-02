# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **6-DOF UR3e robot motion planning system** combining a learned Configuration-Dependent Distance Field (CDF) with Model Predictive Path Integral (MPPI) planning for real-time, obstacle-aware trajectory generation.

## Key Commands

### Data Generation
```bash
# Generate training data (grid^3 task-space points, N configs each)
python cdf/data_generator.py   # outputs data/data.npy
```

### CDF Training
```bash
python cdf/nn_cdf.py           # loads data/data.npy, trains MLP, saves data/model_best.pt
```

### Motion Planning
```bash
python planner/planner_ur3e.py     # runs 10 MPPI trials, saves to saved_traj_ur3e/
```

### Visualization
```bash
python planner/visualize_obstacles.py                          # show start/goal + obstacles
python planner/visualize_planner.py --dir saved_traj_ur3e/    # replay trajectories
python planner/visualize_planner.py --file path/to/trial_0.npz
```

## Architecture

The system has three layers that build on each other:

### 1. Kinematics Layer (`ur3e/ur3e.py`)
`URRobot` is a PyTorch `nn.Module` implementing **batched forward kinematics** for all 7 links. It transforms mesh vertices/normals and is used by both the CDF training pipeline and the MPPI planner. Mesh STL files live in `ur3e/model/`.

### 2. Distance Field Layer
Two complementary distance field models:
- **RDF** (`rdf/ur_rdf.py` — `BPSDF` class): Pre-trained ball-projection SDF, used as ground truth during data generation and as a collision verifier during planning. Model file: `rdf/models/BP_8.pt`. This module is symlinked/imported from `/home/kklab-ur-robot/ur_sdf/RDF/`.
- **CDF** (`cdf/nn_cdf.py` — `CDF` class, `cdf/mlp.py` — `MLPRegression`): Learned MLP (input: 9D = 3D position + 6D joint angles with NERF encoding; output: 1D distance). Trained with MSE + Eikonal + tension + gradient losses. Model file: `cdf/model/model_best.pt`.

The CDF training data format (`cdf/data/data.npy` / `cdf/data/data.pt`) is a dict keyed by task-space point index, each value having `'x'` (3D coords) and `'q'` (N×6 zero-level-set configurations).

### 3. MPPI Planner (`planner/planner_ur3e.py`)
`MPPIUR3e` runs joint-space trajectory optimization:
- **Fast cost evaluation**: uses CDF `inference_d_wrt_q()` for 200-sample batch cost
- **Collision verification**: uses RDF `get_whole_body_sdf_batch()` before accepting each step
- **Obstacle geometry**: `CrossObstacleManager` (`planner/obstacle_manager.py`) — cross-shaped obstacle at (0.3, 0.0, 0.2) composed of 0.05m-radius spheres
- Trajectories saved as `.npz` files in `saved_traj_ur3e/`

### PyBullet Interface (`ur3e/ur3e_pybullet.py`)
`UR3ePyBullet` wraps PyBullet for visualization, collision detection, and trajectory playback. Used in visualization scripts and for ground-truth distance computation during evaluation.

## Project Structure

```
ur3e_cdf_mppi/
├── ur3e/                    # robot kinematics only
│   ├── ur3e.py              # URRobot batched FK module
│   ├── ur3e_pybullet.py     # PyBullet visualization wrapper
│   └── model/               # STL mesh files
├── cdf/                     # CDF learning package
│   ├── mlp.py               # MLPRegression network
│   ├── nn_cdf.py            # CDF class: data processing + training + inference
│   └── data_generator.py    # DataGenerator: zero-level-set q search
├── rdf/                     # pre-trained ball-projection SDF
│   ├── ur_rdf.py            # BPSDF class
│   ├── utils.py             # transform helpers
│   ├── models/              # BP_8.pt model weights
│   └── data/                # SDF point data
├── planner/                 # MPPI planner + visualization
│   ├── planner_ur3e.py      # MPPIUR3e planner
│   ├── obstacle_manager.py  # CrossObstacleManager
│   ├── visualize_obstacles.py
│   └── visualize_planner.py
└── cdf/
    ├── mlp.py
    ├── nn_cdf.py
    ├── data_generator.py
    ├── data/                # raw + processed training data (data.npy, data.pt)
    └── model/               # CDF checkpoints (model_best.pt, model_dict.pt)
```

## Important Details

- **Workspace**: X[-0.4, 0.4], Y[-0.4, 0.4], Z[0, 0.4] for CDF training; planner uses fixed start/goal `FIXED_PAIRS` in `planner_ur3e.py` (only joint[0] differs by π)
- **External dependency**: `RDF/ur_rdf.py` is from `/home/kklab-ur-robot/ur_sdf/RDF/`; ensure that path exists
- **Device**: All code auto-selects CUDA if available (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`)
- **Mixed precision**: CDF training uses `torch.cuda.amp` for speed
- **MPPI params** (lines 33–86 of `planner_ur3e.py`): `NUM_TRIALS=10`, `MAX_STEPS=3000`, `ALG_N_SAMPLES=200`, `CONS_U=1.5 rad/s`
- **Run from project root**: all scripts now use `_ROOT`-relative paths; run as `python cdf/nn_cdf.py` or `python planner/planner_ur3e.py` from project root

## Dependencies

All dependencies live in `.venv` (Python 3.10). Key packages:
- `torch`, `numpy`, `scipy`, `pybullet`, `trimesh`, `opencv-python`
- `torchmin` (installed from `https://github.com/rfeinman/pytorch-minimize.git`)
- `scikit-image`, `mesh-to-sdf`
- `pytorch3d` — only needed for `cdf/data_generator.py` (training data), not for real-world pipeline
- `rtde_receive` — system-level package at `/usr/lib/python3/dist-packages/`, exposed to `.venv` via `.venv/lib/python3.10/site-packages/system_extras.pth`
- ROS2 Humble packages — exposed via `.venv/lib/python3.10/site-packages/ros2_humble.pth`

Always use `.venv/bin/python` to run scripts, not system python or conda.

---

## Sim-to-Real Pipeline

The system transitions from simulation to reality in three phases. All real-world files live in `visual/` and `planner/`.

### Hardware
- **Robot**: UR3e at `192.168.1.102`, controlled via RTDE (`rtde_receive`, `rtde_control`)
- **Camera**: Orbbec Gemini 335L, fixed in environment (**eye-to-hand**), USB 3.0
- **ROS2 workspace**: `~/ros2_ws` (Humble)

### Phase 1 — Hand-Eye Calibration

**One-time setup. Re-run only if camera is physically moved.**

ChArUco board (7×5, square=40mm, marker=30mm, `DICT_5X5_50`) is rigidly mounted on the robot flange during calibration, then removed.

```bash
python3 visual/generate_charuco.py         # → visual/charuco_board.png
bash visual/start_camera.sh
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
python3 visual/collect_calib_poses.py      # → visual/calib_data/poses.npz
python3 visual/solve_handeye.py            # → visual/calib_data/T_cam2base.npz + .yaml
```

### Phase 2 — Real-Time Obstacle Detection

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
python3 visual/obstacle_detector.py
```

- Loads `T_cam2base`, transforms depth point cloud to robot base frame
- **X/Y axes are flipped** after transform (`pts_base[:, :2] *= -1`) — this corrects for camera mounting orientation
- Filters to workspace X[-0.4,0.4] × Y[-0.4,0.4] × Z[0,0.4], voxel downsamples at 0.02m
- Saves `visual/calib_data/obstacles.npy`

### Phase 3 — Real-World MPPI Planner

```bash
.venv/bin/python planner/planner_ur3e_real.py
```

Key behaviour:
- Reads `q_start` from robot via RTDE at planning time
- Loads `obstacles.npy` from camera
- After planning, runs **cubic spline interpolation** (`resample_trajectory`) to 4000 points before preview/send
- PyBullet preview plays 200 frames at real-time speed (~8s); Ctrl+C skips to menu
- Menu: `s` send · `r` replay · `p` replan (re-reads robot + obstacles) · `q` cancel
- Planning failure does **not** exit — shows preview and lets user decide

### Full Run Order (Real World)

```bash
bash start_all_terminals.sh          # full (7 terminals)
bash start_all_terminals.sh obs      # obs debug only (camera + detector + PyBullet)
bash start_all_terminals.sh calib    # calibration only
```

### Real-World Project Structure

```
visual/
├── start_camera.sh
├── generate_charuco.py
├── collect_calib_poses.py
├── solve_handeye.py
├── obstacle_detector.py             # depth cloud → base-frame obstacle points
├── visualize_obstacles_pybullet.py  # real-time robot pose + obstacles in PyBullet
├── pick_color.py                    # HSV color calibration for cube detection
├── record_video.py                  # record color stream to mp4 with live preview
└── calib_data/
    ├── poses.npz
    ├── T_cam2base.npz               # calibration result
    ├── T_cam2base.yaml
    ├── cube_hsv.npz                 # cube HSV color range
    └── obstacles.npy                # current obstacle point cloud (N, 3)

planner/
├── planner_ur3e.py                  # simulation planner
├── planner_ur3e_real.py             # real-world planner
└── obstacle_manager.py

recordings/                          # video recordings (auto-created)
```
