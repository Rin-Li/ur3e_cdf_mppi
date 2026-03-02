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

`torch`, `pybullet`, `trimesh`, `numpy`, `pytorch3d` (for farthest point sampling in data processing), `torchmin` (L-BFGS for data generation)
