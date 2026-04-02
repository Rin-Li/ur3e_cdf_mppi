#!/bin/bash
# Start all experiment terminals for UR3e CDF-MPPI real-world experiment.
#
# Usage:
#   bash start_all_terminals.sh          # full experiment (all 6 terminals)
#   bash start_all_terminals.sh calib    # calibration only (camera + collect + solve)
#   bash start_all_terminals.sh obs      # obstacle detection + PyBullet visualizer only

ROOT_DIR="/home/kklab-ur-robot/ur3e_cdf_mppi"
ROS_SETUP="/opt/ros/humble/setup.bash"
CAMERA_WS="$HOME/ros2_ws/install/setup.bash"
ROBOT_WS="$(cd "$(dirname "$0")" && pwd)/ros2_ws/install/setup.bash"
ROBOT_IP="${ROBOT_IP:-192.168.1.102}"

MODE="${1:-full}"

# ROS2-aware python (for nodes that need cv_bridge / rclpy)
ROS_PYTHON="source '$ROS_SETUP' && source '$CAMERA_WS' && python3"

open_term() {
    local title="$1"
    local cmd="$2"
    gnome-terminal --title="$title" -- bash -lc "$cmd; exec bash" >/dev/null 2>&1 &
    sleep 0.3
}

# ── Terminal 1: Camera ─────────────────────────────────────────────────────
open_term "1 Camera" "
cd '$ROOT_DIR'
echo '=== Starting Orbbec camera ==='
bash visual/start_camera.sh
"

sleep 2   # give camera time to start before launching dependent nodes

# ── Terminal 2: Obstacle Detector ─────────────────────────────────────────
if [[ "$MODE" != "calib" ]]; then
    open_term "2 Obstacle Detector" "
source '$ROS_SETUP'
source '$CAMERA_WS'
cd '$ROOT_DIR'
echo '=== Obstacle Detector (RGBD color segmentation) ==='
echo 'Requires: visual/calib_data/cube_hsv.npz (run pick_color.py first)'
python3 visual/obstacle_detector.py
"
fi

# ── Terminal 3: Hand-Eye Calibration (calib mode only) ────────────────────
if [[ "$MODE" == "calib" ]]; then
    open_term "3 Collect Calib Poses" "
source '$ROS_SETUP'
source '$CAMERA_WS'
cd '$ROOT_DIR'
echo '=== Calibration Data Collection ==='
echo 'SPACE = capture, D = delete last, Q = save & quit'
python3 visual/collect_calib_poses.py
"
fi

# ── Terminal 4: PyBullet Visualizer (robot pose + obstacles) ──────────────
if [[ "$MODE" != "calib" ]]; then
    open_term "4 PyBullet Visualizer" "
cd '$ROOT_DIR'
echo '=== PyBullet: real robot pose (RTDE) + obstacle spheres ==='
'$ROOT_DIR/.venv/bin/python' visual/visualize_obstacles_pybullet.py
"
fi

# ── Terminal 5: Receiver + ServoJ (full mode only) ────────────────────────
if [[ "$MODE" == "full" ]]; then
    open_term "5 Receiver" "
source '$ROS_SETUP'
source '$ROBOT_WS'
echo '=== Receiver: reading robot joint angles ==='
ros2 run receiver_from_follower receiver_from_follower_node \
    --ros-args -p robot_ip:=$ROBOT_IP
"

    open_term "6 ServoJ Sender" "
source '$ROS_SETUP'
source '$ROBOT_WS'
echo '=== ServoJ: waiting for /target/multi_joint_trajectory ==='
ros2 run sender_to_follower servo_j_to_follower_node \
    --ros-args -p robot_ip:=$ROBOT_IP
"

    open_term "7 MPPI Planner" "
cd '$ROOT_DIR'
echo '=== MPPI Planner (real world) ==='
echo 'Edit GOAL_Q in planner/planner_ur3e_real.py before running.'
echo ''
read -p 'Press Enter to start planner...'
'$ROOT_DIR/.venv/bin/python' planner/planner_ur3e_real.py
"
fi

echo ""
echo "Launched terminals (mode=$MODE):"
if [[ "$MODE" == "calib" ]]; then
    echo "  1  Camera"
    echo "  3  Collect Calib Poses  ← SPACE to capture, Q to save"
    echo ""
    echo "After collecting poses, run in any terminal:"
    echo "  cd $ROOT_DIR && python3 visual/solve_handeye.py"
elif [[ "$MODE" == "obs" ]]; then
    echo "  1  Camera"
    echo "  2  Obstacle Detector"
    echo "  4  PyBullet Visualizer"
    echo ""
    echo "If cube_hsv.npz missing, run first:"
    echo "  source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash"
    echo "  cd $ROOT_DIR && python3 visual/pick_color.py"
else
    echo "  1  Camera"
    echo "  2  Obstacle Detector"
    echo "  4  PyBullet Visualizer  ← verify robot pose + obstacles here"
    echo "  5  Receiver"
    echo "  6  ServoJ Sender"
    echo "  7  MPPI Planner         ← press Enter when ready to plan"
    echo ""
    echo "Checklist before pressing 's' in planner:"
    echo "  [ ] Obstacle detector showing points"
    echo "  [ ] PyBullet shows robot at correct pose + red spheres at cube"
    echo "  [ ] ServoJ sender shows 'Connected to UR robot'"
fi
