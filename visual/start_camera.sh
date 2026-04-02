#!/bin/bash
# Start Orbbec Gemini 335L camera via ROS2 (USB 3.0)
# Usage: bash visual/start_camera.sh

set -e

ROS_DISTRO="humble"
WS="$HOME/ros2_ws"

echo "[INFO] Sourcing ROS2 ${ROS_DISTRO} and workspace..."
source /opt/ros/${ROS_DISTRO}/setup.bash
source "${WS}/install/setup.bash"

echo "[INFO] Checking USB connection..."
if ! lsusb | grep -qi "2bc5"; then
    echo "[WARN] Orbbec device not detected via lsusb. Make sure the camera is plugged into a USB 3.0 port."
fi

echo "[INFO] Launching Gemini 335L camera node..."
ros2 launch orbbec_camera gemini_330_series.launch.py \
    color_width:=1280 \
    color_height:=720 \
    color_fps:=30 \
    depth_width:=848 \
    depth_height:=480 \
    depth_fps:=30 \
    enable_point_cloud:=true \
    enable_colored_point_cloud:=true
