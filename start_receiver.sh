#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/ros2_ws/install/setup.bash"
ros2 run receiver_from_follower receiver_from_follower_node --ros-args -p robot_ip:=192.168.1.102
