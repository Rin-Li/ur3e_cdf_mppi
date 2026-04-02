# Orbbec Gemini 335L Camera Setup

## Hardware

| Item | Detail |
|------|--------|
| Camera model | Orbbec Gemini 335L |
| Serial number | CP2535300076 |
| Firmware | 1.4.60 |
| Required connection | **USB 3.0 / USB 3.2** (blue port) |

> Using a USB 2.0 port will cause bandwidth errors or failed depth stream initialization.

---

## Prerequisites

1. **ROS2 Humble** installed at `/opt/ros/humble/`
2. **orbbec_camera** ROS2 package built in `~/ros2_ws`
3. Camera physically connected to a **USB 3.0 (blue) port**

Verify the device is detected:
```bash
lsusb | grep 2bc5
# Expected output example:
# Bus 002 Device 003: ID 2bc5:0660 Orbbec ...
```

---

## Quick Start

From the project root, run:

```bash
bash visual/start_camera.sh
```

Or manually:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch orbbec_camera gemini_330_series.launch.py \
    color_width:=1280 color_height:=720 color_fps:=30 \
    depth_width:=848  depth_height:=480 depth_fps:=30 \
    enable_point_cloud:=true \
    enable_colored_point_cloud:=true
```

---

## Published Topics

Once the node is running, the following topics become available:

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB color image (1280×720 @ 30 fps) |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Depth image (848×480 @ 30 fps) |
| `/camera/depth/points` | `sensor_msgs/PointCloud2` | Uncolored point cloud |
| `/camera/depth_registered/points` | `sensor_msgs/PointCloud2` | Color-registered point cloud |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | Color camera intrinsics |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | Depth camera intrinsics |

Check active topics:
```bash
ros2 topic list | grep camera
```

---

## Visualization with RViz2

```bash
# In a new terminal
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run rviz2 rviz2
```

Inside RViz2:
- Click **Add** → **By topic**
- Add `/camera/color/image_raw` (type: **Image**) to see the color stream
- Add `/camera/depth/points` (type: **PointCloud2**) to see the point cloud
- Set **Fixed Frame** to `camera_link` (or `map` if TF is published)

---

## Verify Stream Data

```bash
# Check color image is publishing
ros2 topic hz /camera/color/image_raw

# Check depth image is publishing
ros2 topic hz /camera/depth/image_raw

# Check point cloud
ros2 topic hz /camera/depth/points

# Print one camera info message
ros2 topic echo /camera/color/camera_info --once
```

---

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Device not found | Not plugged in or USB 2.0 port | Use blue USB 3.0 port |
| Depth stream fails to start | Bandwidth issue | Disconnect other USB 3.0 devices |
| `image_transport` compressed errors | `image-transport-plugins` not installed | Safe to ignore; raw topics still work |
| Node crashes on launch | Another orbbec node already running | `pkill -f orbbec_camera`, then relaunch |

Kill a stale camera process:
```bash
pkill -f "component_container"
# or
ros2 lifecycle set /camera shutdown  # if lifecycle node
```

---

## Stream Parameters Reference

Key parameters for `gemini_330_series.launch.py`:

| Parameter | Default | Options |
|-----------|---------|---------|
| `color_width` | 1280 | 640, 1280 |
| `color_height` | 720 | 480, 720 |
| `color_fps` | 30 | 15, 30 |
| `depth_width` | 848 | 424, 848 |
| `depth_height` | 480 | 240, 480 |
| `depth_fps` | 30 | 15, 30 |
| `enable_point_cloud` | true | true/false |
| `enable_colored_point_cloud` | true | true/false |
