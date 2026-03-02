"""
UR3e PyBullet Kinematics Wrapper
This module provides a PyBullet-based kinematics interface for UR3e robot.
It uses custom forward kinematics from ur3e.py but leverages PyBullet for
visualization, collision detection, and distance computation.
"""

import numpy as np
import os
import sys
import torch
import pybullet as p
import trimesh
from typing import List, Tuple, Optional, Union

# Handle both package import and direct execution
try:
    from .ur3e import URRobot
except ImportError:
    # If running as script, add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ur3e import URRobot

class UR3ePyBullet:
    """
    PyBullet wrapper for UR3e robot using custom kinematics.
    
    This class combines your custom UR3e kinematics with PyBullet's
    visualization and collision detection capabilities.
    """
    
    def __init__(self, device='cuda', gui=False, mesh_dir=None):
        """
        Initialize UR3e PyBullet interface.
        
        Args:
            device: torch device ('cuda' or 'cpu')
            gui: whether to show PyBullet GUI
            mesh_dir: directory containing STL mesh files (default: ur3e/model)
        """
        self.device = device
        self.gui_mode = gui
        
        # Store reference to pybullet module for easy access
        self.p = p
        
        # Initialize your custom UR3e kinematics
        self.ur3e = URRobot(device)
        
        # Set mesh directory
        if mesh_dir is None:
            mesh_dir = os.path.join(os.path.dirname(__file__), 'model')
        self.mesh_dir = mesh_dir
        
        # Initialize PyBullet
        self._init_pybullet()
        
        # Load robot meshes into PyBullet
        self._load_robot_meshes()
        
        # Current joint configuration
        self.current_joint_angles = np.zeros(6)
        
    def _init_pybullet(self):
        """Initialize PyBullet physics engine."""
        if self.gui_mode:
            # Try to fix OpenGL context issues
            import platform
            if platform.system() == 'Linux':
                # Set environment variables to help with OpenGL issues
                os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')
                os.environ.setdefault('MESA_GLSL_VERSION_OVERRIDE', '330')
            
            try:
                self.physics_client = p.connect(p.GUI)
                # Set camera position for better view
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.5,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=[0, 0, 0.3]
                )
                print("PyBullet GUI mode initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize GUI mode: {e}")
                print("Falling back to DIRECT mode (no visualization)")
                self.gui_mode = False
                self.physics_client = p.connect(p.DIRECT)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setGravity(0, 0, -9.8)
        
        # Add ground plane if in GUI mode
        if self.gui_mode:
            try:
                import pybullet_data
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.loadURDF("plane.urdf")
            except:
                # If pybullet_data not available, create simple ground
                ground_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2, 2, 0.01])
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_shape, 
                                basePosition=[0, 0, -0.01])
    
    def _load_robot_meshes(self):
        """Load UR3e mesh files as separate bodies in PyBullet."""
        self.link_names = ['base', 'shoulder', 'upperarm', 'forearm', 'wrist1', 'wrist2', 'wrist3']
        self.link_body_ids = []
        self.link_visual_ids = []
        
        # Color scheme for different links
        colors = [
            [0.8, 0.2, 0.2, 1.0],  # base - red
            [0.2, 0.8, 0.2, 1.0],  # shoulder - green
            [0.2, 0.2, 0.8, 1.0],  # upperarm - blue
            [0.8, 0.8, 0.2, 1.0],  # forearm - yellow
            [0.8, 0.2, 0.8, 1.0],  # wrist1 - magenta
            [0.2, 0.8, 0.8, 1.0],  # wrist2 - cyan
            [0.5, 0.5, 0.5, 1.0],  # wrist3 - gray
        ]
        
        print(f"Loading UR3e robot from {self.mesh_dir}...")
        
        for i, link_name in enumerate(self.link_names):
            stl_file = os.path.join(self.mesh_dir, f'{link_name}.stl')
            
            if not os.path.exists(stl_file):
                print(f"Warning: {stl_file} not found, skipping")
                continue
            
            # Load mesh using trimesh
            mesh = trimesh.load(stl_file)
            
            # Create collision shape from mesh
            collision_shape = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=stl_file,
                meshScale=[1, 1, 1]
            )
            
            # Create visual shape from mesh
            visual_shape = p.createVisualShape(
                p.GEOM_MESH,
                fileName=stl_file,
                meshScale=[1, 1, 1],
                rgbaColor=colors[i]
            )
            
            # Create multi-body
            body_id = p.createMultiBody(
                baseMass=0,  # Static (no dynamics)
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0, 0, 0]
            )
            
            self.link_body_ids.append(body_id)
            self.link_visual_ids.append(visual_shape)
            print(f"  Loaded {link_name}: body_id={body_id}")
        
        print(f"UR3e robot loaded with {len(self.link_body_ids)} links")
    
    def set_joint_angles(self, joint_angles: Union[np.ndarray, torch.Tensor, List[float]]):
        """
        Set robot joint configuration using custom kinematics.
        
        Args:
            joint_angles: 6 joint angles (radians)
        """
        # Convert to numpy array
        if isinstance(joint_angles, torch.Tensor):
            joint_angles = joint_angles.cpu().numpy()
        elif isinstance(joint_angles, list):
            joint_angles = np.array(joint_angles)
        
        joint_angles = joint_angles.flatten()
        assert len(joint_angles) == 6, "UR3e requires 6 joint angles"
        
        self.current_joint_angles = joint_angles
        
        # Use custom UR3e kinematics to compute transformations
        pose = torch.eye(4).unsqueeze(0).to(self.device).float()
        theta = torch.from_numpy(joint_angles).unsqueeze(0).to(self.device).float()
        
        # Get transformation for each link
        trans_list = self.ur3e.get_transformations_each_link(pose, theta)
        
        # Update PyBullet bodies
        for i, body_id in enumerate(self.link_body_ids):
            if i < len(trans_list):
                trans = trans_list[i].squeeze().cpu().numpy()
                position = trans[:3, 3]
                rotation_matrix = trans[:3, :3]
                quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
                
                p.resetBasePositionAndOrientation(body_id, position, quaternion)
    
    def get_joint_angles(self) -> np.ndarray:
        """Get current joint angles."""
        return self.current_joint_angles.copy()
    
    def compute_closest_distance(self, query_point: Union[np.ndarray, torch.Tensor, List[float]]) -> dict:
        """
        Compute closest distance from a query point to the robot surface.
        
        Args:
            query_point: 3D point coordinates
            
        Returns:
            dict with keys:
                - 'distance': minimum distance to robot surface (float)
                - 'position_on_robot': closest point on robot surface (np.ndarray)
                - 'link_index': index of closest link (int)
                - 'query_point': query point coordinates (np.ndarray)
        """
        # Convert to numpy
        if isinstance(query_point, torch.Tensor):
            query_point = query_point.cpu().numpy()
        elif isinstance(query_point, list):
            query_point = np.array(query_point)
        
        query_point = query_point.flatten()[:3]
        
        # Create temporary sphere at query point
        temp_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.001)
        temp_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=temp_shape,
            basePosition=query_point
        )
        
        # Find closest point among all links
        min_distance = float('inf')
        closest_point = None
        closest_link_idx = -1
        
        for i, body_id in enumerate(self.link_body_ids):
            closest_points = p.getClosestPoints(
                bodyA=body_id,
                bodyB=temp_body,
                distance=10.0
            )
            
            if len(closest_points) > 0:
                contact_distance = closest_points[0][8]
                if abs(contact_distance) < abs(min_distance):
                    min_distance = contact_distance
                    closest_point = np.array(closest_points[0][5])  # position on robot
                    closest_link_idx = i
        
        # Clean up
        p.removeBody(temp_body)
        
        # Return dict format for consistency
        if closest_point is not None:
            return {
                'distance': min_distance,
                'position_on_robot': closest_point,
                'link_index': closest_link_idx,
                'query_point': query_point
            }
        else:
            return None
    
    def check_self_collision(self) -> bool:
        """
        Check if robot is in self-collision.
        
        Returns:
            True if in collision, False otherwise
        """
        for i, body_i in enumerate(self.link_body_ids):
            for j, body_j in enumerate(self.link_body_ids):
                if i >= j:
                    continue
                
                # Skip adjacent links (they naturally touch)
                if abs(i - j) <= 1:
                    continue
                
                contacts = p.getClosestPoints(body_i, body_j, distance=0.0)
                if len(contacts) > 0 and contacts[0][8] < 0:
                    return True
        
        return False
    
    def visualize_point(self, point: Union[np.ndarray, List[float]], 
                       color: List[float] = [1, 0, 0, 1],
                       size: float = 0.02,
                       label: str = None) -> int:
        """
        Add a visual marker sphere at a point.
        
        Args:
            point: 3D coordinates
            color: RGBA color [r, g, b, a]
            size: sphere radius (renamed from radius for consistency)
            label: optional text label (ignored for now)
            
        Returns:
            body_id of created sphere (or -1 if no GUI)
        """
        if not self.gui_mode:
            return -1
        
        if isinstance(point, torch.Tensor):
            point = point.cpu().numpy()
        point = np.array(point).flatten()[:3]
        
        # Ensure RGBA has 4 components
        if len(color) == 3:
            color = list(color) + [1.0]
        
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=size,
            rgbaColor=color
        )
        
        body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=point
        )
        
        return body_id
    
    def visualize_line(self, point_a: Union[np.ndarray, List[float]],
                      point_b: Union[np.ndarray, List[float]],
                      color: List[float] = [1, 1, 0],
                      width: float = 2.0,
                      lifetime: float = 0) -> int:
        """
        Draw a line between two points.
        
        Args:
            point_a: start point
            point_b: end point
            color: RGB color [r, g, b]
            width: line width
            lifetime: how long to display (0 = forever)
            
        Returns:
            line_id
        """
        if not self.gui_mode:
            return -1
        
        if isinstance(point_a, torch.Tensor):
            point_a = point_a.cpu().numpy()
        if isinstance(point_b, torch.Tensor):
            point_b = point_b.cpu().numpy()
        
        point_a = np.array(point_a).flatten()[:3]
        point_b = np.array(point_b).flatten()[:3]
        
        line_id = p.addUserDebugLine(
            point_a,
            point_b,
            lineColorRGB=color,
            lineWidth=width,
            lifeTime=lifetime
        )
        
        return line_id
    
    def visualize_text(self, text: str, position: Union[np.ndarray, List[float]],
                      color: List[float] = [1, 1, 1],
                      size: float = 1.5,
                      lifetime: float = 0) -> int:
        """
        Add text label at a position.
        
        Args:
            text: text to display
            position: 3D position
            color: RGB color
            size: text size
            lifetime: how long to display (0 = forever)
            
        Returns:
            text_id
        """
        if not self.gui_mode:
            return -1
        
        if isinstance(position, torch.Tensor):
            position = position.cpu().numpy()
        position = np.array(position).flatten()[:3]
        
        text_id = p.addUserDebugText(
            text,
            position,
            textColorRGB=color,
            textSize=size,
            lifeTime=lifetime
        )
        
        return text_id
    
    def clear_visualization(self):
        """Clear all visual markers (points, lines, text)."""
        if self.gui_mode:
            p.removeAllUserDebugItems()
    
    def get_link_pose(self, link_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get position and orientation of a specific link.
        
        Args:
            link_index: index of link (0-6)
            
        Returns:
            position: [x, y, z]
            orientation: quaternion [x, y, z, w]
        """
        if link_index >= len(self.link_body_ids):
            raise ValueError(f"Link index {link_index} out of range")
        
        body_id = self.link_body_ids[link_index]
        pos, orn = p.getBasePositionAndOrientation(body_id)
        
        return np.array(pos), np.array(orn)
    
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get end effector (wrist3) pose.
        
        Returns:
            position: [x, y, z]
            orientation: quaternion [x, y, z, w]
        """
        return self.get_link_pose(-1)  # Last link is end effector
    
    def reset(self, joint_angles: Optional[Union[np.ndarray, List[float]]] = None):
        """
        Reset robot to home position or specified configuration.
        
        Args:
            joint_angles: optional joint configuration (default: all zeros)
        """
        if joint_angles is None:
            joint_angles = np.zeros(6)
        
        self.set_joint_angles(joint_angles)
        self.clear_visualization()
    
    def step_simulation(self):
        """Step the simulation (useful if dynamics are enabled)."""
        p.stepSimulation()
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> List[float]:
        """
        Convert rotation matrix to quaternion [x, y, z, w].
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            quaternion [x, y, z, w]
        """
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
    
    def disconnect(self):
        """Explicitly disconnect PyBullet."""
        try:
            p.disconnect(self.physics_client)
        except:
            pass
    
    def __del__(self):
        """Cleanup PyBullet connection."""
        self.disconnect()


def demo():
    """Demonstration of UR3ePyBullet usage."""
    import time
    
    print("="*60)
    print("UR3e PyBullet Kinematics Demo")
    print("="*60)
    
    # Create robot interface with GUI
    robot = UR3ePyBullet(device='cuda', gui=True)
    
    # Demo 1: Set different joint configurations
    print("\nDemo 1: Setting different joint configurations...")
    configs = [
        [0, 0, 0, 0, 0, 0],
        [0, -np.pi/4, np.pi/2, -np.pi/4, np.pi/2, 0],
        [np.pi/2, -np.pi/3, np.pi/3, -np.pi/2, np.pi/2, np.pi/4],
    ]
    
    for i, config in enumerate(configs):
        print(f"Configuration {i+1}: {config}")
        robot.set_joint_angles(config)
        time.sleep(1.5)
    
    # Demo 2: Distance computation
    print("\nDemo 2: Computing distances to query points...")
    robot.reset([0, -0.5, 0.5, -1.0, 0, 0])
    
    query_points = [
        [0.3, 0.2, 0.1],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.2],
    ]
    
    for point in query_points:
        result = robot.compute_closest_distance(point)
        print(f"\nQuery: {point}")
        print(f"  Distance: {result['distance']:.4f}m")
        print(f"  Closest point: {result['position_on_robot']}")
        print(f"  Closest link: {robot.link_names[result['link_index']]}")
        
        # Visualize
        robot.visualize_point(point, color=[1, 0, 0, 1])
        robot.visualize_point(result['position_on_robot'], color=[0, 1, 0, 1])
        robot.visualize_line(point, result['position_on_robot'])
        robot.visualize_text(f"d={result['distance']:.3f}", 
                           [(point[i] + result['position_on_robot'][i])/2 for i in range(3)])
        
        time.sleep(1.5)
    
    # Demo 3: Trajectory following
    print("\nDemo 3: Following a circular trajectory...")
    robot.clear_visualization()
    
    trajectory = []
    for t in np.linspace(0, 2*np.pi, 50):
        q = [
            np.sin(t) * 0.5,
            -np.pi/4 + np.cos(t) * 0.3,
            np.pi/2,
            -np.pi/4,
            np.pi/2,
            t
        ]
        trajectory.append(q)
    
    for config in trajectory:
        robot.set_joint_angles(config)
        time.sleep(0.05)
    
    print("\nDemo complete! Press Enter to exit...")
    input()


if __name__ == "__main__":
    demo()
