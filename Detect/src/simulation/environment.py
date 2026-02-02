import pybullet as p
import pybullet_data
import numpy as np
import yaml
from pathlib import Path
import math
import random

class RobotEnvironment:
    
    
    def __init__(self, config_path="config/config.yaml", gui=True):
        
        self.config = self._load_config(config_path)
        self.gui = gui
        
        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Physics settings
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1/240)

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        # Storage for spawned objects
        self.objects = {}  # {object_id: {class, position, ...}}
        self.robot_id = None
        
        # CRITICAL FIX: Load camera parameters for camera-relative spawning
        self.cam_config = self.config['camera']
        self.cam_pos = np.array(self.cam_config['position'])
        self.cam_target = np.array(self.cam_config['target'])
        self.cam_up = np.array(self.cam_config['up'])
        
        # Compute camera frame vectors (same as training)
        self._compute_camera_frame()
        
        # Load spawning parameters (should match training)
        spawn_config = self.config.get('spawning', self.config.get('data_generation', {}))
        self.spawn_distances = spawn_config.get('distances', [0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
        self.spawn_angles = spawn_config.get('angles', [-20, -10, 0, 10, 20])
        self.spawn_height_offsets = spawn_config.get('height_offsets', [-0.1, 0.0, 0.1])
        
        print(f"✓ PyBullet environment initialized (Client: {self.client})")
        print(f"  Camera-relative spawning enabled")
        print(f"  Distance range: {min(self.spawn_distances):.1f}m - {max(self.spawn_distances):.1f}m")
        print(f"  Angle range: {min(self.spawn_angles)}° - {max(self.spawn_angles)}°")
    
    def _load_config(self, config_path):
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _compute_camera_frame(self):
        
        # Forward vector (camera looking direction)
        forward = self.cam_target - self.cam_pos
        self.forward = forward / np.linalg.norm(forward)
        
        # Right vector (perpendicular to forward and up)
        right = np.cross(self.forward, self.cam_up)
        self.right = right / np.linalg.norm(right)
        
        # Up vector (perpendicular to forward and right)
        up = np.cross(self.right, self.forward)
        self.up = up / np.linalg.norm(up)
        
        print(f"  Camera frame computed:")
        print(f"    Forward: {self.forward}")
        print(f"    Right: {self.right}")
        print(f"    Up: {self.up}")
    
    def _random_position_world_planar(self, object_type):
        
        import math
        import random

        # Camera position
        cam_x, cam_y, _ = self.cam_pos

        # Sample distance & angle
        distance = random.uniform(
            self.config['spawn']['min_distance'],
            self.config['spawn']['max_distance']
        )
        angle_deg = random.uniform(
            self.config['spawn']['min_angle'],
            self.config['spawn']['max_angle']
        )
        angle = math.radians(angle_deg)

        # Planar offset
        dx = distance * math.cos(angle)
        dy = distance * math.sin(angle)

        # Object height (CRITICAL)
        obj_cfg = self.config['objects'][object_type]

        if object_type == "sphere":
            object_height = obj_cfg["diameter"] / 2
        elif object_type == "cylinder":
            object_height = obj_cfg["height"] / 2
        else:
            object_height = obj_cfg["size"][2] / 2

        GROUND_Z = 0.0

        return [
            cam_x + dx,
            cam_y + dy,
            GROUND_Z + object_height
        ]

    def spawn_object(self, object_type, position=None, orientation=None):
       
        if object_type not in self.config['objects']:
            raise ValueError(f"Unknown object type: {object_type}")
        
        obj_config = self.config['objects'][object_type]
        
        
        if position is None:
            position = self._random_position_world_planar(object_type)
        
        # Default orientation (no rotation)
        if orientation is None:
            orientation = [0, 0, 0, 1]  #
        
        # Create visual and collision shapes based on type
        if object_type == 'cube':
            half_extents = [s/2 for s in obj_config['size']]
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=obj_config['color']
            )
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=half_extents
            )
        
        elif object_type == 'sphere':
            radius = obj_config['diameter'] / 2
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=radius,
                rgbaColor=obj_config['color']
            )
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                radius=radius
            )
        
        elif object_type == 'cylinder':
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=obj_config['radius'],
                length=obj_config['height'],
                rgbaColor=obj_config['color']
            )
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=obj_config['radius'],
                height=obj_config['height']
            )
        
        elif object_type == 'pyramid':
            # Use cone instead of missing mesh file
            base_size = obj_config['size'][0]
            height = obj_config['size'][2]
            
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                vertices=[
                    [0, 0, height],  # Top vertex
                    [base_size/2, base_size/2, 0],   # Base corners
                    [base_size/2, -base_size/2, 0],
                    [-base_size/2, -base_size/2, 0],
                    [-base_size/2, base_size/2, 0],
                ],
                indices=[
                    0, 1, 2, 
                    0, 2, 3,
                    0, 3, 4,
                    0, 4, 1,
                    1, 2, 3,  
                    1, 3, 4,
                ],
                rgbaColor=obj_config['color']
            )
            # Use box for collision (simpler)
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[s/2 for s in obj_config['size']]
            )
        
        elif object_type == 'box':
            half_extents = [s/2 for s in obj_config['size']]
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=obj_config['color']
            )
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=half_extents
            )
        
        # Create multi-body
        obj_id = p.createMultiBody(
            baseMass=0.5,  # 500g
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        # Store object info
        self.objects[obj_id] = {
            'type': object_type,
            'class_id': obj_config['class_id'],
            'position': position,
            'orientation': orientation,
            'visited': False
        }
        
        return obj_id
    
    def _random_position_camera_relative(self):

        # Use spawning ranges from config
        distance = np.random.choice(self.spawn_distances)
        angle_deg = np.random.choice(self.spawn_angles)
        height_offset = np.random.choice(self.spawn_height_offsets)
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle_deg)
        
        # Calculate position relative to camera
        position = (self.cam_pos + 
                distance * np.cos(angle_rad) * self.forward +  # Forward distance
                distance * np.sin(angle_rad) * self.right +    # Sideways (angle)
                height_offset * self.up)                        # Height variation
        
        # Ensure object stays at reasonable height
        position[2] = np.clip(position[2], 0.3, 1.5)  # Between 30cm and 1.5m
        
        # Validate position is in front of camera
        to_object = position - self.cam_pos
        actual_distance = np.linalg.norm(to_object)
        print(f"  DEBUG spawn: target={distance:.2f}m, actual={actual_distance:.2f}m, pos={position}")
        
        # Log if position seems problematic
        if actual_distance < 0.7 or actual_distance > 2.0:
            print(f"    ⚠ Warning: Object spawned at distance {actual_distance:.2f}m (target: {distance:.2f}m)")
        
        return position.tolist()
    
    def _random_position(self):
       
        print("⚠ WARNING: Using deprecated world-frame spawning!")
        print("  Use _random_position_camera_relative() for camera-aligned spawning")
        x = np.random.uniform(0.5, 4.0)
        y = np.random.uniform(-2.0, 2.0)
        z = 0.5
        return [x, y, z]
    
    def spawn_random_scene(self, num_each_type=1, settle_steps=100, min_separation=0.4):
        
        object_types = list(self.config['objects'].keys())
        
        print(f"\nSpawning scene with {num_each_type} of each object type...")
        
        occupied_positions = []  # Track spawned positions
        
        for obj_type in object_types:
            for i in range(num_each_type):
                # Try to find non-overlapping position
                max_attempts = 20
                for attempt in range(max_attempts):
                    # Generate random position
                    position = self._random_position_camera_relative()
                    
                    # Check if too close to existing objects
                    is_clear = True
                    for occupied_pos in occupied_positions:
                        distance = np.linalg.norm(np.array(position) - np.array(occupied_pos))
                        if distance < min_separation:
                            is_clear = False
                            break
                    
                    if is_clear:
                        break  # Good position found
                
                if not is_clear:
                    print(f"  ⚠ Warning: Could not find clear position for {obj_type}")
                
                # Spawn object
                obj_id = self.spawn_object(obj_type, position=position)
                occupied_positions.append(position)
                
                # Get actual distance from camera
                distance_from_cam = np.linalg.norm(np.array(position) - self.cam_pos)
                
                print(f"  ✓ {obj_type} at {position} (distance: {distance_from_cam:.2f}m)")
        
        # Let physics settle
        print(f"  Settling physics ({settle_steps} steps)...")
        for _ in range(settle_steps):
            p.stepSimulation()
        
        print(f"✓ Scene ready with {len(self.objects)} objects")
        
        # Print statistics
        self._print_scene_statistics()
    
    def _print_scene_statistics(self):
        """Print statistics about spawned objects relative to camera"""
        if len(self.objects) == 0:
            return
        
        distances = []
        angles = []
        heights = []
        
        for obj_id in self.objects.keys():
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            pos = np.array(pos)
            
            # Calculate distance from camera
            distance = np.linalg.norm(pos - self.cam_pos)
            distances.append(distance)
            
            to_object = pos - self.cam_pos
            to_object_normalized = to_object / np.linalg.norm(to_object)
            angle_rad = np.arctan2(np.dot(to_object_normalized, self.right),
                                np.dot(to_object_normalized, self.forward))
            angles.append(np.rad2deg(angle_rad))
            
            # Height
            heights.append(pos[2])
        
        print(f"\n  Scene Statistics:")
        print(f"    Distance from camera: {min(distances):.2f}m - {max(distances):.2f}m (mean: {np.mean(distances):.2f}m)")
        print(f"    Angle from camera: {min(angles):.1f}° - {max(angles):.1f}° (mean: {np.mean(angles):.1f}°)")
        print(f"    Height above ground: {min(heights):.2f}m - {max(heights):.2f}m (mean: {np.mean(heights):.2f}m)")
    
    def get_object_ground_truth(self, obj_id):
       
        if obj_id not in self.objects:
            return None
        
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        
        return {
            'id': obj_id,
            'type': self.objects[obj_id]['type'],
            'class_id': self.objects[obj_id]['class_id'],
            'position': np.array(pos),
            'orientation': np.array(orn),
            'visited': self.objects[obj_id]['visited']
        }
    
    def get_all_objects_ground_truth(self):
        """Get ground truth for all objects"""
        return [self.get_object_ground_truth(obj_id) 
                for obj_id in self.objects.keys()]
    
    def mark_object_visited(self, obj_id):
        """Mark object as visited"""
        if obj_id in self.objects:
            self.objects[obj_id]['visited'] = True
    
    def step(self):
        """Step simulation forward"""
        p.stepSimulation()
    
    def reset(self):
       
        # Get all body IDs
        num_bodies = p.getNumBodies()
        
        for i in range(num_bodies):
            body_id = p.getBodyUniqueId(i)
            
            # Don't remove the plane (ID=0 usually)
            if body_id != self.plane_id:
                p.removeBody(body_id)
        
        self.objects.clear()
        print("✓ Environment reset")
    
    def close(self):
        """Disconnect from PyBullet"""
        p.disconnect()
        print("✓ Environment closed")


# Test code
if __name__ == "__main__":
    print("="*70)
    print("TESTING FIXED ENVIRONMENT - Camera-Relative Spawning")
    print("="*70)
    
    # Create environment
    env = RobotEnvironment(gui=True)
    
    # Spawn random scene
    env.spawn_random_scene(num_each_type=2)
    
    # Get ground truth
    gt_objects = env.get_all_objects_ground_truth()
    print("\nGround Truth Objects:")
    for obj in gt_objects:
        distance_from_cam = np.linalg.norm(obj['position'] - env.cam_pos)
        print(f"  {obj['type']}: position {obj['position']} (dist: {distance_from_cam:.2f}m)")
    
    # Run for a while
    print("\nSimulation running... (Close window to exit)")
    try:
        while True:
            env.step()
    except KeyboardInterrupt:
        pass
    
    env.close()