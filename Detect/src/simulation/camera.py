import pybullet as p
import numpy as np
import cv2
import yaml


class Camera:
    
    
    def __init__(self, config_path="config/config.yaml"):
       
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['camera']
        
        # Image dimensions
        self.width = self.config['width']
        self.height = self.config['height']
        
        # Camera parameters
        self.fov = self.config['fov']
        self.near = self.config['near']
        self.far = self.config['far']
        self.aspect = self.width / self.height
        
        # Camera pose
        self.cam_position = self.config['position']
        self.cam_target = self.config['target']
        self.cam_up = self.config['up']
        
        # Compute projection and view matrices
        self._update_matrices()
        
        # Compute focal length (useful for geometric calculations)
        self.focal_length = self._compute_focal_length()
        
        print(f"✓ Camera initialized: {self.width}x{self.height}, FOV={self.fov}°")
        print(f"  Focal length: {self.focal_length:.1f} pixels")
    
    def _update_matrices(self):
        """Compute view and projection matrices"""
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.cam_position,
            cameraTargetPosition=self.cam_target,
            cameraUpVector=self.cam_up
        )
        
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far
        )
    
    def _compute_focal_length(self):
        
        fov_rad = np.deg2rad(self.fov)
        focal_length = (self.width / 2) / np.tan(fov_rad / 2)
        return focal_length
    
    def set_pose(self, position, target=None, up=None):
        
        self.cam_position = position
        if target is not None:
            self.cam_target = target
        if up is not None:
            self.cam_up = up
        
        self._update_matrices()
    
    def get_image(self):
        
        # Get camera image
        _, _, rgb, depth, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to numpy array
        rgb_array = np.array(rgb).reshape(self.height, self.width, 4)
        
        # Remove alpha channel and convert to uint8
        rgb_array = rgb_array[:, :, :3].astype(np.uint8)
        
        return rgb_array
    
    def get_depth(self):
        
        # Get camera image
        _, _, _, depth, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to numpy array
        depth_buffer = np.array(depth).reshape(self.height, self.width)
        
        depth_meters = self.far * self.near / (
            self.far - (self.far - self.near) * depth_buffer
        )
        
        return depth_meters.astype(np.float32)
    
    def get_rgbd(self):
        _, _, rgb, depth, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # RGB
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(
            self.height, self.width, 4
        )[:, :, :3]

        # Depth (meters)
        depth_buffer = np.array(depth).reshape(self.height, self.width)
        depth_meters = self.far * self.near / (
            self.far - (self.far - self.near) * depth_buffer
        )

        # Segmentation
        seg_array = np.array(seg).reshape(self.height, self.width)

        return (
            rgb_array,
            depth_meters.astype(np.float32),
            seg_array
        )
    
    def get_camera_intrinsics(self):
        
        return {
            'fx': self.focal_length,             
            'fy': self.focal_length,              
            'cx': self.width / 2,                
            'cy': self.height / 2,               
            'image_width': self.width,
            'image_height': self.height,
            'fov': self.fov,
            'near': self.near,
            'far': self.far,
            'K': np.array([                     
                [self.focal_length, 0, self.width/2],
                [0, self.focal_length, self.height/2],
                [0, 0, 1]
            ])
        }
    
    def visualize_depth(self, depth_image):
        
        depth_clipped = np.clip(depth_image, 0, 10.0)

        depth_normalized = (depth_clipped / 10.0 * 255).astype(np.uint8)

        depth_vis = 255 - depth_normalized
        
        return depth_vis


# Test code
if __name__ == "__main__":
    import pybullet_data
    
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    cube = p.loadURDF("cube.urdf", [1, 0, 0.5])
    p.changeVisualShape(cube, -1, rgbaColor=[1, 0, 0, 1])
    
    sphere = p.loadURDF("sphere2.urdf", [2, 0.5, 0.5])
    p.changeVisualShape(sphere, -1, rgbaColor=[0, 0, 1, 1])
    
 
    camera = Camera()

    print("\nCapturing images...")
    rgb, depth = camera.get_rgbd()
    
    print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
    print(f"Depth range: {depth.min():.2f}m to {depth.max():.2f}m")
    
    cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow("Depth", camera.visualize_depth(depth))
    
    print("\nCamera intrinsics:")
    intrinsics = camera.get_camera_intrinsics()
    for key, value in intrinsics.items():
        print(f"  {key}: {value}")
    
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    p.disconnect()