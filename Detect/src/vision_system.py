import numpy as np
import cv2
import yaml
import time
from pathlib import Path

# Import our modules
import sys
sys.path.append('src')

from detection.detector import ObjectDetector
from perception.depth_estimator import DepthEstimator, PositionCalculator


class VisionSystem:
    
    
    def __init__(self, camera, model_path=None, config_path="config/config.yaml"):
       
        print("\n" + "="*60)
        print("INITIALIZING VISION SYSTEM")
        print("="*60)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Store camera
        self.camera = camera
        self.camera_intrinsics = camera.get_camera_intrinsics()
        
        # Initialize components
        print("\n1. Loading object detector...")
        self.detector = ObjectDetector(model_path, config_path)
        
        print("\n2. Initializing depth estimator...")
        self.depth_estimator = DepthEstimator(config_path)
        
        print("\n3. Initializing position calculator...")
        self.position_calculator = PositionCalculator(self.camera_intrinsics, config_path)
        
        # Performance tracking
        self.frame_times = []
        
        print("\n" + "="*60)
        print("✓ VISION SYSTEM READY")
        print("="*60)
    
    def detect_and_measure(self):
        
        start_time = time.time()
        
        # 1. Capture RGB, depth, segmentation
        rgb_image, depth_image, _ = self.camera.get_rgbd()

        # 2. Decode segmentation
        detections = self.detector.detect(rgb_image)

        enhanced_detections = []
        filtered_detections = []
        image_height = self.camera_intrinsics['image_height']
        image_width = self.camera_intrinsics['image_width']

        for det in detections:
            # Estimate distance
            distance_result = self.depth_estimator.estimate_distance(det, depth_image)
            
            # Skip if distance estimation failed
            if not distance_result['valid'] or distance_result['distance'] is None:
                continue
            
            distance = distance_result['distance']
            
            # Validate distance is in reasonable range
            max_distance = self.config['distance'].get('max_range', 5.0)

            if distance > max_distance:
                continue

            position_result = self.position_calculator.calculate_3d_position(det, distance)
            

            # Combine all information
            enhanced_det = {
                **det,  # Original detection info
                'distance': distance,
                'distance_std': distance_result['std_dev'],
                'position': position_result['position'],
                'position_camera': position_result['position_camera'],
                'position_confidence': position_result['confidence']
            }
            
            enhanced_detections.append(enhanced_det)
        
        # 4. Suppress duplicate detections
        enhanced_detections = self._suppress_duplicates(enhanced_detections)
    
        processing_time = time.time() - start_time
        self.frame_times.append(processing_time)
        if len(self.frame_times) > 100:
            self.frame_times = self.frame_times[-100:]
        
        return {
            'detections': enhanced_detections,
            'rgb_image': rgb_image,
            'depth_image': depth_image,
            'num_detections': len(enhanced_detections),
            'processing_time': processing_time
        }


    def get_fps(self):
        """Get average FPS of complete vision pipeline"""
        if len(self.frame_times) == 0:
            return 0.0
        
        avg_time = np.mean(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_detector_fps(self):
        """Get detector-only FPS"""
        return self.detector.get_fps()
    
    def visualize(self, result):
        
        rgb_image = result['rgb_image']
        detections = result['detections']
        
        vis_image = rgb_image.copy()
        
        # Define colors for each class
        colors = [
            (255, 0, 0),    # Red - cube
            (0, 0, 255),    # Blue - sphere
            (0, 255, 0),    # Green - cylinder
            (255, 255, 0),  # Yellow - pyramid
            (255, 165, 0),  # Orange - box
        ]
        
        for det in detections:
            # Get bbox
            x, y, w, h = det['bbox']
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # Get color
            class_id = det['class_id']
            color = colors[class_id % len(colors)]
            
            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            distance_label = f"Dist: {det['distance']:.2f}m"
            
            # Background for text
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_h - 25),
                (x1 + max(label_w, 100), y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                vis_image, label,
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1
            )
            cv2.putText(
                vis_image, distance_label,
                (x1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1
            )
            
            # Draw center point
            cx, cy = det['center']
            cv2.circle(vis_image, (int(cx), int(cy)), 4, color, -1)
        
        # Draw system info
        info_y = 30
        cv2.putText(
            vis_image,
            f"Detections: {len(detections)}",
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2
        )
        
        cv2.putText(
            vis_image,
            f"FPS: {self.get_fps():.1f}",
            (10, info_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2
        )
        
        return vis_image
    
    def print_detections(self, detections):
        
        if len(detections) == 0:
            print("No objects detected")
            return
        
        print(f"\n{'='*70}")
        print(f"DETECTED {len(detections)} OBJECTS")
        print(f"{'='*70}")
        
        for i, det in enumerate(detections, 1):
            print(f"\n[{i}] {det['class_name'].upper()}")
            print(f"    Confidence: {det['confidence']:.2%}")
            print(f"    Distance:   {det['distance']:.3f}m (±{det['distance_std']:.4f}m)")
            print(f"    Position:   X={det['position'][0]:+.3f}m, Y={det['position'][1]:+.3f}m, Z={det['position'][2]:+.3f}m")
            print(f"    BBox:       x={det['bbox'][0]:.0f}, y={det['bbox'][1]:.0f}, w={det['bbox'][2]:.0f}, h={det['bbox'][3]:.0f}")
        
        print(f"{'='*70}\n")
    
    
    def _compute_iou(self, bbox1, bbox2):
        """
        Compute Intersection over Union between two bounding boxes
        
        Args:
            bbox1, bbox2: [x, y, w, h] format
        
        Returns:
            IoU score (0-1)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to corner format
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1_max, x2_max)
        yi2 = min(y1_max, y2_max)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def _suppress_duplicates(self, detections, iou_threshold=0.5):
       
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for det in detections:
            # Check if this detection overlaps with any kept detection
            is_duplicate = False
            
            for kept_det in keep:
               
                if det['class_name'] == kept_det['class_name']:
                    iou = self._compute_iou(det['bbox'], kept_det['bbox'])
                    
                    if iou > iou_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                keep.append(det)
        
        return keep

# Example usage
if __name__ == "__main__":
   
    import pybullet as p
    import pybullet_data
    from simulation.environment import RobotEnvironment
    from simulation.camera import Camera
    
    print("VISION SYSTEM EXAMPLE")
    print("="*60)
    
    # Setup environment
    env = RobotEnvironment(gui=True)
    env.spawn_random_scene(num_each_type=1)
    
    # Create camera
    camera = Camera()
    
    # Create vision system
    vision = VisionSystem(camera)
    
    print("\nRunning detection...")
    print("Press ESC to exit\n")
    
    try:
        while True:
            # Run detection
            result = vision.detect_and_measure()
            
            # Print results
            if result['num_detections'] > 0:
                vision.print_detections(result['detections'])
            
            # Visualize
            vis_image = vision.visualize(result)
            cv2.imshow("Vision System", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
            # Check for exit
            if cv2.waitKey(1) == 27:  # ESC
                break
            
            # Step simulation
            env.step()
    
    except KeyboardInterrupt:
        pass
    
    print("\nShutting down...")
    cv2.destroyAllWindows()
    env.close()