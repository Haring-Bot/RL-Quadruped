import pybullet as p
import numpy as np
import cv2
import yaml
import time
import sys
from pathlib import Path
from collections import defaultdict

sys.path.append('src')

from simulation.environment import RobotEnvironment
from simulation.camera import Camera
from vision_system import VisionSystem

class DetectionTester:
    """Test vision system"""
    
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Test parameters from config
        self.test_distances = self.config['data_generation']['distances']
        self.test_angles = self.config['data_generation']['angles']
        self.object_types = list(self.config['objects'].keys())
        
        # Camera pose for coordinate transformation
        self.cam_position = None
        self.cam_target = None
        self.cam_up = None
        
        # Results storage
        self.results = {
            'total_objects': 0,
            'detected_objects': 0,
            'false_positives': 0,
            'by_class': defaultdict(lambda: {'tp': 0, 'fn': 0, 'fp': 0}),
            'distance_errors': [],
            'position_errors': []
        }
    
    def camera_to_world(self, p_cam):
       
        cam_pos = np.array(self.cam_position)
        cam_target = np.array(self.cam_target)
        cam_up = np.array(self.cam_up)
        
        # Calculate camera basis vectors
        forward = cam_target - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, cam_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Extract camera coordinates
        x_c, y_c, z_c = p_cam  # x=right, y=vertical offset, z=depth
        
        # Transform to world
        world_pos = cam_pos + z_c * forward + x_c * right + y_c * up
        
        return world_pos
    
    def spawn_test_object(self, env, object_type, distance=None, angle=None, max_retries=5):
       
        for retry in range(max_retries):
            if distance is None:
                distance = np.random.choice(self.test_distances)
            if angle is None:
                angle = np.random.choice(self.test_angles)
            
            cam_pos = np.array(self.cam_position)
            cam_target = np.array(self.cam_target)
            cam_up = np.array(self.cam_up)
            
            # Calculate camera orientation
            forward = cam_target - cam_pos
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(forward, cam_up)
            right = right / np.linalg.norm(right)
            
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            # Convert angle to radians
            angle_rad = np.deg2rad(angle)
            
            # Randomize vertical offset slightly
            vertical_offset = np.random.uniform(-0.05, 0.15)
            
            position = (cam_pos + 
                    distance * np.cos(angle_rad) * forward +
                    distance * np.sin(angle_rad) * right +
                    vertical_offset * up)
          
            position[2] = np.clip(position[2], 0.3, 1.5)
            
            actual_distance = np.linalg.norm(position - cam_pos)
            
           
            if 0.7 <= actual_distance <= 2.0:
                break
            
            if retry == max_retries - 1:
                print(f"    ⚠ Could not find ideal position, using distance={actual_distance:.2f}m")
        
       
        obj_id = env.spawn_object(object_type, position=position.tolist())
        
        for _ in range(50):
            env.step()

        ground_truth = env.get_object_ground_truth(obj_id)
        ground_truth['expected_distance'] = distance
        ground_truth['expected_angle'] = angle

        actual_pos = ground_truth['position']
        actual_dist = np.linalg.norm(actual_pos - cam_pos)
        print(f"  Spawned {object_type}: expected {distance:.2f}m @ {angle}°, actual {actual_dist:.2f}m at {actual_pos}")
        
        return ground_truth
    
    def match_detection_to_ground_truth(self, detection, ground_truth_list, match_threshold=1.0):
        
        det_pos = np.array(detection['position'])
        det_class = detection['class_name']
        
        best_match = None
        best_distance = float('inf')
        
        for gt in ground_truth_list:
            gt_pos = gt['position']
            gt_class = gt['type']
            distance = np.linalg.norm(det_pos - gt_pos)
            
            
            if det_class == gt_class and distance < match_threshold:
                if distance < best_distance:
                    best_match = gt
                    best_distance = distance
        
        return best_match
    
    def evaluate_detections(self, detections, ground_truth_list):
     
        matched_gt = set()
        
        for det in detections:
            gt_match = self.match_detection_to_ground_truth(det, ground_truth_list)
            
            if gt_match:
                # True positive
                matched_gt.add(gt_match['id'])
                self.results['detected_objects'] += 1
                self.results['by_class'][det['class_name']]['tp'] += 1
                
                # Calculate errors
                det_pos = np.array(det['position'])
                gt_pos = gt_match['position']
                position_error = np.linalg.norm(det_pos - gt_pos)
                self.results['position_errors'].append(position_error)
                
                # Distance error (from camera)
                det_distance = det['distance']
                gt_distance = np.linalg.norm(gt_pos - self.cam_position)
                distance_error = abs(det_distance - gt_distance)
                self.results['distance_errors'].append(distance_error)
            else:
                
                self.results['false_positives'] += 1
                self.results['by_class'][det['class_name']]['fp'] += 1
        
        
        for gt in ground_truth_list:
            if gt['id'] not in matched_gt:
                self.results['by_class'][gt['type']]['fn'] += 1
    
    def run_test(self, num_scenes=10, objects_per_scene=5, show_visualization=True):
        
        print("\n" + "="*70)
        print("DETECTION SYSTEM TEST")
        print("="*70)
        print(f"\nTest Configuration:")
        print(f"  Scenes: {num_scenes}")
        print(f"  Objects per scene: {objects_per_scene}")
        print(f"  Total objects: {num_scenes * objects_per_scene}")
        print(f"  Test distances: {self.test_distances}")
        print(f"  Test angles: {self.test_angles}°")
        
        # Setup
        print("\nInitializing environment...")
        env = RobotEnvironment(gui=show_visualization)
        camera = Camera()
        
        # Store camera pose
        self.cam_position = np.array(camera.cam_position)
        self.cam_target = np.array(camera.cam_target)
        self.cam_up = np.array(camera.cam_up)
        
        print(f"Camera position: {self.cam_position}")
        print(f"Camera target: {self.cam_target}")
        
        print("\nLoading vision system...")
        
        model_path = Path("weights/best.pt")
        if not model_path.exists():
            print("⚠ Using pretrained YOLOv8n (no custom training)")
            model_path = None
        else:
            print(f"✓ Using trained model: {model_path}")
        
        vision = VisionSystem(camera, model_path=model_path)
        
        print("\n" + "-"*70)
        print("RUNNING TESTS")
        print("-"*70)
        
        # Run tests
        for scene_num in range(num_scenes):
            print(f"\n[Scene {scene_num + 1}/{num_scenes}]")
            
            env.reset()

            ground_truth_objects = []
            for i in range(objects_per_scene):
                obj_type = np.random.choice(self.object_types)
                gt = self.spawn_test_object(env, obj_type)
                ground_truth_objects.append(gt)
                print(f"  Spawned {obj_type} at {gt['position']}")
            
            self.results['total_objects'] += len(ground_truth_objects)

            result = vision.detect_and_measure()
            detections = result['detections']
            
            print(f"  Detected {len(detections)} objects")
            for det in detections:
                print(f"    {det['class_name']} at {det['position']} (conf: {det['confidence']:.2f})")

            self.evaluate_detections(detections, ground_truth_objects)

            if show_visualization:
                vis_image = vision.visualize(result)
                cv2.imshow("Detection Test", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1000)  # Show for 1 second
                if key == 27:  # ESC to exit early
                    print("\nTest interrupted by user")
                    break
        
        
        if show_visualization:
            cv2.destroyAllWindows()
        env.close()
        
        # Print results
        self.print_results()
    
    def print_results(self):
        """Print test results"""
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        
        # Overall stats
        total = self.results['total_objects']
        detected = self.results['detected_objects']
        fps = self.results['false_positives']
        missed = total - detected
        
        print(f"\nOverall Performance:")
        print(f"  Total objects: {total}")
        print(f"  Detected: {detected}")
        print(f"  Missed: {missed}")
        print(f"  False positives: {fps}")
        
        # Precision and Recall
        precision = detected / (detected + fps) if (detected + fps) > 0 else 0
        recall = detected / total if total > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nMetrics:")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1 Score: {f1:.2%}")
        
        # Per-class performance
        print(f"\nPer-Class Performance:")
        print(f"  {'Class':<12} {'TP':>6} {'FN':>6} {'FP':>6} {'Precision':>10} {'Recall':>10}")
        print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*10}")
        
        for class_name, stats in self.results['by_class'].items():
            tp = stats['tp']
            fn = stats['fn']
            fp = stats['fp']
            
            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"  {class_name:<12} {tp:>6} {fn:>6} {fp:>6} {class_precision:>9.1%} {class_recall:>9.1%}")
        
        # Distance accuracy
        if self.results['distance_errors']:
            dist_errors = np.array(self.results['distance_errors'])
            print(f"\nDistance Estimation:")
            print(f"  Mean error: {np.mean(dist_errors):.3f}m")
            print(f"  Std dev: {np.std(dist_errors):.3f}m")
            print(f"  Max error: {np.max(dist_errors):.3f}m")
            print(f"  Min error: {np.min(dist_errors):.3f}m")
        
        # Position accuracy
        if self.results['position_errors']:
            pos_errors = np.array(self.results['position_errors'])
            print(f"\nPosition Estimation:")
            print(f"  Mean error: {np.mean(pos_errors):.3f}m")
            print(f"  Std dev: {np.std(pos_errors):.3f}m")
            print(f"  Max error: {np.max(pos_errors):.3f}m")
        
        # Performance targets
        print(f"\n" + "-"*70)
        print("Performance vs Targets:")
        
        target_f1 = self.config['performance']['target_f1_score']
        target_dist_error = self.config['performance']['target_distance_error']
        target_pos_error = self.config['performance']['target_position_error']
        
        print(f"  F1 Score: {f1:.2%} (target: {target_f1:.2%}) {'✓' if f1 >= target_f1 else '✗'}")
        
        if self.results['distance_errors']:
            mean_dist_error = np.mean(dist_errors)
            print(f"  Distance Error: {mean_dist_error:.3f}m (target: <{target_dist_error}m) {'✓' if mean_dist_error <= target_dist_error else '✗'}")
        
        if self.results['position_errors']:
            mean_pos_error = np.mean(pos_errors)
            print(f"  Position Error: {mean_pos_error:.3f}m (target: <{target_pos_error}m) {'✓' if mean_pos_error <= target_pos_error else '✗'}")
        
        print("="*70)


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test object detection system')
    parser.add_argument('--scenes', type=int, default=5,
                       help='Number of test scenes')
    parser.add_argument('--objects', type=int, default=3,
                       help='Objects per scene')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Run test
    tester = DetectionTester()
    tester.run_test(
        num_scenes=args.scenes,
        objects_per_scene=args.objects,
        show_visualization=not args.no_viz
    )


if __name__ == "__main__":
    main()