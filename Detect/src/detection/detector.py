from ultralytics import YOLO
import numpy as np
import cv2
import yaml
from pathlib import Path
import time


class ObjectDetector:
    
    def __init__(self, model_path=None, config_path="config/config.yaml"):
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.detection_config = self.config['detection']
        
        # Load model
        if model_path is None:
            # Try to find trained model
            model_path = Path(self.config['paths']['models_dir']) / "robot_vision" / "weights" / "best.pt"
            
            if not model_path.exists():
                # Fall back to pretrained
                print("⚠ Trained model not found, using pretrained YOLOv8n")
                model_path = "yolov8n.pt"
        
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        
        # Detection parameters
        self.conf_threshold = self.detection_config['confidence_threshold']
        self.iou_threshold = self.detection_config['iou_threshold']
        self.max_detections = self.detection_config['max_detections']
        
        # Class names
        self.class_names = list(self.config['objects'].keys())
        
        # Performance tracking
        self.inference_times = []
        
        print(f"✓ Detector initialized")
        print(f"  Confidence threshold: {self.conf_threshold}")
        print(f"  Classes: {self.class_names}")
    
    def detect(self, image):

        start_time = time.time()
        
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False
        )
        
        # Parse results
        detections = []
        
        if len(results) > 0:
            result = results[0]  # First image
            
            # Get boxes, confidences, classes
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Bounding box in xyxy format
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Convert to xywh
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                
                # Center point
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Class and confidence
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Get class name
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                else:
                    class_name = f"unknown_{class_id}"
                
                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'center': [float(cx), float(cy)]
                }
                
                detections.append(detection)
        
        # Track inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Keep only last 100 measurements
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
        
        return detections
    
    def get_fps(self):
       
        if len(self.inference_times) == 0:
            return 0.0
        
        avg_time = np.mean(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_avg_inference_time(self):
       
        if len(self.inference_times) == 0:
            return 0.0
        
        return np.mean(self.inference_times) * 1000  # Convert to ms
    
    def visualize_detections(self, image, detections):
       
        vis_image = image.copy()
        
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
            
            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            
            # Background for text
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Draw center point
            cx, cy = det['center']
            cv2.circle(vis_image, (int(cx), int(cy)), 4, color, -1)
        
        # Draw FPS
        fps_text = f"FPS: {self.get_fps():.1f}"
        cv2.putText(
            vis_image,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        return vis_image


# Test code
if __name__ == "__main__":
    import pybullet as p
    import pybullet_data
    import sys
    sys.path.append('src')
    from simulation.camera import Camera
    
    print("Testing Object Detector")
    print("="*60)
    
    # Setup PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.8)
    
    # Spawn some objects
    cube = p.loadURDF("cube.urdf", [1.5, 0, 0.5])
    p.changeVisualShape(cube, -1, rgbaColor=[1, 0, 0, 1])
    
    sphere = p.loadURDF("sphere2.urdf", [2, 0.5, 0.3])
    p.changeVisualShape(sphere, -1, rgbaColor=[0, 0, 1, 1])
    
    # Let objects settle
    for _ in range(100):
        p.stepSimulation()
    
    # Create camera and detector
    camera = Camera()
    detector = ObjectDetector()
    
    print("\nRunning detection loop...")
    print("Press ESC to exit")
    
    try:
        while True:
            # Get image
            rgb_image = camera.get_image()
            
            # Detect objects
            detections = detector.detect(rgb_image)
            
            # Visualize
            vis_image = detector.visualize_detections(rgb_image, detections)
            
            # Display
            cv2.imshow("Object Detection", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
            # Print detections
            if detections:
                print(f"\nDetections ({len(detections)}):")
                for det in detections:
                    print(f"  {det['class_name']}: {det['confidence']:.2f} at {det['center']}")
            
            # Check for exit
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
    
    except KeyboardInterrupt:
        pass
    
    print("\n" + "="*60)
    print(f"Average FPS: {detector.get_fps():.1f}")
    print(f"Average inference time: {detector.get_avg_inference_time():.1f}ms")
    print("="*60)
    
    cv2.destroyAllWindows()
    p.disconnect()