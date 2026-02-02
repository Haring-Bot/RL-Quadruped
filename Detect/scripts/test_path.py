import sys
sys.path.append('src')

from simulation.environment import RobotEnvironment
from simulation.camera import Camera
from vision_system import VisionSystem

# Setup environment with some objects
env = RobotEnvironment(gui=True)
env.spawn_random_scene(num_each_type=2)

# Setup vision system
camera = Camera()
model_path = "weights/best.pt" 
vision = VisionSystem(camera, model_path=model_path)

print("\n" + "="*70)
print("PATH PLANNING INTERFACE TEST")
print("="*70)

# Run detection (this is what your friend will do)
result = vision.detect_and_measure()

print(f"\nDetected {len(result['detections'])} objects:")
print("-"*70)

for i, detection in enumerate(result['detections'], 1):
    
    target_x = detection['position'][0] 
    target_y = detection['position'][1] 
    
    object_type = detection['class_name']
    confidence = detection['confidence']
    
    print(f"\n[Object {i}]")
    print(f"  Type: {object_type}")
    print(f"  Target X: {target_x:+.3f}m (left/right)")
    print(f"  Target Y: {target_y:+.3f}m (forward/back)")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  → Path planning target: [{target_x:.3f}, {target_y:.3f}]")

print("\n" + "="*70)
print("✓ Coordinates ready for path planning!")
print("="*70)

env.close()