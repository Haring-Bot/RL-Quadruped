import numpy as np
import time
import yaml
from collections import defaultdict


class TrackedObject:
    
    
    def __init__(self, detection, obj_id):
       
        self.id = obj_id
        self.class_id = detection['class_id']
        self.class_name = detection['class_name']
        
        # Position and distance
        self.position = np.array(detection['position'])
        self.distance = detection['distance']
        
        # Tracking info
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.times_seen = 1
        
        # State
        self.visited = False
        self.status = 'active'  # active, approaching, visited, lost
        
        # Confidence
        self.confidence = detection['confidence']
        self.position_confidence = detection['position_confidence']
    
    def update(self, detection):
        """Update with new detection"""
        # Weighted average for position (favor newer data)
        weight_new = 0.6
        weight_old = 0.4
        
        new_pos = np.array(detection['position'])
        self.position = weight_new * new_pos + weight_old * self.position
        
        self.distance = detection['distance']
        self.confidence = max(self.confidence, detection['confidence'])
        self.position_confidence = detection['position_confidence']
        
        self.last_seen = time.time()
        self.times_seen += 1
    
    def time_since_seen(self):
        
        return time.time() - self.last_seen
    
    def is_valid(self, min_sightings=3, lost_timeout=2.0):
        
        if self.status == 'lost':
            return False
        if self.times_seen < min_sightings:
            return False
        if self.time_since_seen() > lost_timeout:
            return False
        return True
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'position': self.position.tolist(),
            'distance': self.distance,
            'confidence': self.confidence,
            'position_confidence': self.position_confidence,
            'visited': self.visited,
            'status': self.status,
            'times_seen': self.times_seen,
            'time_since_seen': self.time_since_seen()
        }


class ObjectTracker:
    """Tracks multiple objects across frames"""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize object tracker"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tracking_config = self.config['tracking']
        
        # Parameters
        self.matching_threshold = self.tracking_config['matching_threshold']
        self.lost_timeout = self.tracking_config['lost_timeout']
        self.min_sightings = self.tracking_config['min_sightings']
        
        # Tracked objects
        self.objects = {}  # {obj_id: TrackedObject}
        self.next_id = 0
        
        print(f"✓ Object tracker initialized")
        print(f"  Matching threshold: {self.matching_threshold}m")
    
    def update(self, detections):
        
        # Mark all as not seen this frame
        for obj in self.objects.values():
            if obj.status != 'visited' and obj.time_since_seen() > self.lost_timeout:
                obj.status = 'lost'
        
        # Match detections to existing objects
        matched_ids = set()
        
        for detection in detections:
            det_pos = np.array(detection['position'])
            
            # Find closest existing object
            best_match_id = None
            best_distance = float('inf')
            
            for obj_id, obj in self.objects.items():
                if obj.status == 'lost' or obj.status == 'visited':
                    continue
                
                dist = np.linalg.norm(det_pos - obj.position)
                
                if dist < self.matching_threshold and dist < best_distance:
                    best_distance = dist
                    best_match_id = obj_id
            
          
            if best_match_id is not None:
                self.objects[best_match_id].update(detection)
                matched_ids.add(best_match_id)
            else:
               
                new_id = f"obj_{self.next_id}"
                self.next_id += 1
                self.objects[new_id] = TrackedObject(detection, new_id)
        
        # Get all valid objects
        valid_objects = [
            obj for obj in self.objects.values()
            if obj.is_valid(self.min_sightings, self.lost_timeout)
        ]
        
        return valid_objects
    
    def get_object(self, obj_id):
        """Get tracked object by ID"""
        return self.objects.get(obj_id)
    
    def mark_visited(self, obj_id):
        """Mark object as visited"""
        if obj_id in self.objects:
            self.objects[obj_id].visited = True
            self.objects[obj_id].status = 'visited'
    
    def mark_approaching(self, obj_id):
        """Mark object as being approached"""
        if obj_id in self.objects:
            self.objects[obj_id].status = 'approaching'
    
    def get_all_objects(self):
        """Get all tracked objects"""
        return list(self.objects.values())
    
    def reset(self):
        """Reset tracker"""
        self.objects.clear()
        self.next_id = 0


class TargetSelector:
    """Selects which object to navigate to"""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize target selector"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.selection_config = self.config['selection']
        
        # Parameters
        self.strategy = self.selection_config['strategy']
        self.reach_threshold = self.selection_config['reach_threshold']
        self.reach_time = self.selection_config['reach_time']
        
        # Current target
        self.current_target_id = None
        self.approach_start_time = None
        self.reached_hold_start = None
        
        # Priority weights (for priority-based strategy)
        self.class_priorities = {
            'cube': 10,
            'pyramid': 8,
            'cylinder': 6,
            'sphere': 4,
            'box': 2
        }
        
        print(f"✓ Target selector initialized")
        print(f"  Strategy: {self.strategy}")
    
    def select_target(self, tracked_objects, robot_position=None):
        
        # Filter valid candidates
        candidates = [
            obj for obj in tracked_objects
            if obj.status != 'visited' and obj.status != 'lost'
        ]
        
        if len(candidates) == 0:
            return None
        
        # Apply selection strategy
        if self.strategy == 'nearest':
            target = self._select_nearest(candidates)
        
        elif self.strategy == 'priority':
            target = self._select_priority(candidates)
        
        elif self.strategy == 'exploration':
            target = self._select_exploration(candidates, robot_position)
        
        else:
            # Default to nearest
            target = self._select_nearest(candidates)
        
        return target
    
    def _select_nearest(self, candidates):
        """Select nearest object"""
        return min(candidates, key=lambda obj: obj.distance)
    
    def _select_priority(self, candidates):
        """Select based on priority and distance"""
        def score_fn(obj):
            priority = self.class_priorities.get(obj.class_name, 1)
            # Higher score = better target
            # score = (priority × confidence) / distance
            score = (priority * obj.confidence) / max(obj.distance, 0.5)
            return score
        
        return max(candidates, key=score_fn)
    
    def _select_exploration(self, candidates, robot_position):
        """Select object in unexplored direction"""
        if robot_position is None:
            return self._select_nearest(candidates)
        
        robot_pos = np.array(robot_position[:2])  # x, y only
        
        # Calculate angles to all candidates
        angles = []
        for obj in candidates:
            obj_pos = np.array(obj.position[:2])
            vec = obj_pos - robot_pos
            angle = np.arctan2(vec[1], vec[0])
            angles.append(angle)
        
        # Prefer objects with different angles
        # (simple heuristic: maximum angle difference from current target)
        if self.current_target_id is not None:
            current_target = next(
                (obj for obj in candidates if obj.id == self.current_target_id),
                None
            )
            if current_target is not None:
                current_angle = np.arctan2(
                    current_target.position[1] - robot_pos[1],
                    current_target.position[0] - robot_pos[0]
                )
                
                # Find object with max angle difference
                angle_diffs = [abs(angle - current_angle) for angle in angles]
                max_diff_idx = np.argmax(angle_diffs)
                return candidates[max_diff_idx]
        
        # Fall back to nearest
        return self._select_nearest(candidates)
    
    def check_target_reached(self, target, robot_position):
        
        if target is None or robot_position is None:
            return False
        
        # Calculate distance to target
        robot_pos = np.array(robot_position)
        target_pos = np.array(target.position)
        distance = np.linalg.norm(robot_pos - target_pos)
        
        # Check if within threshold
        if distance < self.reach_threshold:
            # Start or continue hold timer
            if self.reached_hold_start is None:
                self.reached_hold_start = time.time()
            
            # Check if held long enough
            hold_duration = time.time() - self.reached_hold_start
            if hold_duration >= self.reach_time:
                return True
        else:
            # Reset hold timer
            self.reached_hold_start = None
        
        return False
    
    def set_current_target(self, target):
        """Set current target"""
        if target is not None:
            self.current_target_id = target.id
            self.approach_start_time = time.time()
        else:
            self.current_target_id = None
            self.approach_start_time = None
        
        self.reached_hold_start = None
    
    def get_mission_status(self, tracked_objects):
        
        all_objects = [
            obj for obj in tracked_objects
            if obj.times_seen >= self.config['tracking']['min_sightings']
        ]
        
        total = len(all_objects)
        visited = sum(1 for obj in all_objects if obj.visited)
        remaining = total - visited
        complete = remaining == 0 and total > 0
        
        return {
            'total_objects': total,
            'visited': visited,
            'remaining': remaining,
            'complete': complete
        }


# Test code
if __name__ == "__main__":
    print("Testing Object Tracker & Target Selector")
    print("="*60)
    
    # Create tracker and selector
    tracker = ObjectTracker()
    selector = TargetSelector()
    
    # Simulate detections over multiple frames
    print("\nSimulating multi-frame tracking...")
    
    # Frame 1: Detect 2 objects
    frame1_detections = [
        {
            'class_id': 0, 'class_name': 'cube', 'confidence': 0.9,
            'position': [1.0, 0.5, 0.1], 'distance': 1.12,
            'position_confidence': 0.85
        },
        {
            'class_id': 1, 'class_name': 'sphere', 'confidence': 0.85,
            'position': [2.0, -0.3, 0.1], 'distance': 2.02,
            'position_confidence': 0.80
        }
    ]
    
    tracked = tracker.update(frame1_detections)
    print(f"\nFrame 1: {len(tracked)} tracked objects")
    
    # Frame 2: Same objects, slightly moved
    frame2_detections = [
        {
            'class_id': 0, 'class_name': 'cube', 'confidence': 0.92,
            'position': [1.02, 0.48, 0.1], 'distance': 1.11,
            'position_confidence': 0.87
        },
        {
            'class_id': 1, 'class_name': 'sphere', 'confidence': 0.88,
            'position': [2.01, -0.32, 0.1], 'distance': 2.04,
            'position_confidence': 0.82
        }
    ]
    
    tracked = tracker.update(frame2_detections)
    print(f"Frame 2: {len(tracked)} tracked objects")
    
    # Frame 3: Add new object
    frame3_detections = frame2_detections + [
        {
            'class_id': 2, 'class_name': 'cylinder', 'confidence': 0.75,
            'position': [1.5, 1.0, 0.15], 'distance': 1.8,
            'position_confidence': 0.70
        }
    ]
    
    tracked = tracker.update(frame3_detections)
    print(f"Frame 3: {len(tracked)} tracked objects")
    
    # Display tracked objects
    print("\nTracked Objects:")
    for obj in tracked:
        print(f"  {obj.id}: {obj.class_name}")
        print(f"    Position: {obj.position}")
        print(f"    Distance: {obj.distance:.2f}m")
        print(f"    Times seen: {obj.times_seen}")
        print(f"    Status: {obj.status}")
    
    # Select target
    print("\nTarget Selection:")
    target = selector.select_target(tracked)
    if target:
        print(f"  Selected: {target.id} ({target.class_name})")
        print(f"  Distance: {target.distance:.2f}m")
        selector.set_current_target(target)
        tracker.mark_approaching(target.id)
    
    # Check mission status
    status = selector.get_mission_status(tracked)
    print(f"\nMission Status:")
    print(f"  Total objects: {status['total_objects']}")
    print(f"  Visited: {status['visited']}")
    print(f"  Remaining: {status['remaining']}")
    print(f"  Complete: {status['complete']}")
    
    print("\n" + "="*60)