import numpy as np
import pybullet as p
import heapq

class PathManager:

    def __init__(self, robot, grid_resolution=0.1, arena_size=10): 
        #Will find a target in a square of 10 by 10 around the robot (arena_size). It divides the area into cells to do an A* algorithm
        #Each cell will represent a waypoint, then the path will be smoothed to avoid zigzag
        self.robot = robot
        self.res = grid_resolution
        self.size = arena_size
        self.current_path = []
        self.debug_ids = []

    def _world_to_grid(self, pos):
        col = int((pos[0] + self.size) / self.res)
        row = int((pos[1] + self.size) / self.res)
        return (row, col)

    def _grid_to_world(self, grid_pos):
        x = grid_pos[1] * self.res - self.size
        y = grid_pos[0] * self.res - self.size
        return np.array([x, y])

    def heuristic(self, a, b):
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    def get_neighbors(self, node):
        neighbors = []
        # 8 directions around the actual cell
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            neighbor = (node[0] + dx, node[1] + dy)
            
            if 0 <= neighbor[0] < (self.size*2/self.res) and 0 <= neighbor[1] < (self.size*2/self.res):               
                # Verify collision
                world_pos = self._grid_to_world(neighbor)
                if self.is_position_free(world_pos):
                    neighbors.append(neighbor)
        return neighbors
    
    def is_position_free(self, world_pos, margin = 0.5): #margin = safe zone around a position

        #Define a box arround the point
        min_box = [world_pos[0] - margin, world_pos[1] - margin, 0.05]
        max_box = [world_pos[0] + margin, world_pos[1] + margin, 0.5]
        
        #This thing from Pybullet works very well for checking obstacles
        hits = p.getOverlappingObjects(min_box, max_box)
        
        if hits:
            for hit in hits:
                if hit[0] != self.robot.id and hit[0] != 0:
                    return False
        return True

    def smooth_path(self, path):
        #Delete useless points for a smooth path
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            #Look for the farest point where a straight line is possible
            next_idx = len(path) - 1
            while next_idx > current_idx + 1:
                if self.is_line_clear(path[current_idx], path[next_idx]):
                    break
                next_idx -= 1
            
            smoothed.append(path[next_idx])
            current_idx = next_idx
            
        return smoothed

    def is_line_clear(self, start_p, end_p, margin = 0.5):
        #Verify if the created line is not too close of an object.
        #It creates 2 parallel lines separated by margin value on each side
        dx = end_p[0] - start_p[0]
        dy = end_p[1] - start_p[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist == 0: return True

        perp_x = -dy / dist * margin
        perp_y = dx / dist * margin

        offsets = [(0, 0), (perp_x, perp_y), (-perp_x, -perp_y)]
        
        for ox, oy in offsets:
            start_ray = [start_p[0] + ox, start_p[1] + oy, 0.2]
            end_ray = [end_p[0] + ox, end_p[1] + oy, 0.2]
            
            ray_result = p.rayTest(start_ray, end_ray)
            
            hit_id = ray_result[0][0]
            if hit_id != -1 and hit_id != self.robot.id and hit_id != 0:
                return False # Obstacle detected
                
        return True #no obstacle
    

    def plan_path(self, target_world_pos):
        for debug_id in self.debug_ids:
            p.removeUserDebugItem(debug_id)
        self.debug_ids = []

        # A* 
        start_world = self.robot.get_position()[:2]
        start = self._world_to_grid(start_world)
        goal = self._world_to_grid(target_world_pos[:2])

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current = heapq.heappop(frontier)[1]
            if current == goal: break
            for next_node in self.get_neighbors(current):
                new_cost = cost_so_far[current] + self.heuristic(current, next_node)
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(goal, next_node)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current

        #Building the path
        raw_path = []
        curr = goal
        while curr is not None:
            raw_path.append(self._grid_to_world(curr))
            curr = came_from.get(curr)
        raw_path.reverse()
        
        self.current_path = self.smooth_path(raw_path)
        
        #show the path in simulation
        self.visualize_path(target_world_pos)
        return self.current_path

    def visualize_path(self, target_pos):
        for i in range(len(self.current_path) - 1):
            p1 = self.current_path[i]
            p2 = self.current_path[i+1]
            line_id = p.addUserDebugLine([p1[0], p1[1], 0.05], 
                                         [p2[0], p2[1], 0.05], 
                                         [1, 1, 0], lineWidth=3)
            self.debug_ids.append(line_id)

        #Target (red line)
        target_id = p.addUserDebugLine([target_pos[0], target_pos[1], 0], 
                                      [target_pos[0], target_pos[1], 0.6], 
                                      [1, 0, 0], lineWidth=6)
        self.debug_ids.append(target_id)

    def get_next_waypoint(self, threshold=0.2):
        if not self.current_path: return None
        current_pos = np.array(self.robot.get_position()[:2])
        dist = np.linalg.norm(current_pos - self.current_path[0])
        if dist < threshold:
            self.current_path.pop(0)
        return self.current_path[0] if self.current_path else None