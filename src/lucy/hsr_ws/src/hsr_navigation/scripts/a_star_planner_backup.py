#!/usr/bin/env python3
"""
a_star_planner.py - Enhanced A* with smoothing and safety margins
"""

import numpy as np
import heapq
import cv2
import math

class AStarPlanner:
    def __init__(self, grid_map, resolution=0.05, origin=(0.0, 0.0)):
        self.grid_map = grid_map
        self.rows, self.cols = grid_map.shape
        self.resolution = resolution
        self.origin = origin
        
        # Compute distance map (higher = safer)
        self.distance_map = self.compute_clearance_map(grid_map)
        
        # NEW: Create safety buffer zone
        self.safety_buffer = self.create_safety_buffer(grid_map)

    def compute_clearance_map(self, grid_map):
        """Distance transform: higher value = farther from obstacles"""
        binary_free = np.where(grid_map == 0, 1, 0).astype(np.uint8)
        dist_transform = cv2.distanceTransform(binary_free, cv2.DIST_L2, 5)
        max_dist = np.max(dist_transform)
        return dist_transform / (max_dist + 1e-5)
    
    def create_safety_buffer(self, grid_map, buffer_cells=3):
        """Create a zone around obstacles for penalty weighting"""
        obstacles = (grid_map > 0).astype(np.uint8)
        kernel = np.ones((buffer_cells*2+1, buffer_cells*2+1), dtype=np.uint8)
        buffered = cv2.dilate(obstacles, kernel, iterations=1)
        return buffered

    def get_nearest_free_cell(self, coord, max_radius_cells=20):
        """Find nearest free cell if start/goal is in obstacle"""
        if self.grid_map[coord[0], coord[1]] == 0:
            return coord
        
        for radius in range(1, max_radius_cells):
            for angle in np.linspace(0, 2*np.pi, 8*radius):
                nr = int(coord[0] + radius * np.cos(angle))
                nc = int(coord[1] + radius * np.sin(angle))
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.grid_map[nr, nc] == 0:
                        return (nr, nc)
        return None

    def heuristic(self, a, b):
        """Euclidean distance heuristic"""
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def get_neighbors(self, node):
        """8-connected grid with diagonal cost adjustment"""
        directions = [
            (1,0), (-1,0), (0,1), (0,-1),      # cardinal
            (1,1), (-1,-1), (1,-1), (-1,1)     # diagonal
        ]
        neighbors = []
        for dr, dc in directions:
            nr, nc = node[0] + dr, node[1] + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid_map[nr, nc] == 0:
                    move_cost = np.hypot(dr, dc)
                    neighbors.append(((nr, nc), move_cost))
        return neighbors

    def grid_to_world(self, grid_coord):
        x = grid_coord[1] * self.resolution + self.origin[0]
        y = grid_coord[0] * self.resolution + self.origin[1]
        return (x, y)

    def world_to_grid(self, world_coord):
        col = int((world_coord[0] - self.origin[0]) / self.resolution)
        row = int((world_coord[1] - self.origin[1]) / self.resolution)
        return (row, col)

    def plan(self, start_world, goal_world):
        """A* with enhanced safety and smoothing"""
        start = self.world_to_grid(start_world)
        goal = self.world_to_grid(goal_world)

        # Bounds check
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start {start} out of bounds")
        if not (0 <= goal[0] < self.rows and 0 <= goal[1] < self.cols):
            raise ValueError(f"Goal {goal} out of bounds")
        
        # Snap to free space if needed
        start = self.get_nearest_free_cell(start) or start
        goal = self.get_nearest_free_cell(goal) or goal
        
        if self.grid_map[start] != 0:
            raise ValueError(f"Start {start} in obstacle")
        if self.grid_map[goal] != 0:
            raise ValueError(f"Goal {goal} in obstacle")

        # A* search with enhanced cost function
        open_set = []
        heapq.heappush(open_set, (self.heuristic(start, goal), 0, start))
        came_from = {}
        cost_so_far = {start: 0}

        while open_set:
            _, cost, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                # Convert to world coordinates
                world_path = [self.grid_to_world(p) for p in path]
                
                # Apply smoothing
                smoothed = self.smooth_path(world_path)
                
                return smoothed

            for neighbor, move_cost in self.get_neighbors(current):
                # Multi-factor cost calculation
                base_cost = move_cost
                
                # 1. Clearance penalty (stay away from obstacles)
                clearance = self.distance_map[neighbor[0], neighbor[1]]
                clearance_penalty = (1.0 - clearance) * 12.0  # Increased from 8.0
                
                # 2. Safety buffer penalty (extra cost near obstacles)
                if self.safety_buffer[neighbor[0], neighbor[1]] > 0:
                    buffer_penalty = 5.0
                else:
                    buffer_penalty = 0.0
                
                # 3. Direction change penalty (prefer straight paths)
                direction_penalty = 0.0
                if current in came_from:
                    prev = came_from[current]
                    prev_dir = (current[0] - prev[0], current[1] - prev[1])
                    curr_dir = (neighbor[0] - current[0], neighbor[1] - current[1])
                    # Penalize direction changes
                    if prev_dir != curr_dir:
                        direction_penalty = 0.5
                
                total_cost = base_cost + clearance_penalty + buffer_penalty + direction_penalty
                new_cost = cost + total_cost
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
        
        return None
    
    def smooth_path(self, path, iterations=3):
        """Apply gradient descent smoothing while checking collision"""
        if len(path) <= 2:
            return path
        
        smoothed = list(path)
        
        for _ in range(iterations):
            for i in range(1, len(smoothed) - 1):
                prev = smoothed[i-1]
                curr = smoothed[i]
                next_pt = smoothed[i+1]
                
                # Compute smooth position (average of neighbors)
                smooth_x = 0.5 * (prev[0] + next_pt[0])
                smooth_y = 0.5 * (prev[1] + next_pt[1])
                
                # Blend with current position (0.3 = smoothing strength)
                new_x = 0.7 * curr[0] + 0.3 * smooth_x
                new_y = 0.7 * curr[1] + 0.3 * smooth_y
                
                # Check if smoothed point is collision-free
                if self.is_line_free(curr, (new_x, new_y)):
                    smoothed[i] = (new_x, new_y)
        
        return smoothed
    
    def is_line_free(self, p1, p2, num_samples=5):
        """Check if line segment is collision-free"""
        for t in np.linspace(0, 1, num_samples):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            grid_pos = self.world_to_grid((x, y))
            if not (0 <= grid_pos[0] < self.rows and 0 <= grid_pos[1] < self.cols):
                return False
            if self.grid_map[grid_pos[0], grid_pos[1]] != 0:
                return False
        
        return True