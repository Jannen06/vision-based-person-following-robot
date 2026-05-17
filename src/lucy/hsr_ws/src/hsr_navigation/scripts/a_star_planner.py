#!/usr/bin/env python3
"""
a_star_planner.py - Optimized A* for restaurant navigation
Reduced penalties for smoother, more direct paths.

This module provides an A* path planning implementation designed for 2D grid maps.
It features custom clearance-based cost functions, direction penalties to favor 
straight paths, and post-processing path smoothing via gradient descent.
"""

import numpy as np
import heapq
import cv2


class AStarPlanner:
    """
    An optimized A* path planner for 2D grid environments.

    This planner calculates a distance map upon initialization to favor paths 
    that maintain a safe distance from obstacles. It also includes methods to 
    convert between real-world coordinates and grid coordinates.
    """

    def __init__(self, grid_map, resolution=0.05, origin=(0.0, 0.0)):
        """
        Initializes the AStarPlanner with map data.

        Args:
            grid_map (numpy.ndarray): A 2D array representing the occupancy grid. 
                                      0 indicates free space, non-zero indicates an obstacle.
            resolution (float): The size of each grid cell in real-world units (e.g., meters/cell).
            origin (tuple): The (x, y) real-world coordinates of the grid's bottom-left or origin cell.
        """
        self.grid_map = grid_map
        self.rows, self.cols = grid_map.shape
        self.resolution = resolution
        self.origin = origin

        # Compute distance map (higher = safer)
        self.distance_map = self.compute_clearance_map(grid_map)

    def compute_clearance_map(self, grid_map):
        """
        Calculates a distance transform of the grid map to represent obstacle clearance.

        Args:
            grid_map (numpy.ndarray): The 2D occupancy grid.

        Returns:
            numpy.ndarray: A 2D float array of the same shape as grid_map. Values range 
                           from 0.0 (obstacle) to 1.0 (furthest from any obstacle).
        """
        binary_free = np.where(grid_map == 0, 1, 0).astype(np.uint8)
        dist_transform = cv2.distanceTransform(binary_free, cv2.DIST_L2, 5)
        max_dist = np.max(dist_transform)
        return dist_transform / (max_dist + 1e-5)

    def get_nearest_free_cell(self, coord, max_radius_cells=20):
        """
        Finds the nearest free cell to a given coordinate using a spiraling search.
        Useful for snapping invalid start/goal points into valid free space.

        Args:
            coord (tuple): The (row, col) grid coordinates to check.
            max_radius_cells (int): The maximum search radius in grid cells.

        Returns:
            tuple: The (row, col) of the nearest free cell, or None if no free 
                   cell is found within the maximum radius.
        """
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
        """
        Calculates the Euclidean distance heuristic between two grid cells.

        Args:
            a (tuple): The (row, col) of the first cell.
            b (tuple): The (row, col) of the second cell.

        Returns:
            float: The straight-line Euclidean distance between the two cells.
        """
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def get_neighbors(self, node):
        """
        Identifies valid, obstacle-free neighbor cells in an 8-connected grid.

        Args:
            node (tuple): The (row, col) coordinates of the current cell.

        Returns:
            list: A list of tuples, where each tuple contains:
                  - A (row, col) tuple of the neighbor's coordinates.
                  - A float representing the physical move cost to that neighbor 
                    (1.0 for cardinal, ~1.414 for diagonal).
        """
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),      # cardinal
            (1, 1), (-1, -1), (1, -1), (-1, 1)     # diagonal
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
        """
        Converts grid coordinates (row, col) to real-world coordinates (x, y).

        Args:
            grid_coord (tuple): The (row, col) indices in the grid.

        Returns:
            tuple: The corresponding (x, y) real-world coordinates in meters.
        """
        x = grid_coord[1] * self.resolution + self.origin[0]
        y = grid_coord[0] * self.resolution + self.origin[1]
        return (x, y)

    def world_to_grid(self, world_coord):
        """
        Converts real-world coordinates (x, y) to grid coordinates (row, col).

        Args:
            world_coord (tuple): The (x, y) real-world coordinates in meters.

        Returns:
            tuple: The corresponding integer (row, col) indices in the grid.
        """
        col = int((world_coord[0] - self.origin[0]) / self.resolution)
        row = int((world_coord[1] - self.origin[1]) / self.resolution)
        return (row, col)

    def plan(self, start_world, goal_world):
        """
        Executes the A* search algorithm with customized cost functions balancing 
        safety (clearance) and smoothness (direction penalties).

        Args:
            start_world (tuple): The starting (x, y) real-world coordinates.
            goal_world (tuple): The target (x, y) real-world coordinates.

        Returns:
            list: A list of (x, y) real-world coordinates representing the smoothed 
                  planned path from start to goal. Returns None if no path is found.

        Raises:
            ValueError: If the start or goal points are out of the map bounds, or 
                        if they fall inside an obstacle and cannot be snapped to a free cell.
        """
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

        # A* search with optimized cost function
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

                # Apply aggressive smoothing
                smoothed = self.smooth_path(world_path, iterations=5)

                return smoothed

            for neighbor, move_cost in self.get_neighbors(current):
                # Simplified cost: mild clearance preference + straight path bonus
                base_cost = move_cost

                # Light clearance penalty (reduced from 12.0 to 4.0)
                clearance = self.distance_map[neighbor[0], neighbor[1]]
                clearance_penalty = (1.0 - clearance) * 4.0

                # Encourage straight paths
                direction_penalty = 0.0
                if current in came_from:
                    prev = came_from[current]
                    prev_dir = (current[0] - prev[0], current[1] - prev[1])
                    curr_dir = (neighbor[0] - current[0], neighbor[1] - current[1])
                    if prev_dir != curr_dir:
                        direction_penalty = 0.2  # Reduced from 0.5

                total_cost = base_cost + clearance_penalty + direction_penalty
                new_cost = cost + total_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

        return None

    def smooth_path(self, path, iterations=5):
        """
        Applies aggressive gradient descent smoothing to the generated path 
        specifically tuned for restaurant environments.

        Args:
            path (list): A list of (x, y) real-world coordinates representing the path.
            iterations (int): The number of smoothing iterations to apply.

        Returns:
            list: The smoothed list of (x, y) real-world coordinates.
        """
        if len(path) <= 2:
            return path

        smoothed = list(path)

        for iteration in range(iterations):
            # Increase smoothing strength each iteration
            alpha = 0.3 + (iteration * 0.1)  # 0.3 -> 0.7

            for i in range(1, len(smoothed) - 1):
                prev = smoothed[i-1]
                curr = smoothed[i]
                next_pt = smoothed[i+1]

                # Compute smooth position (average of neighbors)
                smooth_x = 0.5 * (prev[0] + next_pt[0])
                smooth_y = 0.5 * (prev[1] + next_pt[1])

                # Blend with increasing strength
                new_x = (1 - alpha) * curr[0] + alpha * smooth_x
                new_y = (1 - alpha) * curr[1] + alpha * smooth_y

                # Check if smoothed point is collision-free
                if self.is_line_free(curr, (new_x, new_y), num_samples=10):
                    smoothed[i] = (new_x, new_y)

        return smoothed

    def is_line_free(self, p1, p2, num_samples=10):
        """
        Checks if a straight line segment between two real-world points is 
        completely free of obstacles.

        Args:
            p1 (tuple): The (x, y) coordinates of the start of the line segment.
            p2 (tuple): The (x, y) coordinates of the end of the line segment.
            num_samples (int): The number of points to interpolate and check along the line.

        Returns:
            bool: True if the entire line segment is collision-free, False otherwise.
        """
        for t in np.linspace(0, 1, num_samples):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])

            grid_pos = self.world_to_grid((x, y))
            if not (0 <= grid_pos[0] < self.rows and 0 <= grid_pos[1] < self.cols):
                return False
            if self.grid_map[grid_pos[0], grid_pos[1]] != 0:
                return False

        return True
