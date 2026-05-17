#!/usr/bin/env python3
"""
waypoint_extractor.py - Extract sparse waypoints from dense paths.

This module provides a standalone, framework-agnostic (ROS1/ROS2 compatible) 
utility for reducing a dense list of path coordinates (e.g., from an A* planner) 
into a sparse set of navigational waypoints based on distance and angular changes.
"""

import math
from typing import List, Tuple


class WaypointExtractor:
    """
    Utility class for extracting strategic waypoints from dense continuous paths.
    """

    @staticmethod
    def extract_waypoints(path: List[Tuple[float, float]],
                          method: str = 'distance',
                          distance_threshold: float = 1.0,
                          angle_threshold: float = 0.4) -> List[Tuple[float, float]]:
        """
        Extracts a sparse list of waypoints from a dense path array.

        Args:
            path (List[Tuple[float, float]]): The dense path as a list of (x, y) coordinates.
            method (str): The extraction method to use. Options are:
                          - 'distance': A new waypoint every `distance_threshold` meters.
                          - 'angle': A new waypoint at sharp direction changes.
                          - 'combined': Uses both distance and angle criteria.
            distance_threshold (float): The minimum distance (in meters) between waypoints.
            angle_threshold (float): The minimum angular change (in radians) required 
                                     to trigger a new waypoint.

        Returns:
            List[Tuple[float, float]]: A sparse list of (x, y) coordinates representing the waypoints.
        """
        if len(path) <= 2:
            return path

        waypoints = [path[0]]
        minimum_separation = distance_threshold * 0.5

        if method == 'distance':
            current_waypoint = path[0]
            for point in path[1:]:
                dist = math.sqrt((point[0] - current_waypoint[0])**2 +
                                 (point[1] - current_waypoint[1])**2)
                if dist >= distance_threshold:
                    waypoints.append(point)
                    current_waypoint = point

        elif method == 'angle' or method == 'combined':
            current_waypoint = path[0]
            for i in range(1, len(path) - 1):
                point = path[i]
                next_point = path[i+1]

                dist = math.sqrt((point[0] - current_waypoint[0])**2 +
                                 (point[1] - current_waypoint[1])**2)

                prev_point = path[i-1]
                angle1 = math.atan2(point[1] - prev_point[1], point[0] - prev_point[0])
                angle2 = math.atan2(next_point[1] - point[1], next_point[0] - point[0])

                angle_diff = abs(angle2 - angle1)
                angle_diff = min(angle_diff, 2 * math.pi - angle_diff)

                is_far_enough = (dist >= distance_threshold)
                is_sharp_turn = (angle_diff > angle_threshold and dist > minimum_separation)

                if method == 'combined' and (is_far_enough or is_sharp_turn):
                    waypoints.append(point)
                    current_waypoint = point
                elif method == 'angle' and is_sharp_turn:
                    waypoints.append(point)
                    current_waypoint = point

        last_dist = math.sqrt((path[-1][0] - waypoints[-1][0])**2 +
                              (path[-1][1] - waypoints[-1][1])**2)

        if last_dist < 0.2 and len(waypoints) > 1:
            waypoints[-1] = path[-1]
        else:
            waypoints.append(path[-1])

        return waypoints
