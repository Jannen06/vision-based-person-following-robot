#!/usr/bin/env python3
"""
visualize_goals.py — Debugging tool with Path Safety Validation
This ROS node visualizes candidate navigation goals around a target (e.g., a person),
checks them against a costmap, and validates proposed paths for safety.
"""

import math
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetPlan
from visualization_msgs.msg import Marker, MarkerArray

# --- Tunables ---
# Distances (in meters) from the person to sample potential goals
GOAL_SAMPLE_DIST = [1.0, 1.2, 1.5, 1.8, 2.2, 2.5]
# Number of angular points to sample per distance radius
GOAL_SAMPLE_ANGLES = 16
# Maximum allowable costmap value (0-254) before a space is considered unsafe.
# 60 allows for some inflation but keeps the robot away from hard obstacles.
COSTMAP_SAFE_THRESHOLD = 60


class GoalVisualizer:
    """
    Evaluates and visualizes potential navigation goals around a detected person.
    """

    def __init__(self):
        """Initializes the ROS node, tf listener, publishers, and subscribers."""
        rospy.init_node("goal_visualizer")

        # Set up TF2 buffer and listener to get the robot's current position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # State variables to hold the latest map and person data
        self.global_costmap = None
        self.person_pose = None

        # Wait for the move_base path planning service to become available
        rospy.loginfo("Waiting for /move_base/make_plan service...")
        rospy.wait_for_service('/move_base/make_plan')
        self.make_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)

        # Publishers for RViz visualization
        self.marker_pub = rospy.Publisher("/debug/goal_candidates", MarkerArray, queue_size=1, latch=True)
        self.plan_pub = rospy.Publisher("/debug/proposed_path", Path, queue_size=1, latch=True)

        # Subscribers for the data inputs
        rospy.Subscriber("/person_pose", PoseStamped, self.person_cb)
        rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_cb)

        rospy.loginfo("Goal Visualizer Ready. Waiting for /person_pose...")

    def costmap_cb(self, msg):
        """Callback to store the latest global costmap."""
        self.global_costmap = msg

    def person_cb(self, msg):
        """
        Callback triggered when a new person pose is received. 
        Updates the pose and immediately starts the visualization logic.
        """
        rospy.loginfo(f"Received Person Pose: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        self.person_pose = msg
        self.visualize_scenarios()

    def _robot_pose(self):
        """
        Looks up the robot's current position in the map frame.

        Returns:
            geometry_msgs/Vector3: The translation (x, y, z) of the robot, or None if lookup fails.
        """
        try:
            # Look up the transform from the map frame to the robot's base footprint
            t = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0), rospy.Duration(0.5))
            return t.transform.translation
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"Could not get robot pose: {e}")
            return None

    def _get_cost(self, x, y):
        """
        Retrieves the costmap value at a specific (x, y) coordinate.

        Returns:
            int: The cost value (0-254), or 100 if out of bounds/unknown.
        """
        if self.global_costmap is None:
            return 0  # Default to 0 if costmap hasn't loaded yet

        info = self.global_costmap.info

        # Convert physical (x, y) coordinates to grid map indices
        mx = int((x - info.origin.position.x) / info.resolution)
        my = int((y - info.origin.position.y) / info.resolution)

        # Check if the calculated indices are outside the bounds of the map
        if not (0 <= mx < info.width and 0 <= my < info.height):
            return 100

        # Calculate the 1D array index from the 2D grid coordinates
        idx = my * info.width + mx
        cost = self.global_costmap.data[idx]

        # -1 means the space is unknown; treat it as an obstacle (cost 100) for safety
        if cost == -1:
            return 100
        return cost

    def _is_position_free(self, x, y):
        """Checks if a specific coordinate is below the safe cost threshold."""
        return self._get_cost(x, y) < COSTMAP_SAFE_THRESHOLD

    def _validate_path(self, plan):
        """
        Checks a proposed path to ensure it does not cross over high-cost map areas.

        Returns:
            bool: True if the path is safe, False otherwise.
        """
        if not plan.plan.poses:
            return False

        # Step through the path poses
        for i, pose in enumerate(plan.plan.poses):
            # Check every 5th pose to save computation time while still ensuring safety
            if i % 5 != 0:
                continue

            cx = pose.pose.position.x
            cy = pose.pose.position.y
            cost = self._get_cost(cx, cy)

            if cost >= COSTMAP_SAFE_THRESHOLD:
                rospy.logwarn(f"Path rejected! Point ({cx:.2f}, {cy:.2f}) has cost {cost}")
                return False

        return True

    def create_marker(self, marker_id, x, y, r, g, b, a=1.0, scale=0.15, ns="candidates"):
        """
        Helper function to create an RViz spherical marker.
        """
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.3  # Float it slightly above the ground
        m.pose.orientation.w = 1.0  # Valid quaternion
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color = ColorRGBA(r, g, b, a)
        m.lifetime = rospy.Duration(0)  # 0 means the marker lives forever until deleted
        return m

    def visualize_scenarios(self):
        """
        Main logic: samples goals, filters them, asks for plans, and publishes visualizations.
        """
        if not self.person_pose:
            return

        robot_pos = self._robot_pose()
        if not robot_pos:
            return

        px, py = self.person_pose.pose.position.x, self.person_pose.pose.position.y
        candidates = []
        markers = MarkerArray()

        # Step 1: Tell RViz to clear previous markers
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        # Calculate the angle between the robot and the person to use as a baseline
        base_angle = math.atan2(robot_pos.y - py, robot_pos.x - px)
        marker_id = 0

        # Step 2: Generate candidate points in a circle around the person
        for dist in GOAL_SAMPLE_DIST:
            for i in range(GOAL_SAMPLE_ANGLES):
                angle_offset = (2 * math.pi * i) / GOAL_SAMPLE_ANGLES
                angle = base_angle + angle_offset

                gx = px + dist * math.cos(angle)
                gy = py + dist * math.sin(angle)

                # Step 3: Check if the generated point is free of obstacles
                if self._is_position_free(gx, gy):
                    # Green marker: Valid candidate space
                    markers.markers.append(
                        self.create_marker(marker_id, gx, gy, 0.0, 1.0, 0.0, 0.5, 0.1, "free")
                    )

                    # Calculate a score: lower score is better.
                    # We penalize distance from the person heavily (dist * 2.0)
                    # and factor in the driving distance from the robot.
                    person_cost = dist * 2.0
                    drive_cost = math.hypot(gx - robot_pos.x, gy - robot_pos.y)
                    score = person_cost + drive_cost
                    candidates.append((score, gx, gy, marker_id))
                else:
                    # Red marker: Blocked by obstacle
                    markers.markers.append(
                        self.create_marker(marker_id, gx, gy, 1.0, 0.0, 0.0, 0.8, 0.1, "blocked")
                    )

                marker_id += 1

        # Sort valid candidates so the lowest (best) score is first
        candidates.sort(key=lambda x: x[0])

        # Prepare the starting pose for path planning requests
        start = PoseStamped()
        start.header.frame_id = "map"
        start.pose.position = robot_pos
        start.pose.orientation.w = 1.0

        found_best = False
        check_limit = 20  # Only query move_base for the top 20 candidates to save time

        # Step 4: Validate paths for the top candidates
        for i, (score, gx, gy, m_id) in enumerate(candidates[:check_limit]):
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.pose.position.x = gx
            goal.pose.position.y = gy
            goal.pose.orientation.w = 1.0

            try:
                # Request a path from the move_base service with a 0.2m tolerance
                plan_resp = self.make_plan(start, goal, 0.2)

                if plan_resp.plan.poses:
                    # Validate that the returned path is physically safe
                    if self._validate_path(plan_resp):
                        if not found_best:
                            rospy.loginfo(f"✓ BEST GOAL: ({gx:.2f}, {gy:.2f})")

                            # Publish the successful path
                            path_msg = plan_resp.plan
                            path_msg.header.frame_id = "map"
                            path_msg.header.stamp = rospy.Time.now()
                            self.plan_pub.publish(path_msg)

                            # Blue marker: The chosen goal
                            markers.markers.append(
                                self.create_marker(999, gx, gy, 0.0, 0.5, 1.0, 1.0, 0.35, "selected")
                            )
                            found_best = True
                    else:
                        # Orange marker: A path was found, but it cuts dangerously close to obstacles
                        markers.markers.append(
                            self.create_marker(m_id + 5000, gx, gy, 1.0, 0.5, 0.0, 1.0, 0.15, "unsafe_path")
                        )
                else:
                    # Orange/Yellow marker: Move_base could not find a path here at all
                    markers.markers.append(
                        self.create_marker(m_id + 5000, gx, gy, 1.0, 0.6, 0.0, 1.0, 0.15, "unreachable")
                    )
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"make_plan service call failed: {e}")

        # Finally, publish all constructed markers to RViz
        self.marker_pub.publish(markers)


if __name__ == "__main__":
    try:
        GoalVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
