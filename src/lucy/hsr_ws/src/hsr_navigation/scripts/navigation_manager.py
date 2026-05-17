#!/usr/bin/env python3
"""
navigation_manager.py - Restaurant Service Workflow

This module manages the high-level state machine and navigation workflow for a 
restaurant service robot. It handles state transitions between idling, taking 
orders, returning to the bar, and delivering items, while coordinating with 
the A* planner and waypoint extractor for movement.
"""
from waypoints_extractor import WaypointExtractor
from a_star_planner import AStarPlanner
import sys
import os
import rospy
import numpy as np
import cv2
import math
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, ColorRGBA
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from tf.transformations import euler_from_quaternion, quaternion_from_euler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class NavigationManager:
    """
    Manages the service workflow and coordinates path planning and execution.
    """

    def __init__(self):
        """
        Initializes the NavigationManager node, loads parameters, sets up ROS 
        publishers/subscribers, and prepares the transform listener.
        """
        rospy.init_node('navigation_manager')
        rospy.loginfo("Navigation Manager: Initialized with Queue-Ready Workflow")

        self.robot_radius = rospy.get_param('~robot_radius', 0.45)
        self.waypoint_distance = rospy.get_param('~waypoint_distance', 1.0)
        self.waypoint_angle = rospy.get_param('~waypoint_angle', 0.4)
        self.waypoint_method = rospy.get_param('~waypoint_method', 'distance')
        self.replan_distance = rospy.get_param('~replan_distance', 2.0)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.5)
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 0.6)

        self.inflated_map = None
        self.resolution = None
        self.origin = None
        self.start_pose = None

        self.state = "IDLE"

        self.person_pose = None
        self.customer_parking_spot = None
        self.final_goal_pose = None
        self.home_orientation = None
        self.home_point_odom = None
        self.outbound_path = None

        self.order_taken = False
        self.items_ready = False

        self.idle_start_time = rospy.Time.now()

        self.current_target_pose = None
        self.goal_orientation = None
        self.current_waypoints = []
        self.current_waypoint_index = 0
        self.navigating = False
        self.goal_received = False
        self.last_replan_time = rospy.Time(0)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber('/map', OccupancyGrid, self.map_cb, queue_size=1)
        rospy.Subscriber('/person_pose', PoseStamped, self.person_detected_cb)
        rospy.Subscriber('/goal_pose', PoseStamped, self.manual_goal_cb)
        rospy.Subscriber('/flag_in', String, self.flag_in_cb)

        self.waypoint_pub = rospy.Publisher('/waypoint_goal', PoseStamped, queue_size=1)
        self.speech_pub = rospy.Publisher('/speak', String, queue_size=5)
        self.flag_out_pub = rospy.Publisher('/flag_out', String, queue_size=5)

        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=1, latch=True)
        self.waypoint_viz_pub = rospy.Publisher('/waypoints_viz', MarkerArray, queue_size=1, latch=True)
        self.costmap_pub = rospy.Publisher('/navigation/costmap', OccupancyGrid, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher('/navigation/status', String, queue_size=1)

        rospy.Timer(rospy.Duration(0.5), self.try_plan)
        rospy.Timer(rospy.Duration(1.0), self.check_replan)
        rospy.Timer(rospy.Duration(0.2), self.check_waypoint_progression)
        rospy.Timer(rospy.Duration(1.0), self.check_idle)

        self._wait_for_localization()
        self._speak("I am Lucy, and I am ready to serve.")
        rospy.loginfo("STATE: IDLE - Waiting for customers")

    def flag_in_cb(self, msg: String):
        """
        Callback to handle incoming workflow triggers from the queue or HRI managers.

        Args:
            msg (std_msgs.msg.String): The incoming flag command (e.g., "order_taken", "items_ready").
        """
        command = msg.data

        if command == "order_taken" and self.state == "TAKING_ORDER":
            rospy.loginfo("✓ 'order_taken' flag received")
            self.order_taken = True
            self._trigger_return_from_order()

        elif command == "items_ready" and self.state == "WAITING_FOR_ITEMS":
            rospy.loginfo("✓ 'items_ready' flag received")
            self.items_ready = True
            self._trigger_delivery()

        elif command == "customer_reached" and self.navigating:
            if self.current_waypoint_index == len(self.current_waypoints) - 1:
                self.handle_arrival()
            else:
                self.current_waypoint_index += 1
                if self.current_waypoint_index < len(self.current_waypoints):
                    self.publish_next_waypoint()

    def person_detected_cb(self, msg: PoseStamped):
        """
        Callback triggered when a new customer is detected. Initiates the service workflow.

        Args:
            msg (geometry_msgs.msg.PoseStamped): The pose of the detected customer.
        """
        if self.state not in ["IDLE", "RETURNING_HOME"]:
            rospy.logwarn(f"Person detected but robot busy (state={self.state})")
            return

        if self.state == "RETURNING_HOME":
            rospy.loginfo("Interrupting return trip to serve new customer in queue!")
            self.navigating = False

        person_pos = (msg.pose.position.x, msg.pose.position.y)
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.person_pose = person_pos
        self.customer_parking_spot = self.find_parking_spot(self.person_pose)

        rospy.loginfo(f"✓ CUSTOMER DETECTED at ({person_pos[0]:.2f}, {person_pos[1]:.2f})")

        self.final_goal_pose = self.customer_parking_spot
        dy = self.person_pose[1] - self.customer_parking_spot[1]
        dx = self.person_pose[0] - self.customer_parking_spot[0]
        self.goal_orientation = math.atan2(dy, dx)
        self.state = "TAKING_ORDER"
        self.goal_received = True
        self.outbound_path = None
        self.order_taken = False

        rospy.loginfo("STATE: IDLE → TAKING_ORDER")

    def manual_goal_cb(self, msg: PoseStamped):
        """
        Fallback callback to manually trigger navigation via rviz or command line.

        Args:
            msg (geometry_msgs.msg.PoseStamped): The manually specified goal pose.
        """
        if self.state in ["IDLE", "RETURNING_HOME"]:
            self.person_detected_cb(msg)

    def _trigger_return_from_order(self):
        """
        Transitions the workflow to return to the home/bar location after an order is taken.
        Attempts to reuse the reversed outbound path if available.
        """
        rospy.loginfo("STATE: TAKING_ORDER → RETURNING_FROM_ORDER")
        self._speak("Order received. Returning to get your items.")

        try:
            tf = self.tf_buffer.lookup_transform("map", "odom", rospy.Time(0), rospy.Duration(1.0))
            home_map = do_transform_point(self.home_point_odom, tf)
            self.final_goal_pose = (home_map.point.x, home_map.point.y)
            self.goal_orientation = self.home_orientation
            self.state = "RETURNING_FROM_ORDER"

            if self.outbound_path and len(self.outbound_path) > 2:
                self._navigate_reverse_path()
            else:
                self.goal_received = True

        except Exception as e:
            rospy.logerr(f"Failed to return home: {e}")
            self.state = "IDLE"

    def _trigger_delivery(self):
        """
        Transitions the workflow to deliver items back to the waiting customer.
        """
        if self.customer_parking_spot is None:
            self.state = "IDLE"
            return

        rospy.loginfo("STATE: WAITING_FOR_ITEMS → DELIVERING")
        self._speak("Items ready. Delivering to customer.")

        self.final_goal_pose = self.customer_parking_spot
        dy = self.person_pose[1] - self.customer_parking_spot[1]
        dx = self.person_pose[0] - self.customer_parking_spot[0]
        self.goal_orientation = math.atan2(dy, dx)
        self.state = "DELIVERING"
        self.goal_received = True
        self.outbound_path = None

    def _complete_service(self):
        """
        Finalizes the delivery, resets the state machine variables, and notifies 
        the system that the robot is ready for the next customer.
        """
        rospy.loginfo("STATE: DELIVERING → IDLE (Service Complete)")
        self._speak("Mission complete. Ready for next customer.")

        self.state = "IDLE"
        self.person_pose = None
        self.customer_parking_spot = None
        self.outbound_path = None
        self.order_taken = False
        self.items_ready = False
        self.publish_waypoints([])
        self.idle_start_time = rospy.Time.now()

        rospy.sleep(0.5)
        self.flag_out_pub.publish(String(data="delivery_complete"))
        rospy.loginfo("Published 'delivery_complete' — queue manager will decide next step")

    def _trigger_return_home_idle(self):
        """
        Triggers a return to the base station when the robot has been idle for too long.
        """
        rospy.loginfo("STATE: IDLE → RETURNING_HOME (Queue Empty)")
        self._speak("No pending orders. Returning to base.")
        try:
            tf_trans = self.tf_buffer.lookup_transform("map", "odom", rospy.Time(0), rospy.Duration(1.0))
            home_map = do_transform_point(self.home_point_odom, tf_trans)
            self.final_goal_pose = (home_map.point.x, home_map.point.y)
            self.goal_orientation = self.home_orientation
            self.state = "RETURNING_HOME"
            self.goal_received = True
            self.outbound_path = None
        except Exception as e:
            rospy.logerr(f"Failed to return home: {e}")

    def check_idle(self, event=None):
        """
        Timer callback that monitors the idle duration and triggers a return home 
        if the threshold is exceeded.
        """
        if self.state == "IDLE" and not self.navigating:
            try:
                elapsed = (rospy.Time.now() - self.idle_start_time).to_sec()
                if elapsed > 15.0:
                    robot_pos = self.get_robot_pose()
                    if robot_pos and self.home_point_odom:
                        tf_trans = self.tf_buffer.lookup_transform(
                            "map", "odom", rospy.Time(0), rospy.Duration(0.5))
                        home_map = do_transform_point(self.home_point_odom, tf_trans)
                        dist = math.hypot(
                            home_map.point.x - robot_pos[0],
                            home_map.point.y - robot_pos[1]
                        )
                        if dist > 1.0:
                            self._trigger_return_home_idle()
                        else:
                            self.idle_start_time = rospy.Time.now()
            except Exception:
                pass

    def _navigate_reverse_path(self):
        """
        Reverses the cached outbound path to generate an efficient return route.
        """
        reversed_path = list(reversed(self.outbound_path))
        waypoints = WaypointExtractor.extract_waypoints(
            reversed_path,
            method=self.waypoint_method,
            distance_threshold=self.waypoint_distance,
            angle_threshold=self.waypoint_angle
        )
        self.current_waypoints = waypoints
        self.current_waypoint_index = 0
        self.navigating = True
        self.last_replan_time = rospy.Time.now()
        self.publish_path(reversed_path)
        self.publish_waypoints(waypoints)
        if waypoints:
            self.publish_next_waypoint()

    def handle_arrival(self):
        """
        Handles the logic executed upon reaching the final waypoint of a route, 
        progressing the state machine accordingly.
        """
        self.navigating = False
        self.current_waypoints = []
        self.status_pub.publish(String(data="goal_reached"))

        if self.state == "TAKING_ORDER":
            rospy.loginfo("✓ ARRIVED at customer for order")
            self._speak("I have arrived. Please tell me your order.")
            self.flag_out_pub.publish(String(data="customer_reached"))
            rospy.loginfo("  Waiting for 'order_taken' flag...")

        elif self.state == "RETURNING_FROM_ORDER":
            rospy.loginfo("✓ ARRIVED home after taking order")
            self._speak("I am back. Waiting for your items.")
            self.flag_out_pub.publish(String(data="bar_reached"))
            self.state = "WAITING_FOR_ITEMS"
            rospy.loginfo("STATE: RETURNING_FROM_ORDER → WAITING_FOR_ITEMS")
            rospy.loginfo("  Waiting for 'items_ready' flag...")

        elif self.state == "DELIVERING":
            rospy.loginfo("✓ ARRIVED at customer for delivery")
            self._speak("Here are your items. Enjoy!")
            self.flag_out_pub.publish(String(data="delivery_complete"))
            rospy.Timer(rospy.Duration(5.0), lambda e: self._complete_service(), oneshot=True)

        elif self.state == "RETURNING_HOME":
            rospy.loginfo("✓ ARRIVED at base from idle timeout.")
            self._speak("I am back at my station.")
            self.state = "IDLE"
            self.idle_start_time = rospy.Time.now()
            self.flag_out_pub.publish(String(data="home_reached"))

    def _speak(self, text: str):
        """
        Publishes a text string to the speech synthesis node.

        Args:
            text (str): The sentence for the robot to speak.
        """
        self.speech_pub.publish(String(data=text))
        rospy.loginfo(f"[SPEECH] {text}")

    def _wait_for_localization(self):
        """
        Blocks initialization until the robot's transform tree is available, 
        saving the initial start location as the permanent home base.
        """
        rospy.loginfo("Waiting for TF to save Home position...")
        rate = rospy.Rate(1)
        while not rospy.is_shutdown() and self.home_point_odom is None:
            try:
                self.tf_buffer.lookup_transform(
                    "map", "base_footprint", rospy.Time(0), rospy.Duration(0.5))
                t = self.tf_buffer.lookup_transform(
                    "odom", "base_footprint", rospy.Time(0), rospy.Duration(0.5))
                pt = PointStamped()
                pt.header.frame_id = "odom"
                pt.point = t.transform.translation
                q = t.transform.rotation
                _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
                self.home_orientation = yaw
                self.home_point_odom = pt
                rospy.loginfo(
                    f"Home saved: pos=({pt.point.x:.2f}, {pt.point.y:.2f}), "
                    f"yaw={math.degrees(yaw):.1f}°"
                )
            except Exception:
                rate.sleep()

    def map_cb(self, msg: OccupancyGrid):
        """
        Callback to receive and process the global costmap, applying a dilation 
        filter based on the robot's radius.

        Args:
            msg (nav_msgs.msg.OccupancyGrid): The raw map data.
        """
        try:
            self.resolution = msg.info.resolution
            self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)
            width = msg.info.width
            height = msg.info.height
            data = np.array(msg.data, dtype=np.int8).reshape((height, width))
            obstacles = np.zeros((height, width), dtype=np.uint8)
            obstacles[data > 50] = 1
            robot_radius_cells = int(self.robot_radius / self.resolution)
            kernel = np.ones(
                (2 * robot_radius_cells + 1, 2 * robot_radius_cells + 1), dtype=np.uint8)
            self.inflated_map = cv2.dilate(obstacles, kernel, iterations=1)
        except Exception:
            pass

    def find_parking_spot(self, person_pt):
        """
        Identifies a collision-free target point near the detected customer for 
        the robot to navigate towards.

        Args:
            person_pt (tuple): The (x, y) coordinates of the customer.

        Returns:
            tuple: The calculated (x, y) target coordinates.
        """
        robot_pos = self.get_robot_pose()
        if not robot_pos or self.inflated_map is None:
            return person_pt

        px, py = person_pt
        rx, ry = robot_pos
        base_angle = math.atan2(py - ry, px - rx)

        candidates = []
        sample_distances = [1.0, 1.2, 1.5]

        for dist in sample_distances:
            for i in range(16):
                angle = base_angle + (2 * math.pi * i) / 16
                gx = px + dist * math.cos(angle)
                gy = py + dist * math.sin(angle)
                grid_x = int((gx - self.origin[0]) / self.resolution)
                grid_y = int((gy - self.origin[1]) / self.resolution)
                if (0 <= grid_y < self.inflated_map.shape[0] and
                        0 <= grid_x < self.inflated_map.shape[1]):
                    if self.inflated_map[grid_y, grid_x] == 0:
                        score = dist * 10.0 + math.hypot(gx - rx, gy - ry)
                        candidates.append((score, (gx, gy)))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        return person_pt

    def get_robot_pose(self):
        """
        Retrieves the robot's current (x, y) position in the map frame.

        Returns:
            tuple: The (x, y) coordinates of the robot, or None if the lookup fails.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_footprint", rospy.Time(0), rospy.Duration(0.2))
            return (t.transform.translation.x, t.transform.translation.y)
        except Exception:
            return None

    def check_waypoint_progression(self, event=None):
        """
        Timer callback that verifies if the robot has reached its current waypoint 
        and updates the index to publish the next one.
        """
        if not self.navigating or not self.current_waypoints:
            return
        robot_pos = self.get_robot_pose()
        if not robot_pos:
            return

        is_final = (self.current_waypoint_index == len(self.current_waypoints) - 1)
        wp = self.current_waypoints[self.current_waypoint_index]

        if is_final:
            dist_to_final = math.hypot(
                self.final_goal_pose[0] - robot_pos[0],
                self.final_goal_pose[1] - robot_pos[1]
            )
            if dist_to_final <= self.goal_tolerance:
                rospy.loginfo(f"Final waypoint reached ({dist_to_final:.2f}m)")
                self.handle_arrival()
            return

        dist_to_wp = math.hypot(wp[0] - robot_pos[0], wp[1] - robot_pos[1])
        if dist_to_wp < self.lookahead_distance:
            self.current_waypoint_index += 1
            if self.current_waypoint_index < len(self.current_waypoints):
                self.publish_next_waypoint()

    def find_reachable_goal(self, final_goal):
        """
        Validates the goal against the inflated map. If the goal lies inside an 
        obstacle, it walks backwards towards the robot to find a valid free cell.

        Args:
            final_goal (tuple): The intended (x, y) goal coordinates.

        Returns:
            tuple: The nearest collision-free (x, y) target coordinates.
        """
        robot_pos = self.get_robot_pose()
        if robot_pos is None:
            return final_goal

        if self.state in ["RETURNING_FROM_ORDER", "DELIVERING", "RETURNING_HOME"]:
            return final_goal

        num_samples = 50
        for i in range(num_samples, 0, -1):
            t = i / num_samples
            test_x = robot_pos[0] + t * (final_goal[0] - robot_pos[0])
            test_y = robot_pos[1] + t * (final_goal[1] - robot_pos[1])
            grid_x = int((test_x - self.origin[0]) / self.resolution)
            grid_y = int((test_y - self.origin[1]) / self.resolution)
            if (0 <= grid_y < self.inflated_map.shape[0] and
                    0 <= grid_x < self.inflated_map.shape[1]):
                if self.inflated_map[grid_y, grid_x] == 0:
                    return (test_x, test_y)

        return (robot_pos[0] + 0.5, robot_pos[1])

    def try_plan(self, event=None):
        """
        Timer callback that initiates an A* path planning request if a new goal 
        has been received. Parses the resulting path into waypoints.
        """
        if self.inflated_map is None or not self.goal_received or self.navigating:
            return

        self.start_pose = self.get_robot_pose()
        if self.start_pose is None:
            return

        self.current_target_pose = self.find_reachable_goal(self.final_goal_pose)
        self.status_pub.publish(String(data="planning"))
        rospy.loginfo(
            f"A* Planning from ({self.start_pose[0]:.2f}, {self.start_pose[1]:.2f}) "
            f"to ({self.current_target_pose[0]:.2f}, {self.current_target_pose[1]:.2f})"
        )

        planner = AStarPlanner(self.inflated_map, self.resolution, self.origin)

        try:
            path = planner.plan(self.start_pose, self.current_target_pose)
        except ValueError as e:
            rospy.logwarn(f"A* error: {e}")
            self.goal_received = False
            self._handle_planning_failure()
            return

        if path is None:
            rospy.logerr("No path found")
            self.goal_received = False
            self._handle_planning_failure()
            return

        if self.state in ["TAKING_ORDER", "DELIVERING"]:
            self.outbound_path = path
            rospy.loginfo(f"Saved {len(path)} path points for return trip")

        waypoints = WaypointExtractor.extract_waypoints(
            path,
            method=self.waypoint_method,
            distance_threshold=self.waypoint_distance,
            angle_threshold=self.waypoint_angle
        )

        rospy.loginfo(f"Path ready: {len(waypoints)} waypoints")
        self.current_waypoints = waypoints
        self.current_waypoint_index = 0
        self.navigating = True
        self.last_replan_time = rospy.Time.now()

        self.publish_path(path)
        self.publish_waypoints(waypoints)

        if waypoints:
            self.publish_next_waypoint()
        self.goal_received = False

    def _handle_planning_failure(self):
        """
        Recovers the workflow state if the A* planner fails to find a valid route.
        Resets parameters to IDLE and signals external queues to proceed.
        """
        rospy.logwarn(
            f"Planning failed in state '{self.state}' — resetting to IDLE"
        )
        self.state = "IDLE"
        self.person_pose = None
        self.customer_parking_spot = None
        self.outbound_path = None
        self.order_taken = False
        self.items_ready = False
        self.navigating = False
        self.idle_start_time = rospy.Time.now()

        self.flag_out_pub.publish(String(data="delivery_complete"))
        rospy.loginfo("Published 'delivery_complete' after planning failure — queue will recover")

    def check_replan(self, event=None):
        """
        Timer callback that monitors the robot's distance to the target and triggers 
        a recalculation of the path if a threshold is crossed.
        """
        if not self.navigating or self.final_goal_pose is None or self.current_target_pose is None:
            return
        robot_pos = self.get_robot_pose()
        if robot_pos is None:
            return

        dist_to_target = math.hypot(
            self.current_target_pose[0] - robot_pos[0],
            self.current_target_pose[1] - robot_pos[1]
        )
        dist_to_final = math.hypot(
            self.final_goal_pose[0] - robot_pos[0],
            self.final_goal_pose[1] - robot_pos[1]
        )
        dist_target_to_final = math.hypot(
            self.final_goal_pose[0] - self.current_target_pose[0],
            self.final_goal_pose[1] - self.current_target_pose[1]
        )

        if (dist_target_to_final > 0.5 and
                dist_to_target < self.replan_distance and
                dist_to_final > self.goal_tolerance):
            if (rospy.Time.now() - self.last_replan_time).to_sec() > 2.0:
                rospy.loginfo("Approaching border. Replanning.")
                self.navigating = False
                self.goal_received = True

    def publish_next_waypoint(self):
        """
        Publishes the upcoming waypoint to the local movement controller, adjusting 
        the orientation based on the final goal or the trajectory of the path.
        """
        if self.current_waypoint_index >= len(self.current_waypoints):
            return

        wp = self.current_waypoints[self.current_waypoint_index]
        is_final = (self.current_waypoint_index == len(self.current_waypoints) - 1)

        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = wp[0]
        msg.pose.position.y = wp[1]
        msg.pose.position.z = 0.0

        if is_final and self.goal_orientation is not None:
            q = quaternion_from_euler(0, 0, self.goal_orientation)
        else:
            if self.current_waypoint_index < len(self.current_waypoints) - 1:
                next_wp = self.current_waypoints[self.current_waypoint_index + 1]
                yaw = math.atan2(next_wp[1] - wp[1], next_wp[0] - wp[0])
            else:
                yaw = 0.0
            q = quaternion_from_euler(0, 0, yaw)

        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        rospy.loginfo(f"Waypoint {self.current_waypoint_index + 1}/{len(self.current_waypoints)}")
        self.waypoint_pub.publish(msg)

    def publish_path(self, path):
        """
        Publishes the continuous A* path to rviz for visual debugging.

        Args:
            path (list): The list of (x, y) coordinates forming the path.
        """
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        for x, y in path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def publish_waypoints(self, waypoints):
        """
        Publishes marker arrays to visualize the extracted waypoints, the final 
        goal, and the customer's position in rviz.

        Args:
            waypoints (list): The list of (x, y) coordinates serving as intermediate targets.
        """
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)
        self.waypoint_viz_pub.publish(markers)
        rospy.sleep(0.05)

        markers = MarkerArray()

        if self.person_pose and self.state != "IDLE":
            pm = Marker()
            pm.header.frame_id = "map"
            pm.header.stamp = rospy.Time.now()
            pm.ns = "person_target"
            pm.id = 999
            pm.type = Marker.SPHERE
            pm.action = Marker.ADD
            pm.pose.position.x = self.person_pose[0]
            pm.pose.position.y = self.person_pose[1]
            pm.pose.position.z = 0.5
            pm.pose.orientation.w = 1.0
            pm.scale.x = pm.scale.y = pm.scale.z = 0.4
            pm.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
            markers.markers.append(pm)

            pt = Marker()
            pt.header = pm.header
            pt.ns = "labels"
            pt.id = 1000
            pt.type = Marker.TEXT_VIEW_FACING
            pt.action = Marker.ADD
            pt.pose.position.x = self.person_pose[0]
            pt.pose.position.y = self.person_pose[1]
            pt.pose.position.z = 1.0
            pt.scale.z = 0.25
            pt.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            pt.text = "Customer"
            markers.markers.append(pt)

        for i, (x, y) in enumerate(waypoints):
            is_final = (i == len(waypoints) - 1)

            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = "waypoints"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.2
            m.pose.orientation.w = 1.0

            if is_final:
                m.scale.x = m.scale.y = m.scale.z = 0.35
                m.color = ColorRGBA(1.0, 0.5, 0.0, 1.0)
            else:
                m.scale.x = m.scale.y = m.scale.z = 0.3
                m.color = ColorRGBA(0.0, 0.8, 1.0, 1.0)

            markers.markers.append(m)

            if not is_final:
                t = Marker()
                t.header = m.header
                t.ns = "labels"
                t.id = i + 2000
                t.type = Marker.TEXT_VIEW_FACING
                t.action = Marker.ADD
                t.pose.position.x = x
                t.pose.position.y = y
                t.pose.position.z = 0.5
                t.scale.z = 0.2
                t.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
                t.text = str(i + 1)
                markers.markers.append(t)

        self.waypoint_viz_pub.publish(markers)


if __name__ == '__main__':
    try:
        node = NavigationManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
