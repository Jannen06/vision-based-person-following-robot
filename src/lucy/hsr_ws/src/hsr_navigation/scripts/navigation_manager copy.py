#!/usr/bin/env python3
"""
navigation_manager.py - Topic-Based Global Planner with Parking Logic
Listens to /goal_pose, finds a safe parking spot, runs A*, and publishes waypoints.
"""
import sys
import os
import rospy
import numpy as np
import cv2
import math
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool, ColorRGBA, String
import tf2_ros
import tf2_geometry_msgs
from tf2_geometry_msgs import do_transform_point
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Import your separate tools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a_star_planner import AStarPlanner
from waypoints_extractor import WaypointExtractor


class NavigationManager:
    def __init__(self):
        rospy.init_node('navigation_manager')
        rospy.loginfo("Navigation Manager: Initialized with Parking & Speech")
        
        # Parameters
        self.robot_radius = rospy.get_param('~robot_radius', 0.45)
        self.waypoint_distance = rospy.get_param('~waypoint_distance', 1.0)
        self.waypoint_angle = rospy.get_param('~waypoint_angle', 0.4)
        self.waypoint_method = rospy.get_param('~waypoint_method', 'distance')
        self.replan_distance = rospy.get_param('~replan_distance', 2.0)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.4)
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 0.6)
        
        # Navigation State
        self.inflated_map = None
        self.resolution = None
        self.origin = None
        self.start_pose = None
        
        self.state = "IDLE"          # IDLE, MOVING_TO_PERSON, WAITING, MOVING_HOME
        self.person_pose = None      # Exact coordinates of the human
        self.final_goal_pose = None  # The safe parking spot
        self.home_orientation = None  # home orientation
        self.home_point_odom = None  # Physical starting point

        
        self.current_target_pose = None  
        self.goal_orientation = None
        self.current_waypoints = []
        self.current_waypoint_index = 0
        self.navigating = False
        self.goal_received = False
        self.last_replan_time = rospy.Time(0)
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers (Inputs)
        rospy.Subscriber('/map', OccupancyGrid, self.map_cb, queue_size=1)
        rospy.Subscriber('/person_pose', PoseStamped, self.goal_cb) # From Perception Node
        rospy.Subscriber('/goal_pose', PoseStamped, self.goal_cb)   # From RViz (Manual testing)

        rospy.Subscriber('/flag', String, self.flag_cb) 
        
        # Publishers (Outputs)
        self.waypoint_pub = rospy.Publisher('/waypoint_goal', PoseStamped, queue_size=1)
        self.speech_pub = rospy.Publisher('/speak', String, queue_size=5)
        self.flag_pub = rospy.Publisher('/flag', String, queue_size=5) # To tell Perception Node we are home
        
        # Debug / Visualization Publishers
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=1, latch=True)
        self.waypoint_viz_pub = rospy.Publisher('/waypoints_viz', MarkerArray, queue_size=1, latch=True)
        self.costmap_pub = rospy.Publisher('/navigation/costmap', OccupancyGrid, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher('/navigation/status', String, queue_size=1)
        self.temp_goal_pub = rospy.Publisher('/temp_goal_marker', Marker, queue_size=1)
        
        # Timers
        rospy.Timer(rospy.Duration(0.5), self.try_plan)
        rospy.Timer(rospy.Duration(1.0), self.check_replan)
        rospy.Timer(rospy.Duration(0.2), self.check_waypoint_progression)
        
        self._wait_for_localization()
        self._speak("I am Lucy, and I am ready.")

    # Remove this if thisngs get worse
    def flag_cb(self, msg: String):
        """Listen for waypoint completion from behavior_loop"""
        if msg.data == "customer_reached" and self.navigating:
            rospy.loginfo("Controller reported arrival at waypoint")
            
            # Check if we're at the final waypoint
            if self.current_waypoint_index == len(self.current_waypoints) - 1:
                self.handle_arrival()
            else:
                # Intermediate waypoint - advance to next
                self.current_waypoint_index += 1
                if self.current_waypoint_index < len(self.current_waypoints):
                    self.publish_next_waypoint()

    def _speak(self, text: str):
        """Helper to publish text to the TTS engine and log it."""
        self.speech_pub.publish(String(data=text))
        rospy.loginfo(f"[SPEECH] {text}")

    def _wait_for_localization(self):
        rospy.loginfo("Waiting for TF to save Home position...")
        rate = rospy.Rate(1)
        while not rospy.is_shutdown() and self.home_point_odom is None:
            try:
                self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0), rospy.Duration(0.5))
                t = self.tf_buffer.lookup_transform("odom", "base_footprint", rospy.Time(0), rospy.Duration(0.5))
                pt = PointStamped()
                pt.header.frame_id = "odom"
                pt.point = t.transform.translation
                self.home_point_odom = pt

                # Save orientation (ADD THIS)
                q = t.transform.rotation
                _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
                self.home_orientation = yaw
                # rospy.loginfo("\/ Home position saved safely in Odom frame.")
                rospy.loginfo(f"\/ Home saved: pos=({pt.point.x:.2f}, {pt.point.y:.2f}), yaw={math.degrees(yaw):.1f}°")
            except Exception:
                rate.sleep()

    def map_cb(self, msg: OccupancyGrid):
        try:
            self.resolution = msg.info.resolution
            self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)
            width = msg.info.width
            height = msg.info.height
            data = np.array(msg.data, dtype=np.int8).reshape((height, width))
            
            obstacles = np.zeros((height, width), dtype=np.uint8)
            obstacles[data > 50] = 1
            
            robot_radius_cells = int(self.robot_radius / self.resolution)
            kernel = np.ones((2 * robot_radius_cells + 1, 2 * robot_radius_cells + 1), dtype=np.uint8)
            self.inflated_map = cv2.dilate(obstacles, kernel, iterations=1)
        except Exception as e:
            pass

    def find_parking_spot(self, person_pt):
        """Radial search to find closest safe space near the person."""
        robot_pos = self.get_robot_pose()
        if not robot_pos or self.inflated_map is None: 
            return person_pt
            
        px, py = person_pt
        rx, ry = robot_pos
        base_angle = math.atan2(py - ry, px - rx)
        
        candidates = []
        # sample_distances = [1.0, 1.2, 1.5, 1.8, 2.2, 2.5]
        sample_distances = [0.8, 0.9, 1.0, 1.2, 1.5]

        
        for dist in sample_distances:
            for i in range(16):
                angle = base_angle + (2 * math.pi * i) / 16
                gx = px + dist * math.cos(angle)
                gy = py + dist * math.sin(angle)
                
                grid_x = int((gx - self.origin[0]) / self.resolution)
                grid_y = int((gy - self.origin[1]) / self.resolution)
                
                if 0 <= grid_y < self.inflated_map.shape[0] and 0 <= grid_x < self.inflated_map.shape[1]:
                    if self.inflated_map[grid_y, grid_x] == 0:
                        score = dist * 10.0 + math.hypot(gx - rx, gy - ry)
                        candidates.append((score, (gx, gy)))
                        
        if candidates:
            candidates.sort(key=lambda x: x[0])
            best = candidates[0][1]
            rospy.loginfo(f"Found safe parking spot: ({best[0]:.2f}, {best[1]:.2f})")
            return best
            
        rospy.logwarn("Could not find safe parking spot! Defaulting to exact click.")
        return person_pt

    def goal_cb(self, msg: PoseStamped):
        """Receives final destination, calculates parking spot, and triggers A*"""
        if self.state not in ["IDLE", "WAITING"]:
            rospy.logwarn("Busy! Ignoring new goal.")
            return
            
        new_goal = (msg.pose.position.x, msg.pose.position.y)
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # Save human position for visualization
        self.person_pose = new_goal
        
        # Find safe spot to park
        parking_spot = self.find_parking_spot(self.person_pose)
        
        if self.final_goal_pose != parking_spot:
            self.final_goal_pose = parking_spot
            self.goal_orientation = yaw
            self.state = "MOVING_TO_PERSON"
            self.goal_received = True
            rospy.loginfo(f"Target Received! Person at {new_goal}. Parking at {parking_spot}.")
            
            if self.navigating:
                rospy.logwarn("Interrupting current path - replanning...")
                self.navigating = False

    def trigger_return_home(self, event=None):
        """Triggered 15s after arriving at customer"""
        if self.state != "WAITING": return
        rospy.loginfo("Wait time over. Returning Home...")
        
        try:
            tf = self.tf_buffer.lookup_transform("map", "odom", rospy.Time(0), rospy.Duration(1.0))
            home_map = do_transform_point(self.home_point_odom, tf)
            self.final_goal_pose = (home_map.point.x, home_map.point.y)
            # self.goal_orientation = 0.0
            self.goal_orientation = self.home_orientation
            
            self.state = "MOVING_HOME"
            self.goal_received = True
            self._speak("I am returning to the home position.")
        except Exception as e:
            rospy.logerr(f"Failed to find home coordinates: {e}")
            self.state = "IDLE"

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0), rospy.Duration(0.2))
            return (t.transform.translation.x, t.transform.translation.y)
        except:
            return None
            
    def check_waypoint_progression(self, event=None):
        """Topic-based smooth blending: Monitor distance and publish next waypoint early."""
        if not self.navigating or not self.current_waypoints: return
            
        robot_pos = self.get_robot_pose()
        if not robot_pos: return
            
        is_final = (self.current_waypoint_index == len(self.current_waypoints) - 1)
        wp = self.current_waypoints[self.current_waypoint_index]
        
        # 1. Handle Final Goal Arrival
        if is_final:
            dist_to_final = math.hypot(self.final_goal_pose[0] - robot_pos[0], self.final_goal_pose[1] - robot_pos[1])
            if dist_to_final <= self.goal_tolerance:
                rospy.loginfo(f"Final waypoint reached! ({dist_to_final:.2f}m away)")
                self.handle_arrival()
            return 
            
        # 2. Handle Smooth Blending
        dist_to_wp = math.hypot(wp[0] - robot_pos[0], wp[1] - robot_pos[1])
        if dist_to_wp < self.lookahead_distance:
            self.current_waypoint_index += 1
            if self.current_waypoint_index < len(self.current_waypoints):
                self.publish_next_waypoint()


    def handle_arrival(self):
        """State Machine Logic when A* Path Completes"""
        self.navigating = False
        self.current_waypoints = []
        self.status_pub.publish(String(data="goal_reached"))
        
        if self.state == "MOVING_TO_PERSON":
            rospy.loginfo("\/ ARRIVED at customer parking spot!")
            self._speak("I have arrived. How can I help you?")
            self.state = "WAITING"
            rospy.Timer(rospy.Duration(10.0), self.trigger_return_home, oneshot=True)
            
        elif self.state == "MOVING_HOME":
            rospy.loginfo("\/ Home reached.")
            self._speak("Mission complete. Awaiting next customer.")
            self.state = "IDLE"
            self.person_pose = None
            self.publish_waypoints([])
            # self.flag_pub.publish(String(data="home_reached")) # Re-enable camera
            rospy.sleep(0.5)  # Give time for perception to process
            self.flag_pub.publish(String(data="home_reached"))
            rospy.loginfo("Published 'home_reached' flag to perception node")

    def find_reachable_goal(self, final_goal):
        """Raycast toward final goal to stop at unmapped borders"""
        robot_pos = self.get_robot_pose()
        if robot_pos is None: return final_goal

        # CHANGE: Check if we're returning home with mostly-mapped route
        if self.state == "MOVING_HOME":
            # For return trip, be more optimistic about unknown space
            return final_goal  # Trust that robot can navigate back
        
        num_samples = 50
        for i in range(num_samples, 0, -1):
            t = i / num_samples
            test_x = robot_pos[0] + t * (final_goal[0] - robot_pos[0])
            test_y = robot_pos[1] + t * (final_goal[1] - robot_pos[1])
            
            grid_x = int((test_x - self.origin[0]) / self.resolution)
            grid_y = int((test_y - self.origin[1]) / self.resolution)
            
            if 0 <= grid_y < self.inflated_map.shape[0] and 0 <= grid_x < self.inflated_map.shape[1]:
                if self.inflated_map[grid_y, grid_x] == 0:
                    return (test_x, test_y)
        return (robot_pos[0] + 0.5, robot_pos[1])
    
    def try_plan(self, event=None):
        if self.inflated_map is None or not self.goal_received or self.navigating:
            return

        self.start_pose = self.get_robot_pose()
        if self.start_pose is None: return
        
        self.current_target_pose = self.find_reachable_goal(self.final_goal_pose)
        
        self.status_pub.publish(String(data="planning"))
        rospy.loginfo(f"A* Planning from ({self.start_pose[0]:.2f}, {self.start_pose[1]:.2f}) to ({self.current_target_pose[0]:.2f}, {self.current_target_pose[1]:.2f})")
        planner = AStarPlanner(self.inflated_map, self.resolution, self.origin)
        
        try:
            path = planner.plan(self.start_pose, self.current_target_pose)
        except ValueError as e:
            rospy.logwarn(f"A* planning error: {e}")
            self.goal_received = False
            return
        
        if path is None:
            rospy.logerr("No path found")
            self.goal_received = False
            return
        
        waypoints = WaypointExtractor.extract_waypoints(
            path, method=self.waypoint_method, distance_threshold=self.waypoint_distance, angle_threshold=self.waypoint_angle)
        
        rospy.loginfo(f"Path Ready: {len(waypoints)} waypoints generated")
        self.current_waypoints = waypoints
        self.current_waypoint_index = 0
        self.navigating = True
        self.last_replan_time = rospy.Time.now()
        
        self.publish_path(path)
        self.publish_waypoints(waypoints)
        
        if waypoints:
            self.publish_next_waypoint()
        self.goal_received = False
    
    def check_replan(self, event=None):
        if not self.navigating or self.final_goal_pose is None or self.current_target_pose is None: return
        robot_pos = self.get_robot_pose()
        if robot_pos is None: return
                
        dist_to_target = math.hypot(self.current_target_pose[0] - robot_pos[0], self.current_target_pose[1] - robot_pos[1])
        dist_to_final = math.hypot(self.final_goal_pose[0] - robot_pos[0], self.final_goal_pose[1] - robot_pos[1])
        dist_target_to_final = math.hypot(self.final_goal_pose[0] - self.current_target_pose[0], self.final_goal_pose[1] - self.current_target_pose[1])
        
        should_replan = False
        if dist_target_to_final > 0.5 and dist_to_target < self.replan_distance and dist_to_final > self.goal_tolerance:
            if (rospy.Time.now() - self.last_replan_time).to_sec() > 2.0:
                rospy.loginfo("Approaching mapped border. Replanning towards true final goal.")
                should_replan = True
        
        if should_replan:
            self.navigating = False
            self.goal_received = True
    
    def publish_next_waypoint(self):
        if self.current_waypoint_index >= len(self.current_waypoints): return
        
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
        
        rospy.loginfo(f"Publishing Waypoint {self.current_waypoint_index + 1}/{len(self.current_waypoints)} to /waypoint_goal")
        self.waypoint_pub.publish(msg)

    # ════════════════════════════════════════════════════════════════════
    # Publishing & Visualization
    # ════════════════════════════════════════════════════════════════════
    
    def publish_path(self, path):
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
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)
        self.waypoint_viz_pub.publish(markers)
        rospy.sleep(0.05)
        
        markers = MarkerArray()
        
        # 1. Draw the actual Customer (Massive Red Sphere)
        if self.person_pose and self.state != "MOVING_HOME":
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
            pm.scale.x = 0.4
            pm.scale.y = 0.4
            pm.scale.z = 0.4
            pm.color = ColorRGBA(1.0, 0.0, 0.0, 0.8) # Red
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

        # 2. Draw the path waypoints
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
                # The Parking Spot or Home base
                m.scale.x = m.scale.y = m.scale.z = 0.35
                m.color = ColorRGBA(1.0, 0.5, 0.0, 1.0) # Orange
            else:
                # Intermediate Path
                m.scale.x = m.scale.y = m.scale.z = 0.3
                m.color = ColorRGBA(0.0, 0.8, 1.0, 1.0) # Cyan
                
            markers.markers.append(m)
            
            # Number labels for waypoints
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
                t.text = str(i+1)
                markers.markers.append(t)
                
        self.waypoint_viz_pub.publish(markers)

if __name__ == '__main__':
    try:
        node = NavigationManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass