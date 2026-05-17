#!/usr/bin/env python3
"""
behavior_loop.py - High-Level Navigation State Machine

This node acts as the executive controller for robot navigation. It manages the 
state machine for moving towards a detected person, handles stuck-detection, 
iterative goal convergence, and returning to a designated home position.
"""

import math
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist, PointStamped
from std_msgs.msg import String, ColorRGBA
from std_srvs.srv import Empty
from tf2_geometry_msgs import do_transform_point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetPlan

# Configuration Parameters
ACCEPTABLE_DIST = 1.4          # Distance threshold (meters) to consider the goal reached
CONVERGENCE_GAIN = 0.25        # Minimum improvement required to attempt getting closer
GOAL_SAMPLE_DIST = [1.0, 1.2, 1.5, 1.8, 2.2, 2.5, 3.0]
GOAL_SAMPLE_ANGLES = 16
COSTMAP_SAFE_THRESHOLD = 80
GLOBAL_TIMEOUT = 120.0         # Maximum time allowed for a single mission
STUCK_TIMEOUT = 25.0           # Time to wait before assuming the local planner is stuck
STUCK_DIST = 0.10              # Distance threshold to verify if the robot is actually moving


class BehaviorNode:
    """
    Coordinates high-level navigation tasks, interpreting goals from perception
    and managing the execution state through external navigation managers.
    """

    def __init__(self):
        """
        Initialize the BehaviorNode, setting up ROS publishers, subscribers, 
        and internal state variables.

        Args:
            None

        Returns:
            None
        """
        rospy.init_node("behavior_node")

        # Internal State
        self.state = "IDLE"
        self.person_pose = None
        self.home_point_odom = None
        self.localized = False
        self.global_costmap = None

        self._mission_start = rospy.Time(0)
        self._last_replan = rospy.Time(0)
        self._stuck_check_start = rospy.Time(0)
        self._stuck_check_pos = None
        self._ignore_status_until = rospy.Time(0)
        self._convergence_attempts = 0

        # TF Initialization
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Service Clients
        try:
            self.clear_costmaps = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        except Exception:
            self.clear_costmaps = None

        rospy.loginfo("Waiting for make_plan service...")
        rospy.wait_for_service('/move_base/make_plan', timeout=5.0)
        self.make_plan_srv = rospy.ServiceProxy('/move_base/make_plan', GetPlan)

        # Publishers
        self.goal_pub = rospy.Publisher("/goal_pose", PoseStamped, queue_size=1)
        self.vel_pub = rospy.Publisher("/hsrb/command_velocity", Twist, queue_size=1)
        self.flag_pub = rospy.Publisher("/flag", String, queue_size=10)
        self.speech_pub = rospy.Publisher("/speak", String, queue_size=5)
        self.marker_pub = rospy.Publisher("/behavior/markers", MarkerArray, queue_size=1)
        self.path_debug = rospy.Publisher("/behavior/debug_path", Path, queue_size=1)

        # Subscribers
        rospy.Subscriber("/person_pose", PoseStamped, self.person_cb)
        rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_cb)
        rospy.Subscriber("/navigation/status", String, self.nav_status_cb)

        # Main loops
        rospy.Timer(rospy.Duration(0.1), self.control_loop)
        rospy.Timer(rospy.Duration(1.0), self.publish_markers)

        self._wait_for_localization()
        rospy.loginfo("BehaviorNode initialized and ready.")
        self._speak("I am Lucy and I am ready.")

    def _wait_for_localization(self):
        """
        Block execution until the robot's initial pose is available from the TF tree.
        This ensures the home position is reliably captured before starting.

        Args:
            None

        Returns:
            None
        """
        rospy.loginfo("Waiting for transforms to initialize...")
        rate = rospy.Rate(1)
        while not rospy.is_shutdown() and not self.localized:
            try:
                self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0), rospy.Duration(0.5))
                t = self.tf_buffer.lookup_transform("odom", "base_footprint", rospy.Time(0), rospy.Duration(0.5))
                pt = PointStamped()
                pt.header.frame_id = "odom"
                pt.point = t.transform.translation
                self.home_point_odom = pt
                self.localized = True
                rospy.loginfo("Transforms successfully acquired.")
            except Exception:
                rate.sleep()

    def costmap_cb(self, msg: OccupancyGrid):
        """
        Callback to update the internal global costmap reference.

        Args:
            msg (OccupancyGrid): The latest global costmap message.

        Returns:
            None
        """
        self.global_costmap = msg

    def person_cb(self, msg: PoseStamped):
        """
        Callback to handle incoming person coordinates and initiate a new mission.

        Args:
            msg (PoseStamped): The pose representing the detected person's location.

        Returns:
            None
        """
        if self.state != "IDLE":
            return

        self.person_pose = msg
        rospy.loginfo("Target received. Initiating mission sequence.")
        self._mission_start = rospy.Time.now()
        self._convergence_attempts = 0
        self.state = "MOVING_TO_PERSON"
        self._plan_and_send_goal()

    def nav_status_cb(self, msg: String):
        """
        Process status updates from the Navigation Manager to trigger state transitions.

        Args:
            msg (String): The status string from the navigation pipeline.

        Returns:
            None
        """
        if self.state not in ["MOVING_TO_PERSON", "MOVING_HOME"]:
            return

        if rospy.Time.now() < self._ignore_status_until:
            return

        status = msg.data

        if status == "goal_reached":
            if self.state == "MOVING_TO_PERSON":
                self._handle_arrival_event()
            elif self.state == "MOVING_HOME":
                self._finish_mission()

        elif status in ["no_path", "planning_failed"]:
            rospy.logwarn(f"Navigation Manager reported: {status}. Attempting replan.")
            rospy.sleep(0.5)
            if self.state == "MOVING_TO_PERSON":
                self._plan_and_send_goal()
            elif self.state == "MOVING_HOME":
                self._return_home()

    def control_loop(self, _event):
        """
        Execute the main state machine monitoring loop at a fixed frequency.

        Args:
            _event (rospy.timer.TimerEvent): The timer event triggering this callback.

        Returns:
            None
        """
        if self.state == "MOVING_TO_PERSON":
            self._handle_moving_state()
        elif self.state == "MOVING_HOME":
            if self._dist_to_home() < 0.35:
                self._finish_mission()

    def _handle_moving_state(self):
        """
        Monitor progress toward the goal, handling distance thresholds and 
        stuck-state detection timeouts.

        Args:
            None

        Returns:
            None
        """
        pos, _ = self._robot_pose()
        if pos is None:
            return

        # Check if we are within the acceptable stopping distance
        dist = self._dist_to_person()
        if dist <= ACCEPTABLE_DIST:
            rospy.loginfo(f"Within acceptable distance ({dist:.2f}m). Arrival declared.")
            self._on_arrived()
            return

        # Monitor for stuck conditions
        if self._stuck_check_pos is None:
            self._reset_stuck_check(pos)
        else:
            if (rospy.Time.now() - self._stuck_check_start).to_sec() > STUCK_TIMEOUT:
                moved = math.hypot(pos.x - self._stuck_check_pos[0],
                                   pos.y - self._stuck_check_pos[1])
                if moved < STUCK_DIST:
                    rospy.logwarn(f"Stuck condition detected. Moved {moved:.2f}m in {STUCK_TIMEOUT}s. Replanning.")
                    if self.clear_costmaps:
                        try:
                            self.clear_costmaps()
                        except Exception:
                            pass
                    self._plan_and_send_goal()
                self._reset_stuck_check(pos)

        # Monitor overall mission timeout
        if (rospy.Time.now() - self._mission_start).to_sec() > GLOBAL_TIMEOUT:
            rospy.logwarn("Global mission timeout exceeded. Aborting and returning home.")
            self._return_home()

    def _handle_arrival_event(self):
        """
        Process the goal arrival event, attempting to iteratively converge closer 
        if the initial endpoint was too far away.

        Args:
            None

        Returns:
            None
        """
        dist = self._dist_to_person()
        rospy.loginfo(f"Goal reached. Current distance to target: {dist:.2f}m")

        if dist <= ACCEPTABLE_DIST:
            self._on_arrived()
            return

        self._convergence_attempts += 1

        if self._convergence_attempts >= 3:
            rospy.logwarn("Maximum convergence attempts reached. Concluding approach.")
            self._on_arrived()
            return

        rospy.loginfo(f"Attempting closer convergence (Attempt {self._convergence_attempts}/3).")

        if self.clear_costmaps:
            try:
                self.clear_costmaps()
            except Exception:
                pass

        rospy.sleep(1.0)

        best_goal, new_dist = self._find_best_goal_data()
        improvement = dist - new_dist

        if best_goal and improvement > CONVERGENCE_GAIN:
            rospy.loginfo(f"Closer position found: {new_dist:.2f}m (Improved by: {improvement:.2f}m)")
            self._ignore_status_until = rospy.Time.now() + rospy.Duration(1.5)
            self.goal_pub.publish(best_goal)
        else:
            rospy.logwarn(f"Unable to find a significantly better position. Best available: {new_dist:.2f}m.")
            self._on_arrived()

    def _plan_and_send_goal(self):
        """
        Calculate and publish the best valid goal pose around the target person.

        Args:
            None

        Returns:
            None
        """
        self._last_replan = rospy.Time.now()
        goal_pose, goal_dist = self._find_best_goal_data()

        if goal_pose:
            rospy.loginfo(f"Dispatching goal {goal_dist:.2f}m away from target to Navigation Manager.")
            self._ignore_status_until = rospy.Time.now() + rospy.Duration(1.5)
            self.goal_pub.publish(goal_pose)
        else:
            rospy.logwarn("Failed to find a valid goal candidate. Will retry shortly.")

    def _find_best_goal_data(self):
        """
        Evaluate sampled positions around the target coordinate to find the safest, 
        closest reachable goal.

        Args:
            None

        Returns:
            tuple: A tuple containing:
                - PoseStamped or None: The optimal, obstacle-free reachable goal pose.
                - float: The calculated Euclidean distance to that goal. Returns float('inf') if failed.
        """
        if self.person_pose is None:
            return None, float('inf')

        px = self.person_pose.pose.position.x
        py = self.person_pose.pose.position.y
        robot_pos, _ = self._robot_pose()

        if not robot_pos:
            return None, float('inf')

        candidates = []
        base_angle = math.atan2(py - robot_pos.y, px - robot_pos.x)

        # Sample concentric circles around the target
        for dist in GOAL_SAMPLE_DIST:
            for i in range(GOAL_SAMPLE_ANGLES):
                angle_offset = (2 * math.pi * i) / GOAL_SAMPLE_ANGLES
                angle = base_angle + angle_offset

                gx = px + dist * math.cos(angle)
                gy = py + dist * math.sin(angle)

                if self._is_position_free(gx, gy):
                    # Prioritize proximity to target over total travel distance
                    person_cost = dist * 10.0
                    drive_cost = math.hypot(gx - robot_pos.x, gy - robot_pos.y)
                    score = person_cost + drive_cost
                    candidates.append((score, gx, gy, dist))

        if not candidates:
            rospy.logwarn("All sampled goal positions are currently blocked by obstacles.")
            return None, float('inf')

        candidates.sort(key=lambda x: x[0])

        start = PoseStamped()
        start.header.frame_id = "map"
        start.header.stamp = rospy.Time.now()
        start.pose.position = robot_pos
        start.pose.orientation.w = 1.0

        # Validate the top candidates using the global planner
        for i, (score, gx, gy, d_person) in enumerate(candidates[:10]):
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = rospy.Time.now()
            goal.pose.position.x = gx
            goal.pose.position.y = gy
            goal.pose.position.z = 0.0

            # Orient the robot to face the target person
            yaw = math.atan2(py - gy, px - gx)
            goal.pose.orientation.z = math.sin(yaw / 2.0)
            goal.pose.orientation.w = math.cos(yaw / 2.0)

            try:
                plan = self.make_plan_srv(start, goal, 0.25)
                if plan.plan.poses and len(plan.plan.poses) > 5:
                    self.path_debug.publish(plan.plan)
                    return goal, d_person
            except Exception:
                continue

        return None, float('inf')

    def _is_position_free(self, x: float, y: float) -> bool:
        """
        Check if a given (x, y) map coordinate is free of obstacles based on the 
        global costmap threshold.

        Args:
            x (float): The x-coordinate in the map frame.
            y (float): The y-coordinate in the map frame.

        Returns:
            bool: True if the position is free, False if it is blocked or out of bounds.
        """
        if self.global_costmap is None:
            return True

        info = self.global_costmap.info
        mx = int((x - info.origin.position.x) / info.resolution)
        my = int((y - info.origin.position.y) / info.resolution)

        if not (0 <= mx < info.width and 0 <= my < info.height):
            return False

        idx = my * info.width + mx
        cost = self.global_costmap.data[idx]

        if cost == -1:
            return False
        return cost < COSTMAP_SAFE_THRESHOLD

    def _robot_pose(self):
        """
        Retrieve the current robot pose from the TF tree.

        Args:
            None

        Returns:
            tuple: A tuple containing:
                - Vector3 or None: The position component of the robot's transform.
                - int or None: 0 representing standard yaw indexing, or None if error.
        """
        try:
            t = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0))
            return t.transform.translation, 0
        except Exception:
            return None, None

    def _dist_to_person(self):
        """
        Calculate the Euclidean distance from the robot to the target person.

        Args:
            None

        Returns:
            float: Distance in meters, or float('inf') if either pose is unknown.
        """
        pos, _ = self._robot_pose()
        if not pos or not self.person_pose:
            return float('inf')
        return math.hypot(
            self.person_pose.pose.position.x - pos.x,
            self.person_pose.pose.position.y - pos.y)

    def _dist_to_home(self):
        """
        Calculate the Euclidean distance from the robot to the home point.

        Args:
            None

        Returns:
            float: Distance in meters, or float('inf') if either pose is unknown.
        """
        pos, _ = self._robot_pose()
        if not pos or not self.home_point_odom:
            return float('inf')
        try:
            tf = self.tf_buffer.lookup_transform("map", "odom", rospy.Time(0))
            home = do_transform_point(self.home_point_odom, tf)
            return math.hypot(home.point.x - pos.x, home.point.y - pos.y)
        except Exception:
            return float('inf')

    def _on_arrived(self):
        """
        Execute actions upon successfully reaching the target person.

        Args:
            None

        Returns:
            None
        """
        dist = self._dist_to_person()
        rospy.loginfo(f"Mission complete. Arrived at customer ({dist:.2f}m away).")
        self._stop()
        self._speak("I have arrived. How can I help you?")
        self.flag_pub.publish("customer_reached")
        self.state = "WAITING"
        rospy.Timer(rospy.Duration(15.0), lambda _: self._return_home(), oneshot=True)

    def _return_home(self, event=None):
        """
        Initiate navigation back to the saved home position.

        Args:
            event (rospy.timer.TimerEvent, optional): The timer event if triggered by a callback.

        Returns:
            None
        """
        if self.state == "MOVING_HOME":
            return

        rospy.loginfo("Initiating return sequence to home position.")
        self.state = "MOVING_HOME"
        try:
            tf = self.tf_buffer.lookup_transform("map", "odom", rospy.Time(0), rospy.Duration(1.0))
            home = do_transform_point(self.home_point_odom, tf)

            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = rospy.Time.now()
            goal.pose.position = home.point
            goal.pose.position.z = 0.0
            goal.pose.orientation.w = 1.0

            self._ignore_status_until = rospy.Time.now() + rospy.Duration(1.5)
            self.goal_pub.publish(goal)
        except Exception as e:
            rospy.logerr(f"Failed to calculate return home trajectory: {e}")
            self.state = "IDLE"

    def _finish_mission(self):
        """
        Reset the state machine upon completing the return trip home.

        Args:
            None

        Returns:
            None
        """
        rospy.loginfo("Successfully arrived at home position.")
        self._speak("I have returned to the home position.")
        self.flag_pub.publish("home_reached")
        self.state = "IDLE"
        self.person_pose = None
        self._convergence_attempts = 0

    def _stop(self):
        """
        Publish a zero-velocity command to immediately stop the robot.

        Args:
            None

        Returns:
            None
        """
        self.vel_pub.publish(Twist())

    def _speak(self, text: str):
        """
        Publish a string message to the text-to-speech topic.

        Args:
            text (str): The specific text snippet to vocalize.

        Returns:
            None
        """
        self.speech_pub.publish(text)
        rospy.loginfo(f"[SPEECH] {text}")

    def _reset_stuck_check(self, pos):
        """
        Reset the timer and position tracker used for stuck detection.

        Args:
            pos (Vector3): The current physical translation vector of the robot.

        Returns:
            None
        """
        if pos:
            self._stuck_check_start = rospy.Time.now()
            self._stuck_check_pos = (pos.x, pos.y)

    def publish_markers(self, _event):
        """
        Publish RViz markers for debugging and visualizing the person and goal.

        Args:
            _event (rospy.timer.TimerEvent): The timer event firing this visualization loop.

        Returns:
            None
        """
        if not self.person_pose:
            return

        markers = MarkerArray()
        now = rospy.Time.now()

        # Target Indicator Sphere
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = now
        m.ns = "person_target"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose = self.person_pose.pose
        m.pose.position.z = 0.5
        m.scale.x = 0.4
        m.scale.y = 0.4
        m.scale.z = 0.4
        m.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
        m.lifetime = rospy.Duration(2.0)
        markers.markers.append(m)

        # Information Text Label
        t = Marker()
        t.header.frame_id = "map"
        t.header.stamp = now
        t.ns = "person_label"
        t.id = 1
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose = self.person_pose.pose
        t.pose.position.z = 1.0
        t.scale.z = 0.25
        t.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        px = self.person_pose.pose.position.x
        py = self.person_pose.pose.position.y
        dist = self._dist_to_person()
        t.text = f"Person\n({px:.2f}, {py:.2f})\n{dist:.2f}m"
        t.lifetime = rospy.Duration(2.0)
        markers.markers.append(t)

        self.marker_pub.publish(markers)


if __name__ == "__main__":
    node = BehaviorNode()
    rospy.spin()
