#!/usr/bin/env python3

import rospy
import math
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from tf import TransformListener
from tf.transformations import euler_from_quaternion
from std_msgs.msg import String


class FieldBasedPlanner:
    """
    A ROS-based potential field navigation planner for mobile robots.

    Steers a robot toward a goal using attractive forces (toward the goal) and
    repulsive forces (away from obstacles via LiDAR and depth sensors), with a
    built-in stuck-detection and recovery mechanism to escape local minima.
    """

    def __init__(self):
        """
        Initialise the FieldBasedPlanner node.

        Reads all ROS parameters, sets up publishers and subscribers,
        and initialises internal state variables including stuck-detection
        and recovery state.
        """
        rospy.init_node('field_based_planner')

        # --- Parameters ---
        self.ka = rospy.get_param("~ka", 0.75)
        self.kr = rospy.get_param("~kr", 4.0)
        self.p_0 = rospy.get_param("~p_0", 0.4)
        self.stop_threshold = rospy.get_param("~stop_threshold", 0.35)
        self.slowdown_radius = rospy.get_param("~slowdown_radius", 0.75)
        self.max_linear_vel = rospy.get_param("~max_linear_vel", 0.2)
        self.angular_tolerance = rospy.get_param("~angular_tolerance", 0.1)

        # --- Recovery parameters ---
        self.stuck_time_threshold = rospy.get_param("~stuck_time_threshold", 2.0)
        self.stuck_dist_threshold = rospy.get_param("~stuck_dist_threshold", 0.03)
        self.recovery_duration = rospy.get_param("~recovery_duration", 2.5)
        self.recovery_reverse_vel = rospy.get_param("~recovery_reverse_vel", -0.08)

        # --- Subscribers & Publishers ---
        self.cmd_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=10)
        rospy.Subscriber('/scan',        LaserScan,    self.update_scan)
        rospy.Subscriber('/depth_scan',  LaserScan,    self.update_depth_scan)
        rospy.Subscriber('/waypoint_goal', PoseStamped, self.goal_callback)
        self.goal_status_pub = rospy.Publisher('/flag', String, queue_size=10)
        self.tf_listener = TransformListener()

        # --- State Variables ---
        self.latest_scan = None
        self.latest_depth_scan = None
        self.robot_pose = {"x": 0, "y": 0, "theta": 0}
        self.goal_pose = None
        self.is_final_waypoint = True

        # --- Stuck-detection state ---
        self.last_progress_time = rospy.Time.now()
        self.last_progress_pos = (0.0, 0.0)
        self.is_recovering = False
        self.recovery_start_time = None
        self.recovery_direction = 1.0

        rospy.loginfo("Planner Initialized. Waiting for a 2D Nav Goal from RViz...")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def goal_callback(self, msg):
        """
        Handle an incoming goal pose from the /waypoint_goal topic.

        Extracts the (x, y, theta) target from the PoseStamped message,
        stores it as the current goal, marks this as the final waypoint,
        and resets the stuck-detection state.

        Args:
            msg (geometry_msgs/PoseStamped): The incoming goal pose message.
        """
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        orientation_q = msg.pose.orientation
        _, _, goal_theta = euler_from_quaternion([
            orientation_q.x, orientation_q.y,
            orientation_q.z, orientation_q.w
        ])
        self.goal_pose = {"x": goal_x, "y": goal_y, "theta": goal_theta}
        self.is_final_waypoint = True
        self._reset_stuck_state()
        rospy.loginfo(
            f"New goal received: x={goal_x:.2f}, y={goal_y:.2f}, "
            f"theta={math.degrees(goal_theta):.1f} deg"
        )

    def update_scan(self, msg):
        """
        Store the latest LiDAR scan message.

        Called whenever a new message arrives on the /scan topic.

        Args:
            msg (sensor_msgs/LaserScan): The incoming LiDAR scan message.
        """
        self.latest_scan = msg

    def update_depth_scan(self, msg):
        """
        Store the latest depth camera scan message.

        Called whenever a new message arrives on the /depth_scan topic.

        Args:
            msg (sensor_msgs/LaserScan): The incoming depth scan message,
                formatted as a LaserScan for compatibility.
        """
        self.latest_depth_scan = msg

    def navigate_to(self, goal_pose, is_final=False):
        """
        Programmatically set a new navigation goal.

        This is an alternative to the ROS topic callback, allowing other
        parts of the codebase to command the planner directly.

        Args:
            goal_pose (tuple): A tuple of (x, y) or (x, y, theta) representing
                the target position and optional heading in map frame.
            is_final (bool): If True, the robot will align to the goal heading
                upon arrival. Defaults to False.
        """
        self.goal_pose = {
            "x": goal_pose[0],
            "y": goal_pose[1],
            "theta": goal_pose[2] if len(goal_pose) > 2 else None
        }
        self.is_final_waypoint = is_final
        self._reset_stuck_state()
        rospy.loginfo(f"Navigating to: ({goal_pose[0]:.2f}, {goal_pose[1]:.2f})")

    # ------------------------------------------------------------------
    # TF
    # ------------------------------------------------------------------

    def get_robot_pose(self):
        """
        Look up the robot's current pose in the map frame via TF.

        Queries the transform from 'map' to 'base_footprint' and updates
        ``self.robot_pose`` with the resulting (x, y, theta).

        Returns:
            bool: True if the transform was successfully retrieved,
                  False if a TF exception occurred.
        """
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                'map', 'base_footprint', rospy.Time(0))
            _, _, yaw = euler_from_quaternion(rot)
            self.robot_pose = {"x": trans[0], "y": trans[1], "theta": yaw}
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Force calculation
    # ------------------------------------------------------------------

    def calculate_repulsive_force(self, scan_msg, current_theta):
        """
        Compute the total repulsive force vector from a single scan source.

        Iterates over all valid range readings in the scan. For each reading
        within the influence radius ``p_0``, a repulsive force is computed
        using the potential field formula and projected into world-frame x/y
        components.

        Args:
            scan_msg (sensor_msgs/LaserScan): The scan from which to compute
                repulsive forces. If None, returns zero forces.
            current_theta (float): The robot's current heading in radians,
                used to convert sensor-frame angles to world-frame angles.

        Returns:
            tuple[float, float]: The (x, y) components of the total repulsive
                force vector in the world frame.
        """
        v_rep_x = 0.0
        v_rep_y = 0.0
        if scan_msg is None:
            return v_rep_x, v_rep_y

        for i, r in enumerate(scan_msg.ranges):
            if math.isinf(r) or math.isnan(r) or r == 0.0 or r > self.p_0:
                continue
            if 0.01 < r <= self.p_0:
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                force = self.kr * (1.0 / r - 1.0 / self.p_0) / (r ** 2)
                force = min(force, 5.0)
                world_angle = current_theta + angle
                v_rep_x += -force * math.cos(world_angle)
                v_rep_y += -force * math.sin(world_angle)

        return v_rep_x, v_rep_y

    # ------------------------------------------------------------------
    # Stuck detection helpers
    # ------------------------------------------------------------------

    def _reset_stuck_state(self):
        """
        Reset all stuck-detection and recovery state variables.

        Should be called whenever a new goal is set or a recovery manoeuvre
        completes, so the planner starts fresh progress monitoring.
        """
        self.last_progress_time = rospy.Time.now()
        self.last_progress_pos = (self.robot_pose["x"], self.robot_pose["y"])
        self.is_recovering = False
        self.recovery_start_time = None

    def _check_if_stuck(self):
        """
        Determine whether the robot is stuck based on positional progress.

        Compares the robot's current position against the last recorded
        progress position. If the robot has moved more than
        ``stuck_dist_threshold`` metres, the progress timer is reset and
        the robot is considered moving. If not enough progress has been made
        within ``stuck_time_threshold`` seconds, the robot is declared stuck.

        Returns:
            bool: True if the robot is stuck, False otherwise.
        """
        x, y = self.robot_pose["x"], self.robot_pose["y"]
        dist_moved = math.hypot(x - self.last_progress_pos[0],
                                y - self.last_progress_pos[1])
        now = rospy.Time.now()

        if dist_moved > self.stuck_dist_threshold:
            self.last_progress_time = now
            self.last_progress_pos = (x, y)
            return False

        elapsed = (now - self.last_progress_time).to_sec()
        return elapsed > self.stuck_time_threshold

    def _start_recovery(self):
        """
        Initiate a recovery manoeuvre when the robot is detected as stuck.

        Sets the recovery flag, records the start time, and alternates the
        rotation direction to avoid repeating the same failed escape. A small
        random perturbation is also applied to the direction to break
        symmetric deadlocks where both directions are equally obstructed.
        """
        rospy.logwarn("Robot is STUCK — starting recovery manoeuvre.")
        self.is_recovering = True
        self.recovery_start_time = rospy.Time.now()
        # Alternate rotation direction each time to escape symmetric traps
        self.recovery_direction = self.recovery_direction * -1.0
        # Small random perturbation so we don't repeat the exact same escape
        self.recovery_direction += np.random.uniform(-0.2, 0.2)
        self.recovery_direction = math.copysign(1.0, self.recovery_direction)

    def _execute_recovery(self):
        """
        Publish recovery velocity commands for the duration of the recovery phase.

        Sends a Twist command combining a slight reverse linear velocity with
        a rotational component to help the robot escape obstacles or local
        minima. Once ``recovery_duration`` seconds have elapsed, the recovery
        flag is cleared and normal navigation resumes.
        """
        elapsed = (rospy.Time.now() - self.recovery_start_time).to_sec()
        if elapsed > self.recovery_duration:
            rospy.loginfo("Recovery manoeuvre complete — resuming normal navigation.")
            self._reset_stuck_state()
            return

        twist = Twist()
        twist.linear.x = self.recovery_reverse_vel
        twist.angular.z = self.recovery_direction * 0.4
        self.cmd_pub.publish(twist)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def planner_loop(self):
        """
        Execute one iteration of the potential field navigation loop.

        This method is called at a fixed rate (10 Hz) from the main loop.
        It performs the following steps in order:

        1. Guard checks — exits early if no goal or scan data is available,
           or if the TF lookup fails.
        2. Recovery — if a recovery manoeuvre is active, delegates to
           ``_execute_recovery()`` and returns.
        3. Phase 1 (position control) — if the robot is farther than
           ``stop_threshold`` from the goal, computes attractive and
           repulsive force vectors, adds a small random perturbation to
           break deadlocks, transforms the result into the robot frame,
           and publishes a Twist command.
        4. Phase 2 (orientation control) — once within ``stop_threshold``,
           if this is the final waypoint and a target heading is set,
           rotates in place until within ``angular_tolerance``.
        5. Stop — calls ``stop_robot()`` when the goal is fully reached.
        """
        if self.goal_pose is None or self.latest_scan is None:
            return
        if not self.get_robot_pose():
            return

        # --- Run recovery if active ---
        if self.is_recovering:
            self._execute_recovery()
            return

        x, y, theta = self.robot_pose.values()
        goal_x, goal_y, goal_theta = self.goal_pose.values()

        dx = goal_x - x
        dy = goal_y - y
        dist_to_goal = math.sqrt(dx ** 2 + dy ** 2)

        twist = Twist()

        # Phase 1: Move to goal position
        if dist_to_goal > self.stop_threshold:

            # --- Check for stuck BEFORE computing forces ---
            if self._check_if_stuck():
                self._start_recovery()
                return

            # Attractive force
            attr_force_magnitude = self.ka
            if dist_to_goal < self.slowdown_radius:
                attr_force_magnitude = self.ka * (dist_to_goal / self.slowdown_radius)

            v_attr_x = attr_force_magnitude * (dx / dist_to_goal)
            v_attr_y = attr_force_magnitude * (dy / dist_to_goal)

            # Repulsive forces
            v_rep_lidar_x, v_rep_lidar_y = self.calculate_repulsive_force(
                self.latest_scan, theta)
            v_rep_depth_x, v_rep_depth_y = self.calculate_repulsive_force(
                self.latest_depth_scan, theta)

            # Add a tiny random perturbation to break symmetric deadlocks
            perturb_x = np.random.uniform(-0.02, 0.02)
            perturb_y = np.random.uniform(-0.02, 0.02)

            vx_world = v_attr_x + v_rep_lidar_x + v_rep_depth_x + perturb_x
            vy_world = v_attr_y + v_rep_lidar_y + v_rep_depth_y + perturb_y

            # Transform to robot frame
            vx_robot = vx_world * math.cos(theta) + vy_world * math.sin(theta)

            # Allow small reverse (lower bound -0.05 instead of 0)
            twist.linear.x = np.clip(vx_robot, -0.05, self.max_linear_vel)

            desired_heading = math.atan2(vy_world, vx_world)
            angle_diff = (desired_heading - theta + math.pi) % (2 * math.pi) - math.pi
            twist.angular.z = np.clip(angle_diff * 1.5, -0.5, 0.5)

            # If linear motion is deadbanded to zero, boost angular to
            # allow the robot to rotate out rather than freeze completely
            if abs(twist.linear.x) < 0.05:
                twist.linear.x = 0.0
                twist.angular.z = np.clip(angle_diff * 2.5, -0.5, 0.5)

        # Phase 2: Final orientation alignment
        elif self.is_final_waypoint and goal_theta is not None:
            angle_diff = (goal_theta - theta + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) > self.angular_tolerance:
                twist.angular.z = np.clip(angle_diff * 1.5, -0.5, 0.5)
            else:
                self.stop_robot()
                return
        else:
            self.stop_robot()
            return

        self.cmd_pub.publish(twist)

    def stop_robot(self):
        """
        Halt the robot and clear the current goal.

        Publishes a zero Twist command to stop all motion, sets
        ``goal_pose`` to None, and logs that the goal has been reached.
        The planner then waits passively for the next goal.
        """
        self.cmd_pub.publish(Twist())
        self.goal_pose = None
        rospy.loginfo("Goal reached. Robot stopped. Waiting for next 2D Nav Goal...")


if __name__ == '__main__':
    try:
        planner = FieldBasedPlanner()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            planner.planner_loop()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
