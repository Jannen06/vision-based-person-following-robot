#!/usr/bin/env python3

import rospy
import math
import tf2_ros
import dynamic_reconfigure.client
from geometry_msgs.msg import PoseStamped, Twist, PointStamped
from actionlib_msgs.msg import GoalStatusArray
from std_msgs.msg import String
from std_srvs.srv import Empty
from tf2_geometry_msgs import do_transform_point


class GoToPersonAndReturn:
    def __init__(self):
        rospy.init_node("go_to_person_and_return")

        # --- Settings ---
        self.TIMEOUT = 25.0  # Shorter per-move timeout to catch spinning faster
        self.GLOBAL_TIMEOUT = 180.0
        self.FINAL_APPROACH_DIST = 1.4

        # --- Navigation Strategies ---
        # If blocked, we try shorter steps at wider angles to "feel" around the obstacle.
        self.hop_strategies = [
            {'type': 'nav', 'step': 1.0, 'angle': 0, 'desc': 'Forward Hop'},
            {'type': 'nav', 'step': 0.6, 'angle': 45, 'desc': 'Orbit Right (Short)'},
            {'type': 'nav', 'step': 0.6, 'angle': -45, 'desc': 'Orbit Left (Short)'},
            {'type': 'nav', 'step': 0.5, 'angle': 90, 'desc': 'Sidestep Right'},
            {'type': 'nav', 'step': 0.5, 'angle': -90, 'desc': 'Sidestep Left'},
        ]

        self.park_strategies = [
            {'type': 'nav', 'dist': 1.4, 'angle': 0, 'desc': 'Direct Park'},
            {'type': 'nav', 'dist': 1.5, 'angle': 30, 'desc': 'Angled Park Right'},
            {'type': 'nav', 'dist': 1.5, 'angle': -30, 'desc': 'Angled Park Left'},
        ]

        # --- State ---
        self.active_strategies = []
        self.current_strat_idx = 0
        self.mission_mode = "IDLE"
        self.person_raw_pose = None
        self.state = "IDLE"
        self.localized = False
        self.home_point_odom = None

        # Timers
        self.goal_start_time = rospy.Time(0)
        self.mission_start_time = rospy.Time(0)
        self.ignore_status_until = rospy.Time(0)

        # TF & Services
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.clear_costmaps_srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)

        # Publishers
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.vel_pub = rospy.Publisher("/hsrb/command_velocity", Twist, queue_size=1)
        self.stringpub = rospy.Publisher("/flag", String, queue_size=10)

        # Subscribers
        rospy.Subscriber("/person_pose", PoseStamped, self.person_cb)
        rospy.Subscriber("/move_base/status", GoalStatusArray, self.status_cb)

        # Safety Timer loop
        rospy.Timer(rospy.Duration(0.5), self.safety_loop)

        self.wait_for_localization()
        self.set_safe_navigation_params()

        rospy.loginfo("Behavior Node Ready. Strategy: Anti-Spin & Obstacle Flanking.")

    def set_safe_navigation_params(self):
        """Sets inflation to 0.45m - slightly tighter to allow passing furniture."""
        rospy.loginfo("DEBUG: Applying tuned inflation (0.45m)...")
        params = {'inflation_radius': 0.45, 'cost_scaling_factor': 3.0}
        try:
            all_params = rospy.get_param_names()
            potential_namespaces = set()
            for p in all_params:
                if "inflation_radius" in p:
                    ns = "/".join(p.split("/")[:-1])
                    potential_namespaces.add(ns)
            for ns in potential_namespaces:
                try:
                    client = dynamic_reconfigure.client.Client(ns, timeout=2.0)
                    client.update_configuration(params)
                except:
                    continue
        except Exception as e:
            rospy.logwarn(f"Failed to set inflation: {e}")

    def wait_for_localization(self):
        rospy.loginfo("DEBUG: Checking SLAM & ODOM...")
        rate = rospy.Rate(1)
        while not rospy.is_shutdown() and not self.localized:
            try:
                self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0), rospy.Duration(0.5))
                t = self.tf_buffer.lookup_transform("odom", "base_footprint", rospy.Time(0), rospy.Duration(0.5))
                self.home_point_odom = PointStamped()
                self.home_point_odom.header.frame_id = "odom"
                self.home_point_odom.point = t.transform.translation
                self.localized = True
                rospy.loginfo("✓ Systems Ready.")
            except:
                rate.sleep()

    def get_robot_state(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0))
            pos = t.transform.translation
            rot = t.transform.rotation
            yaw = math.atan2(2 * (rot.w * rot.z + rot.x * rot.y), 1 - 2 * (rot.y * rot.y + rot.z * rot.z))
            return pos, yaw
        except:
            return None, None

    def safety_loop(self, event):
        pos, _ = self.get_robot_state()
        if pos is None:
            if self.state not in ["IDLE", "WAITING_FOR_SLAM"]:
                rospy.logerr("CRITICAL: SLAM LOST!")
                self.vel_pub.publish(Twist())
                self.state = "WAITING_FOR_SLAM"
            return
        elif self.state == "WAITING_FOR_SLAM":
            self.state = "IDLE"
            self.return_home()
            return

        if self.state not in ["IDLE", "MOVING_HOME"] and not self.mission_start_time.is_zero():
            elapsed = (rospy.Time.now() - self.mission_start_time).to_sec()
            if elapsed > self.GLOBAL_TIMEOUT:
                rospy.logwarn("GLOBAL MISSION TIMEOUT. Aborting.")
                self.cancel_goal()
                self.return_home()

    def person_cb(self, msg):
        if self.state == "IDLE":
            rospy.loginfo("Target Pose Received.")
            self.person_raw_pose = msg
            self.mission_start_time = rospy.Time.now()
            self.decide_mission_type()

    def decide_mission_type(self):
        pos, _ = self.get_robot_state()
        if not pos:
            return
        dx = self.person_raw_pose.pose.position.x - pos.x
        dy = self.person_raw_pose.pose.position.y - pos.y
        dist = math.sqrt(dx*dx + dy*dy)

        rospy.loginfo(f"DEBUG: Distance to target: {dist:.1f}m")

        if dist > (self.FINAL_APPROACH_DIST + 0.3):
            self.mission_mode = "HOPPING"
            self.active_strategies = self.hop_strategies
        else:
            self.mission_mode = "PARKING"
            self.active_strategies = self.park_strategies

        self.current_strat_idx = 0
        self.attempt_strategy()

    def attempt_strategy(self):
        if self.current_strat_idx >= len(self.active_strategies):
            rospy.logerr("All strategies blocked. Giving up and returning home.")
            self.return_home()
            return

        strat = self.active_strategies[self.current_strat_idx]
        rospy.loginfo(f"DEBUG: Executing Strategy: {strat['desc']}")

        goal = self.calculate_nav_goal(strat)
        if not goal:
            self.current_strat_idx += 1
            self.attempt_strategy()
            return

        try:
            self.clear_costmaps_srv()
        except:
            pass

        self.goal_start_time = rospy.Time.now()
        self.ignore_status_until = rospy.Time.now() + rospy.Duration(1.0)
        self.state = "MOVING"
        self.goal_pub.publish(goal)

    def calculate_nav_goal(self, strat):
        pos, _ = self.get_robot_state()
        if not pos:
            return None
        px, py = self.person_raw_pose.pose.position.x, self.person_raw_pose.pose.position.y
        mb_goal = PoseStamped()
        mb_goal.header.frame_id = "map"
        mb_goal.header.stamp = rospy.Time.now()
        angle_to_person = math.atan2(py - pos.y, px - pos.x)

        if self.mission_mode == "HOPPING":
            angle = angle_to_person + math.radians(strat.get('angle', 0))
            mb_goal.pose.position.x = pos.x + math.cos(angle) * strat['step']
            mb_goal.pose.position.y = pos.y + math.sin(angle) * strat['step']
            yaw = angle
        else:  # PARKING
            angle = angle_to_person + math.radians(strat.get('angle', 0))
            mb_goal.pose.position.x = px - math.cos(angle) * strat['dist']
            mb_goal.pose.position.y = py - math.sin(angle) * strat['dist']
            yaw = math.atan2(py - mb_goal.pose.position.y, px - mb_goal.pose.position.x)

        mb_goal.pose.orientation.z = math.sin(yaw / 2.0)
        mb_goal.pose.orientation.w = math.cos(yaw / 2.0)
        return mb_goal

    def execute_safety_backup(self):
        """Drives robot backwards for 0.3m to unstick from inflation zones."""
        rospy.logwarn("DEBUG: Path blocked. Performing safety backup to unstick...")
        self.state = "BACKING_UP"
        cmd = Twist()
        cmd.linear.x = -0.1
        start_t = rospy.Time.now()
        rate = rospy.Rate(10)
        while (rospy.Time.now() - start_t).to_sec() < 3.0 and not rospy.is_shutdown():
            self.vel_pub.publish(cmd)
            rate.sleep()
        self.vel_pub.publish(Twist())
        rospy.sleep(0.5)

    def status_cb(self, msg):
        if not msg.status_list or rospy.Time.now() < self.ignore_status_until:
            return
        status = msg.status_list[-1].status

        if self.state == "MOVING":
            if status == 3:  # SUCCEEDED
                if self.mission_mode == "HOPPING":
                    rospy.loginfo("✓ Segment complete. Continuing...")
                    self.state = "IDLE"
                    rospy.sleep(0.5)
                    self.decide_mission_type()
                else:
                    rospy.loginfo("✓ ARRIVED. Returning Home in 5s...")
                    self.state = "WAITING"
                    rospy.Timer(rospy.Duration(5.0), self.return_home, oneshot=True)

            elif status in [4, 5, 9]:  # ABORTED or REJECTED
                rospy.logwarn(f"DEBUG: Move Blocked (Status {status}). Cancelling recovery loop...")
                self.cancel_goal()  # Immediately kill the HSR spinning behavior

                # Perform a backup to unstick
                self.execute_safety_backup()

                self.current_strat_idx += 1
                self.attempt_strategy()

        elif self.state == "MOVING_HOME":
            if status == 3:
                rospy.loginfo("✓ Home Reached.")
                self.stringpub.publish("home_reached")
                self.state = "IDLE"
                self.mission_start_time = rospy.Time(0)

    def cancel_goal(self):
        """Immediately stops the move_base action and kills velocity."""
        self.vel_pub.publish(Twist())
        p, _ = self.get_robot_state()
        if p:
            msg = PoseStamped()
            msg.header.frame_id = "map"
            msg.header.stamp = rospy.Time.now()
            msg.pose.position = p
            msg.pose.orientation.w = 1.0
            self.goal_pub.publish(msg)
            rospy.sleep(0.2)

    def return_home(self, event=None):
        if self.home_point_odom is None or self.state == "MOVING_HOME":
            return
        rospy.loginfo("Homing to physical origin...")
        try:
            transform = self.tf_buffer.lookup_transform("map", "odom", rospy.Time(0), rospy.Duration(1.0))
            home_in_map = do_transform_point(self.home_point_odom, transform)
            home_goal = PoseStamped()
            home_goal.header.frame_id = "map"
            home_goal.header.stamp = rospy.Time.now()
            home_goal.pose.position = home_in_map.point
            home_goal.pose.orientation.w = 1.0
            self.state = "MOVING_HOME"
            self.goal_start_time = rospy.Time.now()
            self.ignore_status_until = rospy.Time.now() + rospy.Duration(1.5)
            self.goal_pub.publish(home_goal)
        except Exception as e:
            rospy.logerr(f"Return home failed: {e}")
            self.state = "IDLE"


if __name__ == "__main__":
    node = GoToPersonAndReturn()
    rospy.spin()
