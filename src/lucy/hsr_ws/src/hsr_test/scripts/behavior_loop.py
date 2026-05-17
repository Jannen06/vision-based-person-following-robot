#!/usr/bin/env python3
"""
behavior_loop.py — Obstacle-Aware Navigation with Iterative Convergence

Key fixes:
1. Increased ACCEPTABLE_DIST to 1.4m to account for local planner offsets.
2. Subscribes to /navigation/status instead of /move_base/status.
3. Removed the aggressive 5-second periodic replan so it stops interrupting A*.
"""

import math
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist, PointStamped
from actionlib_msgs.msg import GoalStatusArray
from std_msgs.msg import String, ColorRGBA
from std_srvs.srv import Empty
from tf2_geometry_msgs import do_transform_point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetPlan

# ── Tunables ────────────────────────────────────────────────────────────────
ACCEPTABLE_DIST      = 1.4    # INCREASED: Declare arrived within this distance
CONVERGENCE_GAIN     = 0.25   # Only retry if can get 0.25m+ closer
GOAL_SAMPLE_DIST     = [1.0, 1.2, 1.5, 1.8, 2.2, 2.5, 3.0]
GOAL_SAMPLE_ANGLES   = 16
COSTMAP_SAFE_THRESHOLD = 80
GLOBAL_TIMEOUT       = 120.0
STUCK_TIMEOUT        = 25.0   # INCREASED: Give colleague's local planner more time to think
STUCK_DIST           = 0.10

class BehaviorNode:

    def __init__(self):
        rospy.init_node("behavior_node")

        # ── State ────────────────────────────────────────────────────────
        self.state              = "IDLE"
        self.person_pose        = None
        self.home_point_odom    = None
        self.localized          = False
        self.global_costmap     = None
        
        self._mission_start     = rospy.Time(0)
        self._last_replan       = rospy.Time(0)
        self._stuck_check_start = rospy.Time(0)
        self._stuck_check_pos   = None
        self._ignore_status_until = rospy.Time(0)  
        self._convergence_attempts = 0             

        # ── ROS ──────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        try:
            self.clear_costmaps = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        except Exception:
            self.clear_costmaps = None
            
        rospy.loginfo("Waiting for make_plan service...")
        rospy.wait_for_service('/move_base/make_plan', timeout=5.0)
        self.make_plan_srv = rospy.ServiceProxy('/move_base/make_plan', GetPlan)

        # Send goals to Navigation Manager
        self.goal_pub    = rospy.Publisher("/goal_pose",  PoseStamped,  queue_size=1)

        self.vel_pub     = rospy.Publisher("/hsrb/command_velocity",  Twist,        queue_size=1)
        self.flag_pub    = rospy.Publisher("/flag",                   String,       queue_size=10)
        self.speech_pub  = rospy.Publisher("/speak",                  String,       queue_size=5)
        self.marker_pub  = rospy.Publisher("/behavior/markers",       MarkerArray,  queue_size=1)
        self.path_debug  = rospy.Publisher("/behavior/debug_path",    Path,         queue_size=1)

        # Subscribers
        rospy.Subscriber("/person_pose",                           PoseStamped,     self.person_cb)
        rospy.Subscriber("/move_base/global_costmap/costmap",      OccupancyGrid,   self.costmap_cb)
        
        # CHANGED: Listen to Navigation Manager instead of move_base
        rospy.Subscriber("/navigation/status",                     String,          self.nav_status_cb)

        rospy.Timer(rospy.Duration(0.1), self.control_loop)
        rospy.Timer(rospy.Duration(1.0), self.publish_markers)

        self._wait_for_localization()
        rospy.loginfo("✓ BehaviorNode ready. Iterative convergence enabled.")

        self._speak("I am Lucy and I am ready.")

    def _wait_for_localization(self):
        rospy.loginfo("Waiting for transforms...")
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
                rospy.loginfo("✓ Transforms ready.")
            except Exception:
                rate.sleep()

    # ════════════════════════════════════════════════════════════════════
    # Subscribers
    # ════════════════════════════════════════════════════════════════════

    def costmap_cb(self, msg: OccupancyGrid):
        self.global_costmap = msg

    def person_cb(self, msg: PoseStamped):
        if self.state != "IDLE":
            return
        self.person_pose = msg
        rospy.loginfo("Target received. Starting mission.")
        self._mission_start = rospy.Time.now()
        self._convergence_attempts = 0
        self.state = "MOVING_TO_PERSON"
        self._plan_and_send_goal()

    def nav_status_cb(self, msg: String):
        """Listens to the A* Navigation Manager for completion events."""
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
            rospy.logwarn(f"Navigation Manager failed ({status}). Re-planning...")
            rospy.sleep(0.5)
            if self.state == "MOVING_TO_PERSON":
                self._plan_and_send_goal()
            elif self.state == "MOVING_HOME":
                self._return_home()

    # ════════════════════════════════════════════════════════════════════
    # Control loop
    # ════════════════════════════════════════════════════════════════════

    def control_loop(self, _event):
        if self.state == "MOVING_TO_PERSON":
            self._handle_moving_state()
        elif self.state == "MOVING_HOME":
            if self._dist_to_home() < 0.35:
                self._finish_mission()

    def _handle_moving_state(self):
        pos, _ = self._robot_pose()
        if pos is None:
            return

        # 1. Check if already close enough 
        dist = self._dist_to_person()
        if dist <= ACCEPTABLE_DIST:
            rospy.loginfo(f"Already within acceptable distance ({dist:.2f}m). Declaring arrived.")
            self._on_arrived()
            return

        # (Removed the aggressive 5-second periodic replan so it stops interrupting the A* planner)

        # 2. Stuck detection
        if self._stuck_check_pos is None:
            self._reset_stuck_check(pos)
        else:
            if (rospy.Time.now() - self._stuck_check_start).to_sec() > STUCK_TIMEOUT:
                moved = math.hypot(pos.x - self._stuck_check_pos[0], 
                                   pos.y - self._stuck_check_pos[1])
                if moved < STUCK_DIST:
                    rospy.logwarn(f"Stuck! Only moved {moved:.2f}m in {STUCK_TIMEOUT}s. Re-planning.")
                    try:
                        self.clear_costmaps()
                    except:
                        pass
                    self._plan_and_send_goal()
                self._reset_stuck_check(pos)

        # 3. Global timeout
        if (rospy.Time.now() - self._mission_start).to_sec() > GLOBAL_TIMEOUT:
            rospy.logwarn("Mission timeout. Returning home.")
            self._return_home()

    def _handle_arrival_event(self):
        """Called when Navigation Manager reports goal reached."""
        dist = self._dist_to_person()
        rospy.loginfo(f"Navigation Manager goal reached. Distance to person: {dist:.2f}m")
        
        # Case 1: Close enough → finish
        if dist <= ACCEPTABLE_DIST:
            self._on_arrived()
            return

        # Case 2: Still far → try to converge closer
        self._convergence_attempts += 1
        
        # Safety limit: max 3 convergence attempts
        if self._convergence_attempts >= 3:
            rospy.logwarn("Max convergence attempts reached. Stopping here.")
            self._on_arrived()
            return

        rospy.loginfo(f"Convergence attempt {self._convergence_attempts}/3: trying to get closer...")
        
        try:
            self.clear_costmaps()
        except:
            pass
        
        rospy.sleep(1.0) 
        
        best_goal, new_dist = self._find_best_goal_data()
        
        improvement = dist - new_dist
        if best_goal and improvement > CONVERGENCE_GAIN:
            rospy.loginfo(f"Found closer position: {new_dist:.2f}m (improvement: {improvement:.2f}m)")
            self._ignore_status_until = rospy.Time.now() + rospy.Duration(1.5)
            self.goal_pub.publish(best_goal)
        else:
            rospy.logwarn(f"No significantly closer position found (best: {new_dist:.2f}m). Stopping.")
            self._on_arrived()

    # ════════════════════════════════════════════════════════════════════
    # Goal planning
    # ════════════════════════════════════════════════════════════════════

    def _plan_and_send_goal(self):
        self._last_replan = rospy.Time.now()
        goal_pose, goal_dist = self._find_best_goal_data()
        
        if goal_pose:
            rospy.loginfo(f"Sending goal at {goal_dist:.2f}m from person to Navigation Manager")
            self._ignore_status_until = rospy.Time.now() + rospy.Duration(1.5)
            self.goal_pub.publish(goal_pose)
        else:
            rospy.logwarn("No valid goal found. Will retry soon.")

    def _find_best_goal_data(self):
        """Returns (PoseStamped goal, float distance_from_person) or (None, inf)"""
        if self.person_pose is None:
            return None, float('inf')

        px = self.person_pose.pose.position.x
        py = self.person_pose.pose.position.y
        robot_pos, _ = self._robot_pose()
        
        if not robot_pos:
            return None, float('inf')
        
        candidates = []
        base_angle = math.atan2(py - robot_pos.y, px - robot_pos.x)

        # Sample goals around person at various distances/angles
        for dist in GOAL_SAMPLE_DIST:
            for i in range(GOAL_SAMPLE_ANGLES):
                angle_offset = (2 * math.pi * i) / GOAL_SAMPLE_ANGLES
                angle = base_angle + angle_offset
                
                gx = px + dist * math.cos(angle)
                gy = py + dist * math.sin(angle)

                if self._is_position_free(gx, gy):
                    # Score: prioritize closeness to person, then distance from robot
                    person_cost = dist * 10.0
                    drive_cost = math.hypot(gx - robot_pos.x, gy - robot_pos.y)
                    score = person_cost + drive_cost
                    candidates.append((score, gx, gy, dist))

        if not candidates:
            rospy.logwarn("All sampled positions blocked!")
            return None, float('inf')

        candidates.sort(key=lambda x: x[0])
        
        start = PoseStamped()
        start.header.frame_id = "map"
        start.header.stamp = rospy.Time.now()
        start.pose.position = robot_pos
        start.pose.orientation.w = 1.0

        for i, (score, gx, gy, d_person) in enumerate(candidates[:10]):
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = rospy.Time.now()
            goal.pose.position.x = gx
            goal.pose.position.y = gy
            goal.pose.position.z = 0.0
            
            yaw = math.atan2(py - gy, px - gx)
            goal.pose.orientation.z = math.sin(yaw / 2.0)
            goal.pose.orientation.w = math.cos(yaw / 2.0)

            try:
                plan = self.make_plan_srv(start, goal, 0.25)
                if plan.plan.poses and len(plan.plan.poses) > 5:
                    self.path_debug.publish(plan.plan)
                    return goal, d_person
            except Exception as e:
                continue
                
        return None, float('inf')

    def _is_position_free(self, x: float, y: float) -> bool:
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

    # ════════════════════════════════════════════════════════════════════
    # TF helpers
    # ════════════════════════════════════════════════════════════════════

    def _robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0))
            return t.transform.translation, 0
        except:
            return None, None

    def _dist_to_person(self):
        pos, _ = self._robot_pose()
        if not pos or not self.person_pose:
            return float('inf')
        return math.hypot(
            self.person_pose.pose.position.x - pos.x,
            self.person_pose.pose.position.y - pos.y)

    def _dist_to_home(self):
        pos, _ = self._robot_pose()
        if not pos or not self.home_point_odom:
            return float('inf')
        try:
            tf = self.tf_buffer.lookup_transform("map", "odom", rospy.Time(0))
            home = do_transform_point(self.home_point_odom, tf)
            return math.hypot(home.point.x - pos.x, home.point.y - pos.y)
        except:
            return float('inf')

    # ════════════════════════════════════════════════════════════════════
    # Actions
    # ════════════════════════════════════════════════════════════════════

    def _on_arrived(self):
        dist = self._dist_to_person()
        rospy.loginfo(f"✓ ARRIVED at customer ({dist:.2f}m away)!")
        self._stop()
        self._speak("I have arrived. How can I help you?")
        self.flag_pub.publish("customer_reached")
        self.state = "WAITING"
        # Reduced waiting time to 15 seconds for testing
        rospy.Timer(rospy.Duration(15.0), lambda _: self._return_home(), oneshot=True)

    def _return_home(self, event=None):
        if self.state == "MOVING_HOME":
            return
        rospy.loginfo("Returning to home position...")
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
            rospy.logerr(f"Return home failed: {e}")
            self.state = "IDLE"

    def _finish_mission(self):
        rospy.loginfo("✓ Home reached.")
        self._speak("I have returned to the home position.")
        self.flag_pub.publish("home_reached")
        self.state = "IDLE"
        self.person_pose = None
        self._convergence_attempts = 0

    def _stop(self):
        self.vel_pub.publish(Twist())

    def _speak(self, text: str):
        self.speech_pub.publish(text)
        rospy.loginfo(f"[SPEECH] {text}")

    def _reset_stuck_check(self, pos):
        if pos:
            self._stuck_check_start = rospy.Time.now()
            self._stuck_check_pos = (pos.x, pos.y)

    # ════════════════════════════════════════════════════════════════════
    # Markers
    # ════════════════════════════════════════════════════════════════════

    def publish_markers(self, _event):
        if not self.person_pose:
            return
            
        markers = MarkerArray()
        now = rospy.Time.now()
        
        # Person marker - red sphere
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = now
        m.ns = "person_target"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose = self.person_pose.pose
        m.pose.position.z = 0.5  # Raise off ground
        m.scale.x = 0.4
        m.scale.y = 0.4
        m.scale.z = 0.4
        m.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)  # red
        m.lifetime = rospy.Duration(2.0)
        markers.markers.append(m)
        
        # Text label
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
        t.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)  # white
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