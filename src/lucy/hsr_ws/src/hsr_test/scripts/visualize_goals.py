#!/usr/bin/env python3
"""
visualize_goals.py — Debugging tool with Path Safety Validation
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
GOAL_SAMPLE_DIST = [1.0, 1.2, 1.5, 1.8, 2.2, 2.5]
GOAL_SAMPLE_ANGLES = 16
COSTMAP_SAFE_THRESHOLD = 60


class GoalVisualizer:
    def __init__(self):
        rospy.init_node("goal_visualizer")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.global_costmap = None
        self.person_pose = None

        rospy.loginfo("Waiting for /move_base/make_plan service...")
        rospy.wait_for_service('/move_base/make_plan')
        self.make_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)

        self.marker_pub = rospy.Publisher("/debug/goal_candidates", MarkerArray, queue_size=1, latch=True)
        self.plan_pub = rospy.Publisher("/debug/proposed_path", Path, queue_size=1, latch=True)

        rospy.Subscriber("/person_pose", PoseStamped, self.person_cb)
        rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_cb)

        rospy.loginfo("Goal Visualizer Ready.")

    def costmap_cb(self, msg):
        self.global_costmap = msg

    def person_cb(self, msg):
        rospy.loginfo(f"Received Person Pose: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        self.person_pose = msg
        self.visualize_scenarios()

    def _robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0), rospy.Duration(0.5))
            return t.transform.translation
        except Exception:
            return None

    def _get_cost(self, x, y):
        if self.global_costmap is None:
            return 0
        info = self.global_costmap.info
        mx = int((x - info.origin.position.x) / info.resolution)
        my = int((y - info.origin.position.y) / info.resolution)

        if not (0 <= mx < info.width and 0 <= my < info.height):
            return 100

        idx = my * info.width + mx
        cost = self.global_costmap.data[idx]
        if cost == -1:
            return 100
        return cost

    def _is_position_free(self, x, y):
        return self._get_cost(x, y) < COSTMAP_SAFE_THRESHOLD

    def _validate_path(self, plan):
        if not plan.plan.poses:
            return False

        for i, pose in enumerate(plan.plan.poses):
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
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.3
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color = ColorRGBA(r, g, b, a)
        m.lifetime = rospy.Duration(0)
        return m

    def visualize_scenarios(self):
        if not self.person_pose:
            return
        robot_pos = self._robot_pose()
        if not robot_pos:
            return

        px, py = self.person_pose.pose.position.x, self.person_pose.pose.position.y
        candidates = []
        markers = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        base_angle = math.atan2(robot_pos.y - py, robot_pos.x - px)
        marker_id = 0

        for dist in GOAL_SAMPLE_DIST:
            for i in range(GOAL_SAMPLE_ANGLES):
                angle_offset = (2 * math.pi * i) / GOAL_SAMPLE_ANGLES
                angle = base_angle + angle_offset

                gx = px + dist * math.cos(angle)
                gy = py + dist * math.sin(angle)

                if self._is_position_free(gx, gy):
                    markers.markers.append(
                        self.create_marker(marker_id, gx, gy, 0.0, 1.0, 0.0, 0.5, 0.1, "free")
                    )
                    person_cost = dist * 2.0
                    drive_cost = math.hypot(gx - robot_pos.x, gy - robot_pos.y)
                    score = person_cost + drive_cost
                    candidates.append((score, gx, gy, marker_id))
                else:
                    markers.markers.append(
                        self.create_marker(marker_id, gx, gy, 1.0, 0.0, 0.0, 0.8, 0.1, "blocked")
                    )

                marker_id += 1

        candidates.sort(key=lambda x: x[0])

        start = PoseStamped()
        start.header.frame_id = "map"
        start.pose.position = robot_pos
        start.pose.orientation.w = 1.0

        found_best = False
        check_limit = 20

        for i, (score, gx, gy, m_id) in enumerate(candidates[:check_limit]):
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.pose.position.x = gx
            goal.pose.position.y = gy
            goal.pose.orientation.w = 1.0

            try:
                plan_resp = self.make_plan(start, goal, 0.2)

                if plan_resp.plan.poses:
                    if self._validate_path(plan_resp):
                        if not found_best:
                            rospy.loginfo(f"✓ BEST GOAL: ({gx:.2f}, {gy:.2f})")
                            path_msg = plan_resp.plan
                            path_msg.header.frame_id = "map"
                            path_msg.header.stamp = rospy.Time.now()
                            self.plan_pub.publish(path_msg)
                            markers.markers.append(
                                self.create_marker(999, gx, gy, 0.0, 0.5, 1.0, 1.0, 0.35, "selected")
                            )
                            found_best = True
                    else:
                        markers.markers.append(
                            self.create_marker(m_id + 5000, gx, gy, 1.0, 0.5, 0.0, 1.0, 0.15, "unsafe_path")
                        )
                else:
                    markers.markers.append(
                        self.create_marker(m_id + 5000, gx, gy, 1.0, 0.6, 0.0, 1.0, 0.15, "unreachable")
                    )
            except Exception:
                pass

        self.marker_pub.publish(markers)


if __name__ == "__main__":
    GoalVisualizer()
    rospy.spin()