#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from actionlib_msgs.msg import GoalStatusArray
from std_msgs.msg import String
import tf2_ros


class GoToPersonAndReturn:
    def __init__(self):
        rospy.init_node("go_to_person_and_return")

        # --- 1. Initialize ALL variables FIRST ---
        self.person_goal = None
        self.state = "IDLE"
        self.goal_reached = False
        self.home_pose = None

        # Setup TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- 2. Capture home position with retry logic ---
        rospy.loginfo("Waiting for TF to be ready...")
        rospy.sleep(3.0)  # Give TF buffer time to fill
        self.capture_home_position()

        # --- 3. Setup publishers ---
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.flag_pub = rospy.Publisher("/flag", String, queue_size=1)

        # --- 4. Setup subscribers LAST ---
        rospy.Subscriber("/person_pose", PoseStamped, self.person_cb)
        rospy.Subscriber("/move_base/status", GoalStatusArray, self.status_cb)

        rospy.loginfo("Node ready")

    def capture_home_position(self):
        """Capture robot's current position as home position with retry logic"""
        max_retries = 5
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                rospy.loginfo(f"Attempting to capture home position (attempt {attempt + 1}/{max_retries})...")

                transform = self.tf_buffer.lookup_transform(
                    "map",
                    "base_footprint",
                    rospy.Time(0),
                    rospy.Duration(5.0)
                )

                self.home_pose = PoseStamped()
                self.home_pose.header.frame_id = "map"
                self.home_pose.header.stamp = rospy.Time.now()

                self.home_pose.pose.position.x = transform.transform.translation.x
                self.home_pose.pose.position.y = transform.transform.translation.y
                self.home_pose.pose.position.z = 0.0

                self.home_pose.pose.orientation = transform.transform.rotation

                rospy.loginfo(f"✓ Home position captured successfully: "
                              f"x={self.home_pose.pose.position.x:.2f}, "
                              f"y={self.home_pose.pose.position.y:.2f}")
                return  # Success, exit the function

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    rospy.loginfo(f"Retrying in {retry_delay} seconds...")
                    rospy.sleep(retry_delay)
                else:
                    rospy.logerr("Failed to capture home position after all retries!")
                    rospy.logerr("Node will not be able to return home.")
                    self.home_pose = None

    def person_cb(self, msg):
        if self.state == "IDLE":
            rospy.loginfo("Person pose received")
            self.person_goal = msg
            self.state = "WAIT_BEFORE_GOAL"
            rospy.Timer(rospy.Duration(0.5), self.send_person_goal, oneshot=True)

    def send_person_goal(self, event):
        if self.person_goal is None:
            return

        rospy.loginfo("Sending goal to person")
        self.person_goal.header.stamp = rospy.Time.now()
        self.goal_pub.publish(self.person_goal)

        self.goal_reached = False
        self.state = "GO_TO_PERSON"

    def status_cb(self, msg):
        if not msg.status_list:
            return

        status = msg.status_list[-1].status
        if status == 3:  # SUCCEEDED
            self.goal_reached = True
            self.handle_goal_reached()

    def handle_goal_reached(self):
        if self.state == "GO_TO_PERSON":
            rospy.loginfo("Reached person, waiting 5 seconds")
            self.state = "WAIT_AT_PERSON"
            rospy.Timer(rospy.Duration(5.0), self.send_home_goal, oneshot=True)

        elif self.state == "GO_HOME":
            rospy.loginfo("Reached home, publishing flag")

            flag_msg = String()
            flag_msg.data = "home_reached"
            self.flag_pub.publish(flag_msg)

            rospy.loginfo("Flag 'home_reached' published, going idle")
            self.state = "IDLE"

    def send_home_goal(self, event):
        if self.home_pose is None:
            rospy.logerr("Cannot return home - home position was never captured!")
            rospy.logerr("Staying at current location and going idle.")
            self.state = "IDLE"
            return

        rospy.loginfo(f"Returning home to ({self.home_pose.pose.position.x:.2f}, "
                      f"{self.home_pose.pose.position.y:.2f})")

        # Update timestamp to current time
        self.home_pose.header.stamp = rospy.Time.now()
        self.goal_pub.publish(self.home_pose)
        self.state = "GO_HOME"


if __name__ == "__main__":
    GoToPersonAndReturn()
    rospy.spin()
