#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from actionlib_msgs.msg import GoalStatusArray


class GoToPersonAndReturn:
    def __init__(self):
        rospy.init_node("go_to_person_and_return")

        self.goal_pub = rospy.Publisher("/goal", PoseStamped, queue_size=1)

        rospy.Subscriber("/person_pose", PoseStamped, self.person_cb)
        rospy.Subscriber("/move_base/status", GoalStatusArray, self.status_cb)

        self.person_goal = None
        self.state = "IDLE"
        self.goal_reached = False

        self.home_pose = PoseStamped()
        self.home_pose.header.frame_id = "map"
        self.home_pose.pose.position.x = 0.0
        self.home_pose.pose.position.y = 0.0
        self.home_pose.pose.orientation.w = 1.0

        rospy.loginfo("Node ready")

    def person_cb(self, msg):
        if self.state == "IDLE":
            rospy.loginfo("Person pose received")
            self.person_goal = msg
            self.state = "WAIT_BEFORE_GOAL"
            rospy.Timer(rospy.Duration(5.0), self.send_person_goal, oneshot=True)

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
            rospy.loginfo("Reached home, going idle")
            self.state = "IDLE"

    def send_home_goal(self, event):
        rospy.loginfo("Returning home")
        self.home_pose.header.stamp = rospy.Time.now()
        self.goal_pub.publish(self.home_pose)
        self.state = "GO_HOME"


if __name__ == "__main__":
    GoToPersonAndReturn()
    rospy.spin()
