#!/usr/bin/env python3
"""
waypoint_follower.py - Adapter between navigation_manager and behavior_loop
Receives waypoints from /waypoint_goal and republishes as /person_pose for behavior_loop.py
"""

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


class WaypointFollower:
    def __init__(self):
        rospy.init_node('waypoint_follower')

        self.current_waypoint = None
        self.goal_active = False

        # Subscribe to waypoints from navigation manager
        rospy.Subscriber('/waypoint_goal', PoseStamped, self.waypoint_cb)

        # Subscribe to flags from behavior_loop
        rospy.Subscriber('/flag', String, self.flag_cb)

        # Publish to behavior_loop as person pose
        self.person_pub = rospy.Publisher('/person_pose', PoseStamped, queue_size=1)

        # Publish completion back to navigation manager
        self.flag_pub = rospy.Publisher('/flag', String, queue_size=1)

        rospy.loginfo("Waypoint Follower ready (bridge mode)")

    def waypoint_cb(self, msg: PoseStamped):
        """Receive waypoint from navigation manager"""
        self.current_waypoint = msg
        self.goal_active = True

        rospy.loginfo(f"New waypoint received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

        # Forward to behavior_loop as person_pose
        rospy.sleep(0.5)  # Small delay to ensure behavior_loop is ready
        self.person_pub.publish(msg)
        rospy.loginfo("Waypoint forwarded to behavior_loop")

    def flag_cb(self, msg: String):
        """Listen for completion from behavior_loop"""
        if msg.data == "customer_reached" and self.goal_active:
            rospy.loginfo("Waypoint reached by behavior_loop - notifying navigation manager")
            self.goal_active = False

            # Notify navigation manager
            rospy.sleep(0.5)
            self.flag_pub.publish(String(data="waypoint_reached"))

        elif msg.data == "home_reached":
            rospy.loginfo("Home reached - resetting")
            self.goal_active = False


if __name__ == '__main__':
    try:
        node = WaypointFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
