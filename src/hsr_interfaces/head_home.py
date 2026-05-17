#!/usr/bin/env python

import rospy
from control_msgs.msg import FollowJointTrajectoryActionGoal
from trajectory_msgs.msg import JointTrajectoryPoint


def publish_head_home():
    rospy.init_node('hsr_head_home_publisher')

    pub = rospy.Publisher(
        '/hsrb/head_trajectory_controller/follow_joint_trajectory/goal',
        FollowJointTrajectoryActionGoal,
        queue_size=1
    )

    rospy.loginfo("Waiting for subscribers...")
    rospy.sleep(1.0)  # allow publisher to register

    goal_msg = FollowJointTrajectoryActionGoal()

    # Joint names
    goal_msg.goal.trajectory.joint_names = [
        'head_pan_joint',
        'head_tilt_joint'
    ]

    # Trajectory point
    point = JointTrajectoryPoint()
    point.positions = [0.0, 0.0]   # Home position
    point.velocities = [0.0, 0.0]
    point.time_from_start = rospy.Duration(2.0)

    goal_msg.goal.trajectory.points.append(point)
    goal_msg.goal.trajectory.header.stamp = rospy.Time.now()

    rospy.loginfo("Publishing head home trajectory goal...")
    pub.publish(goal_msg)

    rospy.loginfo("Goal published")


if __name__ == '__main__':
    try:
        publish_head_home()
    except rospy.ROSInterruptException:
        pass
