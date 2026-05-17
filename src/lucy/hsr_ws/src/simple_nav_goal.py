#!/usr/bin/env python3
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion

"""
HSR Navigation Node
-------------------
This script sends a goal to the 'move_base' action server.
It tells the robot to move to a specific X, Y position and orientation (Theta)
in the map frame.
"""

def move_to_goal(x_target, y_target, w_orientation):
    # 1. Initialize the node
    rospy.init_node('hsr_nav_commander')

    # 2. Create an action client for move_base
    #    HSR uses the standard /move_base topic
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    rospy.loginfo("Waiting for move_base action server...")
    # Wait for the robot's navigation system to be ready
    wait = client.wait_for_server(rospy.Duration(5.0))
    if not wait:
        rospy.logerr("Action server not available! Is the robot running?")
        return

    rospy.loginfo("Connected to move_base server")

    # 3. Define the goal
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"  # We want to move relative to the map
    goal.target_pose.header.stamp = rospy.Time.now()

    # Set Position (Meters)
    goal.target_pose.pose.position.x = x_target
    goal.target_pose.pose.position.y = y_target
    
    # Set Orientation (Quaternion)
    # For simplicity, we just set the 'w' component here. 
    # w=1.0 is facing forward (0 degrees), w=0.7 is 90 degrees approx.
    # For precise angles, use tf.transformations.quaternion_from_euler
    goal.target_pose.pose.orientation.w = w_orientation
    goal.target_pose.pose.orientation.z = 0.0 # You can adjust this for rotation

    # 4. Send the goal
    rospy.loginfo(f"Sending goal: X={x_target}, Y={y_target}...")
    client.send_goal(goal)

    # 5. Wait for result
    wait = client.wait_for_result()
    
    if not wait:
        rospy.logerr("Action server not available!")
    else:
        state = client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Goal execution done!")
        else:
            rospy.loginfo(f"Goal failed with state: {state}")

if __name__ == '__main__':
    try:
        # EXAMPLE: Change these coordinates to a valid point on your map
        # You can find valid points by clicking "Publish Point" in RViz 
        # and looking at the logs.
        TARGET_X = 2.0
        TARGET_Y = 0.5
        TARGET_W = 1.0 

        move_to_goal(TARGET_X, TARGET_Y, TARGET_W)
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")