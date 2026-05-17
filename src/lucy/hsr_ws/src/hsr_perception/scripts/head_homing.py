#!/usr/bin/env python3

import rospy
import tkinter as tk
from control_msgs.msg import FollowJointTrajectoryActionGoal
from trajectory_msgs.msg import JointTrajectoryPoint


class HeadControllerGUI:

    def __init__(self):
        rospy.init_node('hsr_head_slider_controller', anonymous=True)

        self.pub = rospy.Publisher(
            '/hsrb/head_trajectory_controller/follow_joint_trajectory/goal',
            FollowJointTrajectoryActionGoal,
            queue_size=1
        )

        # Create GUI
        self.root = tk.Tk()
        self.root.title("HSR Head Control")

        # Pan slider
        tk.Label(self.root, text="Head Pan").pack()
        self.pan_slider = tk.Scale(
            self.root,
            from_=-1.5,
            to=1.5,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=400
        )
        self.pan_slider.pack()

        # Tilt slider
        tk.Label(self.root, text="Head Tilt").pack()
        self.tilt_slider = tk.Scale(
            self.root,
            from_=-0.7,
            to=0.5,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=400
        )
        self.tilt_slider.pack()

        # Send button
        tk.Button(self.root, text="Send Command",
                  command=self.send_goal).pack(pady=10)

    def send_goal(self):
        pan = self.pan_slider.get()
        tilt = self.tilt_slider.get()

        goal_msg = FollowJointTrajectoryActionGoal()

        goal_msg.goal.trajectory.joint_names = [
            'head_pan_joint',
            'head_tilt_joint'
        ]

        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.velocities = [0.0, 0.0]
        point.time_from_start = rospy.Duration(1.0)

        goal_msg.goal.trajectory.points.append(point)
        goal_msg.goal.trajectory.header.stamp = rospy.Time.now()

        self.pub.publish(goal_msg)

        rospy.loginfo(f"Sent: pan={pan:.2f}, tilt={tilt:.2f}")

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    try:
        gui = HeadControllerGUI()
        gui.run()
    except rospy.ROSInterruptException:
        pass
