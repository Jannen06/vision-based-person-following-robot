#!/usr/bin/env python3
"""
test_navigation_workflow.py - Automated CI tests for the Navigation Manager

This test suite verifies the high-level state machine and workflow logic of 
the restaurant service robot. It aggressively mocks ROS dependencies (publishers, 
subscribers, TF listeners, and time) to test state transitions purely in Python, 
without requiring a running roscore.
"""

from navigation_manager import NavigationManager
from geometry_msgs.msg import PoseStamped, PointStamped
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# Append paths based on the directory structure to locate the modules
current_dir = os.path.dirname(os.path.abspath(__file__))
nav_scripts = os.path.join(current_dir, '../src/lucy/hsr_ws/src/hsr_navigation/scripts')
test_scripts = os.path.join(current_dir, '../src/lucy/hsr_ws/src/hsr_test/scripts')
if os.path.exists(nav_scripts):
    sys.path.insert(0, nav_scripts)
if os.path.exists(test_scripts):
    sys.path.insert(0, test_scripts)


class TestNavigationWorkflow(unittest.TestCase):
    """
    Test suite for validating the NavigationManager's state machine.
    """

    def setUp(self):
        """
        Runs before every test to set up a fresh Navigation Manager.
        """
        # Start the ROS fakes manually inside the setup
        patch('rospy.init_node').start()
        patch('rospy.Subscriber').start()
        patch('rospy.Publisher').start()
        patch('rospy.Timer').start()
        patch('tf2_ros.TransformListener').start()
        patch('tf2_ros.Buffer').start()
        patch('navigation_manager.do_transform_point').start()
        patch('rospy.Time.now', return_value=MagicMock()).start()

        # FIX 1: Explicitly mock TF math functions so they return tuples instead of empty MagicMocks
        patch('navigation_manager.euler_from_quaternion', return_value=(0.0, 0.0, 0.0)).start()
        patch('navigation_manager.quaternion_from_euler', return_value=(0.0, 0.0, 0.0, 1.0)).start()

        # FIX 2: Patch the String message class so we can reliably read the 'data' attribute
        class MockString:
            def __init__(self, data=""):
                self.data = data
        patch('navigation_manager.String', MockString).start()

        # Clean up the fakes after each test finishes
        self.addCleanup(patch.stopall)

        # Mock get_param so it returns default values without needing a ROS Parameter Server
        with patch('rospy.get_param', side_effect=[0.45, 1.0, 0.4, 'distance', 2.0, 0.4, 0.6]):
            self.nav = NavigationManager()

        # FIX 3: Mock basic map data using a real numpy array so map bounds checks work
        self.nav.inflated_map = np.zeros((100, 100), dtype=np.uint8)
        self.nav.resolution = 0.05
        self.nav.origin = (0.0, 0.0)
        self.nav.get_robot_pose = MagicMock(return_value=(0.0, 0.0))

        # Assign Publisher mocks to our class so we can spy on them
        self.nav.publish_path = MagicMock()
        self.nav.waypoint_pub = MagicMock()
        self.nav.speech_pub = MagicMock()
        self.nav.flag_out_pub = MagicMock()

        # Mock the saved home position
        self.nav.home_point_odom = PointStamped()
        self.nav.home_orientation = 0.0

    @patch('navigation_manager.AStarPlanner')
    def test_case_4_path_memory_reversibility(self, MockAStarPlanner):
        """
        Test Case 4: Reversibility
        Verifies that the robot reuses its exact outbound path backwards.
        """
        mock_planner_instance = MockAStarPlanner.return_value

        # Setup: Give the robot a complex outbound path
        complex_path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 1.0), (2.0, 2.0)]
        self.nav.outbound_path = complex_path
        self.nav.state = "TAKING_ORDER"

        # Action: Simulate receiving the 'order_taken' flag from the tablet/UI
        flag_msg = MagicMock()
        flag_msg.data = "order_taken"
        self.nav.flag_in_cb(flag_msg)

        # Assertions
        self.assertEqual(self.nav.state, "RETURNING_FROM_ORDER")
        self.assertTrue(self.nav.order_taken)
        mock_planner_instance.plan.assert_not_called()

        expected_reversed_path = list(reversed(complex_path))
        self.nav.publish_path.assert_called_once_with(expected_reversed_path)

    def test_case_2_impatient_customer_rejection(self):
        """
        Test Case 2: Busy Rejection
        Verifies that the robot ignores new navigation goals/customers if busy.
        """
        self.nav.state = "TAKING_ORDER"
        self.nav.final_goal_pose = (2.0, 2.0)

        new_goal = PoseStamped()
        new_goal.pose.position.x = 5.0
        new_goal.pose.position.y = -5.0
        self.nav.person_detected_cb(new_goal)

        self.assertEqual(self.nav.state, "TAKING_ORDER")
        self.assertEqual(self.nav.final_goal_pose, (2.0, 2.0))

    def test_person_detection_starts_workflow(self):
        """
        Verifies that detecting a person while IDLE correctly transitions 
        the robot into the 'TAKING_ORDER' state.
        """
        self.nav.state = "IDLE"

        pose_msg = PoseStamped()
        pose_msg.pose.position.x = 3.0
        pose_msg.pose.position.y = 4.0
        pose_msg.pose.orientation.w = 1.0

        self.nav.person_detected_cb(pose_msg)

        self.assertEqual(self.nav.state, "TAKING_ORDER")
        self.assertEqual(self.nav.person_pose, (3.0, 4.0))
        self.assertTrue(self.nav.goal_received)

    def test_items_ready_triggers_delivery(self):
        """
        Verifies that receiving the 'items_ready' flag transitions the robot 
        into the 'DELIVERING' state.
        """
        self.nav.state = "WAITING_FOR_ITEMS"
        self.nav.customer_parking_spot = (3.0, 4.0)
        self.nav.person_pose = (3.0, 4.5)

        flag_msg = MagicMock()
        flag_msg.data = "items_ready"
        self.nav.flag_in_cb(flag_msg)

        self.assertEqual(self.nav.state, "DELIVERING")
        self.assertTrue(self.nav.items_ready)
        self.assertTrue(self.nav.goal_received)
        self.assertEqual(self.nav.final_goal_pose, (3.0, 4.0))

    def test_planning_failure_recovery(self):
        """
        Verifies that if the A* planner fails, the robot resets to IDLE and 
        tells the Queue Manager to move on.
        """
        self.nav.state = "TAKING_ORDER"
        self.nav.order_taken = True

        self.nav._handle_planning_failure()

        self.assertEqual(self.nav.state, "IDLE")
        self.assertFalse(self.nav.order_taken)
        self.assertIsNone(self.nav.outbound_path)

        self.nav.flag_out_pub.publish.assert_called_once()
        published_msg = self.nav.flag_out_pub.publish.call_args[0][0]
        self.assertEqual(published_msg.data, "delivery_complete")

    @patch('navigation_manager.do_transform_point')
    def test_idle_timeout_triggers_return_home(self, mock_transform):
        """
        Verifies that if the robot is IDLE for more than 15 seconds, it 
        automatically transitions to RETURNING_HOME.
        """
        self.nav.state = "IDLE"
        self.nav.navigating = False

        self.nav.get_robot_pose = MagicMock(return_value=(10.0, 10.0))
        mock_home = MagicMock()
        mock_home.point.x = 0.0
        mock_home.point.y = 0.0
        mock_transform.return_value = mock_home

        now_mock = MagicMock()
        diff_mock = MagicMock()
        diff_mock.to_sec.return_value = 20.0
        now_mock.__sub__.return_value = diff_mock

        with patch('rospy.Time.now', return_value=now_mock):
            self.nav.check_idle()

        self.assertEqual(self.nav.state, "RETURNING_HOME")
        self.assertTrue(self.nav.goal_received)


if __name__ == '__main__':
    unittest.main()
