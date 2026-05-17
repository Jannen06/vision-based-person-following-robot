#!/usr/bin/env python

import unittest
import rospy
import rosnode


class TestDepthToScanLaunch(unittest.TestCase):

    def test_node_running(self):
        rospy.sleep(3)

        nodes = rosnode.get_node_names()
        self.assertIn('/depthimage_to_laserscan', nodes)

    def test_parameter_loaded(self):
        rospy.sleep(1)
        self.assertTrue(rospy.has_param('/depthimage_to_laserscan/scan_height'))
        self.assertEqual(rospy.get_param('/depthimage_to_laserscan/scan_height'), 8)


if __name__ == '__main__':
    import rostest
    rostest.rosrun('depthimage_to_laserscan',
                   'depth_to_scan_test',
                   TestDepthToScanLaunch)
