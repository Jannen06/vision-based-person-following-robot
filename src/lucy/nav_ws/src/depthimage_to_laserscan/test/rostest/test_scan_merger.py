#!/usr/bin/env python

import unittest
import rospy
import rosnode


class TestScanMergerLaunch(unittest.TestCase):

    def test_node_running(self):
        rospy.sleep(3)

        nodes = rosnode.get_node_names()
        self.assertIn('/scan_merger', nodes)

    def test_global_parameter(self):
        rospy.sleep(1)
        self.assertTrue(rospy.has_param('/scan_merger/laserscan_topics'))

    def test_destination_topic(self):
        rospy.sleep(1)
        self.assertEqual(
            rospy.get_param('/scan_merger/scan_destination_topic'),
            '/scan'
        )


if __name__ == '__main__':
    import rostest
    rostest.rosrun('depthimage_to_laserscan',
                   'scan_merger_test',
                   TestScanMergerLaunch)
