#!/usr/bin/env python3
"""
perception_node.py — Detects waving customer via YOLO pose and publishes map pose.

Key improvements:
  - Wider depth search kernel (5x5) for customers at range (>3 m) where
    individual pixels are often zero.
  - Publishes z=0 (floor-level) map goal — correct for 2D navigation.
  - Resets goal_published on "home_reached" so the robot can serve again.
  - Keeps depth validity check wide: 0.3 -> 10.0 m to cover restaurant distances.
"""

import numpy as np
import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs                    # F401 (registers transform types)
from tf2_geometry_msgs import do_transform_pose
from std_msgs.msg import String

from gesture_detector import GestureDetector


class PersonPerceptionNode:
    def __init__(self):
        rospy.init_node('person_perception_node', anonymous=True)
        rospy.loginfo("Initialising Perception Node...")

        # Config 
        self.detector            = GestureDetector(model_path='yolo11n-pose.pt')
        self.process_every_n     = 3          # only process every Nth frame
        self.depth_min_m         = 0.3
        self.depth_max_m         = 10.0       # restaurant could be 6–8 m across

        # State
        self.bridge           = CvBridge()
        self.tf_buffer        = tf2_ros.Buffer()
        self.tf_listener      = tf2_ros.TransformListener(self.tf_buffer)
        self.frame_count      = 0
        self.goal_published   = False
        self.fx = self.fy = self.cx = self.cy = None

        # Topics 
        rgb_topic   = '/hsrb/head_rgbd_sensor/rgb/image_rect_color'
        depth_topic = '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'
        info_topic  = '/hsrb/head_rgbd_sensor/rgb/camera_info'

        self.info_sub = rospy.Subscriber(info_topic, CameraInfo, self._info_cb)
        rospy.Subscriber('/flag', String, self._flag_cb)

        rgb_sub   = message_filters.Subscriber(rgb_topic,   Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self._image_cb)

        self.pose_pub  = rospy.Publisher('/person_pose',           PoseStamped, queue_size=1)
        self.debug_pub = rospy.Publisher('/detection/debug_image', Image,       queue_size=1)

        rospy.loginfo("Perception node ready. Waiting for camera info...")

  

    def _info_cb(self, msg: CameraInfo):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.info_sub.unregister()
        rospy.loginfo(f"Camera intrinsics loaded: fx={self.fx:.1f} fy={self.fy:.1f}")

    def _flag_cb(self, msg: String):
        if msg.data == "home_reached":
            self.goal_published = False
            rospy.loginfo("Perception reset — ready for next customer.")

    def _image_cb(self, rgb_msg: Image, depth_msg: Image):
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return
        if self.fx is None:
            return

        try:
            frame       = self.bridge.imgmsg_to_cv2(rgb_msg,   "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            rospy.logerr(f"CvBridge error: {e}")
            return

        result = self.detector.process_frame(frame)

        # Publish debug image if anyone is listening
        if self.debug_pub.get_num_connections() > 0:
            try:
                dbg = self.bridge.cv2_to_imgmsg(result.get("annotated_frame", frame), "bgr8")
                dbg.header = rgb_msg.header
                self.debug_pub.publish(dbg)
            except Exception as e:
                rospy.logwarn(f"Debug image publish failed: {e}")

        if result["gesture_detected"] and not self.goal_published:
            self._handle_detection(result, depth_image, rgb_msg.header.frame_id)

    

    def _handle_detection(self, result: dict, depth_image: np.ndarray, frame_id: str):
        u, v = result["nose_coords"]
        h, w = depth_image.shape[:2]

        if not (0 <= v < h and 0 <= u < w):
            rospy.logwarn("Nose coords out of depth image bounds — skipping.")
            return

        z_mm = self._robust_depth(depth_image, u, v, h, w)
        if z_mm is None:
            rospy.logwarn("Could not get valid depth around nose — skipping.")
            return

        z_m = z_mm / 1000.0
        if not (self.depth_min_m < z_m < self.depth_max_m):
            rospy.logwarn(f"Depth {z_m:.2f} m out of valid range — skipping.")
            return

        rospy.loginfo(f"Gesture confirmed at {z_m:.2f} m — publishing map pose.")
        self._publish_map_pose(u, v, z_m, frame_id)

    def _robust_depth(self, depth: np.ndarray, u: int, v: int,
                      h: int, w: int) -> "float | None":
        """
        Returns depth in mm at (u, v), using progressively larger kernels
        (1x1 -> 3x3 -> 5x5 -> 9x9) to handle sparse depth at range.
        Returns None if no valid pixel found.
        """
        for half in (0, 1, 2, 4):
            patch = depth[max(0, v - half): min(h, v + half + 1),
                          max(0, u - half): min(w, u + half + 1)]
            valid = patch[(patch > 0) & ~np.isnan(patch)]
            if len(valid) > 0:
                return float(np.median(valid))
        return None

    def _publish_map_pose(self, u: int, v: int, z: float, frame_id: str):
        # Back-project pixel to 3-D camera space
        x_cam = (u - self.cx) * z / self.fx
        y_cam = (v - self.cy) * z / self.fy

        pose_cam = PoseStamped()
        pose_cam.header.stamp    = rospy.Time.now()
        pose_cam.header.frame_id = frame_id
        pose_cam.pose.position.x = x_cam
        pose_cam.pose.position.y = y_cam
        pose_cam.pose.position.z = z
        pose_cam.pose.orientation.w = 1.0

        try:
            tf = self.tf_buffer.lookup_transform(
                "map", frame_id, rospy.Time(0), rospy.Duration(1.0))
            pose_map = do_transform_pose(pose_cam, tf)

            # Flatten to floor for 2-D nav planner
            pose_map.pose.position.z    = 0.0
            pose_map.pose.orientation.x = 0.0
            pose_map.pose.orientation.y = 0.0
            pose_map.pose.orientation.z = 0.0
            pose_map.pose.orientation.w = 1.0

            self.pose_pub.publish(pose_map)
            self.goal_published = True
            rospy.loginfo(
                f"Map goal: ({pose_map.pose.position.x:.2f}, "
                f"{pose_map.pose.position.y:.2f})"
            )
        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")


if __name__ == '__main__':
    node = PersonPerceptionNode()
    rospy.spin()