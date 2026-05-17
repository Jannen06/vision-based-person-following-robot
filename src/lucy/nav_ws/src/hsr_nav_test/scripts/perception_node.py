#!/usr/bin/env python3
import numpy as np
import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import tf2_ros
from std_msgs.msg import String

# --- FIX FOR TRANSFORM ERROR ---
# We explicitly import the module AND the function to cover all ROS versions
from tf2_geometry_msgs import do_transform_pose

from gesture_detector import GestureDetector


class PersonPerceptionNode:
    def __init__(self):
        rospy.init_node('person_perception_node', anonymous=True)
        rospy.loginfo("Initializing Perception Node...")

        # --- CONFIG ---
        self.detector = GestureDetector(model_path='yolo11n-pose.pt')

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.process_every_n_frames = 3
        self.frame_count = 0

        # State
        self.camera_model = None
        self.goal_published = False
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # --- SUBSCRIBERS ---
        rgb_topic = '/hsrb/head_rgbd_sensor/rgb/image_rect_color'
        depth_topic = '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'
        info_topic = '/hsrb/head_rgbd_sensor/rgb/camera_info'

        self.info_sub = rospy.Subscriber(info_topic, CameraInfo, self.info_callback)
        self.flag_sub = rospy.Subscriber('/flag', String, self.flag_callback)

        rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)

        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self.image_callback)

        # --- PUBLISHERS ---
        self.pose_pub = rospy.Publisher('/person_pose', PoseStamped, queue_size=1)
        self.debug_pub = rospy.Publisher('/detection/debug_image', Image, queue_size=1)

        rospy.loginfo("Node Ready. Waiting for camera info...")

    def info_callback(self, msg):
        self.camera_model = msg
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.info_sub.unregister()
        rospy.loginfo(f"Camera Info Loaded. Intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}")

    def flag_callback(self, msg):
        if msg.data == "home_reached":
            self.goal_published = False
            rospy.loginfo("Resetting search. Ready for new gesture.")

    def image_callback(self, rgb_msg, depth_msg):
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        if self.camera_model is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        # --- DETECT ---
        result = self.detector.process_frame(frame)

        # Publish debug image
        if self.debug_pub.get_num_connections() > 0:
            annotated = result.get("annotated_frame", frame)
            try:
                debug_msg_out = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
                debug_msg_out.header = rgb_msg.header
                self.debug_pub.publish(debug_msg_out)
            except Exception as e:
                rospy.logerr(f"Debug Pub Error: {e}")

        # --- ACT ---
        if result["gesture_detected"] and not self.goal_published:
            self.handle_detection(result, depth_image, rgb_msg.header.frame_id)

    def handle_detection(self, result, depth_image, frame_id):
        nose_u, nose_v = result["nose_coords"]

        h, w = depth_image.shape
        if not (0 <= nose_v < h and 0 <= nose_u < w):
            return

        # Get Depth
        raw_depth = depth_image[nose_v, nose_u]

        # Handle invalid depth with 3x3 kernel
        if raw_depth == 0 or np.isnan(raw_depth):
            kernel = depth_image[max(0, nose_v-1):min(h, nose_v+2),
                                 max(0, nose_u-1):min(w, nose_u+2)]
            valid_depths = kernel[kernel > 0]
            if len(valid_depths) == 0:
                return
            raw_depth = np.median(valid_depths)

        z_meters = float(raw_depth) / 1000.0

        if 0.5 < z_meters < 8.0:  # Increased range to 8m
            rospy.loginfo(f"Gesture Validated! Range: {z_meters:.2f}m")
            self.publish_map_pose(nose_u, nose_v, z_meters, frame_id)

    def publish_map_pose(self, u, v, z, frame_id):
        # 1. Project to 3D Camera Coords
        x_cam = (u - self.cx) * z / self.fx
        y_cam = (v - self.cy) * z / self.fy
        z_cam = z

        pose_cam = PoseStamped()
        pose_cam.header.stamp = rospy.Time.now()
        pose_cam.header.frame_id = frame_id
        pose_cam.pose.position.x = x_cam
        pose_cam.pose.position.y = y_cam
        pose_cam.pose.position.z = z_cam
        pose_cam.pose.orientation.w = 1.0

        try:
            # 2. Transform Camera -> Map
            transform = self.tf_buffer.lookup_transform("map", frame_id, rospy.Time(0), rospy.Duration(1.0))

            # --- FIX: Direct function call ---
            pose_map = do_transform_pose(pose_cam, transform)

            # 3. Flatten to 2D for Navigation
            pose_map.pose.position.z = 0.0
            pose_map.pose.orientation.w = 1.0

            self.pose_pub.publish(pose_map)
            self.goal_published = True
            rospy.loginfo(f"Published Goal to Map: {pose_map.pose.position.x:.2f}, {pose_map.pose.position.y:.2f}")

        except Exception as e:
            rospy.logwarn(f"Transform Error: {e}")


if __name__ == '__main__':
    node = PersonPerceptionNode()
    rospy.spin()
