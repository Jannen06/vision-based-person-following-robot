#!/usr/bin/env python3
"""
perception_node_updated.py — Detects MULTIPLE people with hand-raise gesture and publishes map poses.
Sends poses to queue_manager via separate topic (/queue_person_pose) instead of directly to navigation.

Detection is ACTIVE only when:
  - Robot is moving towards customer (TAKING_ORDER, DELIVERING)
  
Detection is INACTIVE when:
  - Robot is returning home (RETURNING_FROM_ORDER, RETURNING_FROM_DELIVERY)
  - Robot is at bar waiting (WAITING_FOR_ITEMS)
  - Robot is idle (IDLE)
"""

import numpy as np
import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs
from tf2_geometry_msgs import do_transform_pose
from std_msgs.msg import String

from gesture_detector import GestureDetector


class MultiPersonPerceptionNode:
    def __init__(self):
        rospy.init_node('person_perception_node', anonymous=True)
        rospy.loginfo("Initialising Multi-Person Perception Node with Gesture Detection...")

        # Config 
        self.detector = GestureDetector(model_path='yolo11n-pose.pt')
        self.process_every_n = rospy.get_param('~process_every_n', 3)
        self.depth_min_m = rospy.get_param('~depth_min_m', 0.3)
        self.depth_max_m = rospy.get_param('~depth_max_m', 10.0)

        # State
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.frame_count = 0
        self.fx = self.fy = self.cx = self.cy = None
        
        # Detection state - controls when detection is active
        # Active states: TAKING_ORDER, DELIVERING
        # Inactive states: IDLE, RETURNING_FROM_ORDER, WAITING_FOR_ITEMS, RETURNING_FROM_DELIVERY
        self.detection_active = True  # Start active for initial customer detection
        self.current_navigation_state = "IDLE"

        # Topics 
        rgb_topic = '/hsrb/head_rgbd_sensor/rgb/image_rect_color'
        depth_topic = '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'
        info_topic = '/hsrb/head_rgbd_sensor/rgb/camera_info'

        self.info_sub = rospy.Subscriber(info_topic, CameraInfo, self._info_cb)
        
        # Subscribe to flag to track navigation state
        # - customer_reached → TAKING_ORDER started (detection can continue for next customer)
        # - bar_reached → at bar, going to WAITING_FOR_ITEMS (detection inactive)
        # - home_reached → back at home, going to IDLE (detection active for next customer)
        # - delivery_complete → delivered, going to RETURNING_FROM_DELIVERY (detection inactive)
        rospy.Subscriber('/flag_out', String, self._flag_out_cb)
        rospy.Subscriber('/flag', String, self._flag_cb)

        rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self._image_cb)

        # Publishers - send to QUEUE MANAGER only (not directly to navigation)
        self.queue_pose_pub = rospy.Publisher('/queue_person_pose', PoseStamped, queue_size=10)
        self.debug_pub = rospy.Publisher('/detection/debug_image', Image, queue_size=1)
        self.count_pub = rospy.Publisher('/detection/person_count', String, queue_size=1)
        self.status_pub = rospy.Publisher('/perception_status', String, queue_size=1)

        rospy.loginfo("Multi-Person Perception node ready. Waiting for camera info...")

    def _info_cb(self, msg: CameraInfo):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.info_sub.unregister()
        rospy.loginfo(f"Camera intrinsics loaded: fx={self.fx:.1f} fy={self.fy:.1f}")

    def _flag_out_cb(self, msg: String):
        """Track navigation state to enable/disable detection appropriately"""
        flag = msg.data
        
        if flag == "customer_reached":
            # Arrived at customer - stays in TAKING_ORDER state, detection stays active
            # This allows detecting next customer while serving current one
            self.current_navigation_state = "TAKING_ORDER"
            self.detection_active = True
            rospy.loginfo("Navigation: customer_reached - Detection ACTIVE (can detect next customer)")
            
        elif flag == "bar_reached":
            # Arrived at bar - going to WAITING_FOR_ITEMS state
            # Detection should be inactive (robot is stationary waiting, not looking for customers)
            self.current_navigation_state = "WAITING_FOR_ITEMS"
            self.detection_active = False
            rospy.loginfo("Navigation: bar_reached - Detection INACTIVE (waiting at bar)")
            
        elif flag == "home_reached":
            # Returned home - going to IDLE state
            # Detection should be active (ready for next customer)
            self.current_navigation_state = "IDLE"
            self.detection_active = True
            self.detector.raise_counter = 0
            self.detector.is_gesture_active = False
            rospy.loginfo("Navigation: home_reached - Detection ACTIVE (ready for next customer)")
            
        elif flag == "delivery_complete":
            # Delivered items - going to RETURNING_FROM_DELIVERY
            # Detection should be inactive (robot is returning home)
            self.current_navigation_state = "RETURNING_FROM_DELIVERY"
            self.detection_active = False
            rospy.loginfo("Navigation: delivery_complete - Detection INACTIVE (returning home)")
        
        self._publish_status()

    def _flag_cb(self, msg: String):
        if msg.data == "home_reached":
            # Also handle direct flag messages
            self.detection_active = True
            self.current_navigation_state = "IDLE"
            self.detector.raise_counter = 0
            self.detector.is_gesture_active = False
            rospy.loginfo("Perception reset — ready for next customers.")
            self._publish_status()

    def _image_cb(self, rgb_msg: Image, depth_msg: Image):
        # Skip processing if detection is not active
        if not self.detection_active:
            return
            
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return
        if self.fx is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            rospy.logerr(f"CvBridge error: {e}")
            return

        # Use GestureDetector to process frame (handles gesture confirmation logic)
        result = self.detector.process_frame(frame)

        # Publish debug image
        if self.debug_pub.get_num_connections() > 0:
            try:
                dbg = self.bridge.cv2_to_imgmsg(result.get("annotated_frame", frame), "bgr8")
                dbg.header = rgb_msg.header
                self.debug_pub.publish(dbg)
            except Exception as e:
                rospy.logwarn(f"Debug image publish failed: {e}")

        # Only process when gesture is CONFIRMED
        if result["gesture_detected"]:
            rospy.loginfo(f"Gesture confirmed! Frames held: {result['frames_held']}")
            self._detect_and_publish_gesturing_persons(frame, depth_image, rgb_msg.header.frame_id)

    def _detect_and_publish_gesturing_persons(self, frame, depth_image, frame_id):
        """Detect ALL persons with hand-raise gesture and publish their poses to queue"""
        
        # Run YOLO detection ourselves to get all persons
        results = self.detector.model(
            frame,
            verbose=False,
            imgsz=640,
            conf=0.25,
            iou=0.5,
            max_det=5,
        )

        gesturing_persons = []
        
        n_people = results[0].boxes.shape[0]
        
        if n_people == 0:
            return

        # Check each detected person for hand-raise gesture
        for idx in range(n_people):
            try:
                kpts = results[0].keypoints.data[idx]
                conf = results[0].boxes.conf[idx].item()
                
                # Get keypoints
                nose = kpts[0]
                l_shoulder, r_shoulder = kpts[5], kpts[6]
                l_elbow, r_elbow = kpts[7], kpts[8]
                l_wrist, r_wrist = kpts[9], kpts[10]

                # Check for hand raise gesture (same logic as GestureDetector)
                lw_y, rw_y = float(l_wrist[1]), float(r_wrist[1])
                ls_y, rs_y = float(l_shoulder[1]), float(r_shoulder[1])
                le_y, re_y = float(l_elbow[1]), float(r_elbow[1])

                lw_conf, rw_conf = float(l_wrist[2]), float(r_wrist[2])
                le_conf, re_conf = float(l_elbow[2]), float(r_elbow[2])

                l_wrist_up = (lw_conf > 0.3) and (lw_y < ls_y - 10)
                l_elbow_up = (le_conf > 0.5) and (le_y < ls_y - 10)
                r_wrist_up = (rw_conf > 0.3) and (rw_y < rs_y - 10)
                r_elbow_up = (re_conf > 0.5) and (re_y < rs_y - 10)

                is_up = l_wrist_up or l_elbow_up or r_wrist_up or r_elbow_up
                
                # Only process persons with hand raise
                if not is_up:
                    continue

                u = int(nose[0])
                v = int(nose[1])
                
                h, w = depth_image.shape[:2]
                
                if not (0 <= v < h and 0 <= u < w):
                    continue

                # Get depth
                z_mm = self._robust_depth(depth_image, u, v, h, w)
                if z_mm is None:
                    continue

                z_m = z_mm / 1000.0
                if not (self.depth_min_m < z_m < self.depth_max_m):
                    continue

                # Calculate orientation
                orientation = None
                if l_shoulder[2] > 0.5 and r_shoulder[2] > 0.5:
                    from math import atan2
                    orientation = atan2(
                        float(r_shoulder[1]) - float(l_shoulder[1]),
                        float(r_shoulder[0]) - float(l_shoulder[0]),
                    )

                # Create map pose
                pose_map = self._create_map_pose(u, v, z_m, frame_id, orientation)
                
                if pose_map:
                    gesturing_persons.append({
                        'idx': idx,
                        'conf': conf,
                        'pose': pose_map
                    })
                    
                    # Publish to QUEUE MANAGER (not directly to navigation)
                    self.queue_pose_pub.publish(pose_map)
                    rospy.loginfo(
                        f"Gesture Person {idx}: Sent to queue ({pose_map.pose.position.x:.2f}, "
                        f"{pose_map.pose.position.y:.2f}), conf={conf:.2f}"
                    )
                    
            except Exception as e:
                rospy.logerr(f"Error processing person {idx}: {e}")
                continue

        # Publish count
        self.count_pub.publish(String(data=f"gesturing_persons:{len(gesturing_persons)}"))
        rospy.loginfo(f"Sent {len(gesturing_persons)} gesturing person(s) to queue")

    def _robust_depth(self, depth: np.ndarray, u: int, v: int,
                      h: int, w: int) -> "float | None":
        """Returns depth using progressively larger kernels"""
        for half in (0, 1, 2, 4):
            patch = depth[max(0, v - half): min(h, v + half + 1),
                          max(0, u - half): min(w, u + half + 1)]
            valid = patch[(patch > 0) & ~np.isnan(patch)]
            if len(valid) > 0:
                return float(np.median(valid))
        return None

    def _create_map_pose(self, u: int, v: int, z: float, frame_id: str, orientation=None) -> "PoseStamped | None":
        """Convert pixel coordinates to map frame pose"""
        x_cam = (u - self.cx) * z / self.fx
        y_cam = (v - self.cy) * z / self.fy

        pose_cam = PoseStamped()
        pose_cam.header.stamp = rospy.Time.now()
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
            pose_map.pose.position.z = 0.0
            
            # Set orientation if available
            if orientation is not None:
                import tf
                quat = tf.transformations.quaternion_from_euler(0, 0, orientation)
                pose_map.pose.orientation.x = quat[0]
                pose_map.pose.orientation.y = quat[1]
                pose_map.pose.orientation.z = quat[2]
                pose_map.pose.orientation.w = quat[3]
            else:
                pose_map.pose.orientation.x = 0.0
                pose_map.pose.orientation.y = 0.0
                pose_map.pose.orientation.z = 0.0
                pose_map.pose.orientation.w = 1.0

            return pose_map
            
        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return None

    def _publish_status(self):
        """Publish perception status"""
        status_msg = String()
        status_msg.data = f"state:{self.current_navigation_state},active:{self.detection_active}"
        self.status_pub.publish(status_msg)


if __name__ == '__main__':
    node = MultiPersonPerceptionNode()
    rospy.spin()