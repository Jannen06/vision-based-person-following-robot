#!/usr/bin/env python3
"""
person_queue_manager.py - Queue Manager for Multiple Person Detection
Maintains a queue of detected persons and sends them to navigation_manager one by one.

Flow:
  1. perception_node_updated sends to /queue_person_pose
  2. queue_manager adds to queue (logs detection position)
  3. queue_manager waits for "delivery_complete" or "home_reached" flag from navigation
  4. When ready, sends next person to /person_pose (navigation_manager)

Fixes:
  - Replaced immediate send with 1s delayed send to allow queue to accumulate
  - delivery_complete: sends next customer directly (skip home) if queue non-empty
  - home_reached: clears processed position, sends next if any
  - processed_positions cleared on planning failure reset
  - Queue position printed with index on every queue change
"""
import rospy
from collections import deque
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import String, Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class PersonQueueManager:
    def __init__(self):
        rospy.init_node('person_queue_manager')
        rospy.loginfo("Person Queue Manager: Initialized")

        # Queue configuration
        self.person_queue = deque()
        self.max_queue_size = rospy.get_param('~max_queue_size', 10)

        # State tracking
        self.navigation_ready = True
        self.current_person_id = None
        self.current_person_position = None

        # Track processed persons to avoid re-adding same person
        self.processed_positions = set()
        self.position_tolerance = 0.5

        # Duplicate detection threshold
        self.duplication_distance_threshold = rospy.get_param('~duplication_distance', 1.5)

        # Detection log
        self.detection_log = []

        # Pending send timer (to allow queue to accumulate before sending)
        self._pending_send_timer = None

        # Publishers
        self.person_pub = rospy.Publisher('/person_pose', PoseStamped, queue_size=5)
        self.status_pub = rospy.Publisher('/queue_status', String, queue_size=1)
        self.queue_viz_pub = rospy.Publisher('/queue_visualization', MarkerArray, queue_size=1, latch=True)
        self.current_customer_viz_pub = rospy.Publisher(
            '/current_customer_visualization', Marker, queue_size=1, latch=True)

        # Subscribers
        rospy.Subscriber('/queue_person_pose', PoseStamped, self.person_detected_cb)
        rospy.Subscriber('/flag_out', String, self.flag_out_cb)
        rospy.Subscriber('/navigation/status', String, self.nav_status_cb)
        rospy.Subscriber('/request_next_customer', Bool, self.request_next_customer_cb)

        rospy.Timer(rospy.Duration(1.0), self.check_queue)

        rospy.loginfo("Person Queue Manager: Ready")
        rospy.loginfo("  Input:  /queue_person_pose (from perception)")
        rospy.loginfo("  Output: /person_pose (to navigation)")
        self.publish_status()
        self.publish_visualization()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_same_position(self, pos1, pos2, tolerance=None):
        if tolerance is None:
            tolerance = self.position_tolerance
        distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
        return distance < tolerance

    def _get_position_key(self, x, y):
        return (round(x, 1), round(y, 1))

    def _print_queue(self):
        """Print the full queue with positions."""
        if len(self.person_queue) == 0:
            rospy.loginfo("  [QUEUE] Empty")
        else:
            rospy.loginfo(f"  [QUEUE] {len(self.person_queue)} person(s) waiting:")
            for i, q_msg in enumerate(self.person_queue):
                rospy.loginfo(
                    f"    [{i+1}] x={q_msg.pose.position.x:.3f}, "
                    f"y={q_msg.pose.position.y:.3f}"
                )

    # ------------------------------------------------------------------
    # Detection callback
    # ------------------------------------------------------------------

    def person_detected_cb(self, msg: PoseStamped):
        person_x = msg.pose.position.x
        person_y = msg.pose.position.y
        timestamp = rospy.get_time()
        position_key = self._get_position_key(person_x, person_y)

        # Skip if currently being served
        if position_key in self.processed_positions:
            rospy.loginfo(
                f"Person at ({person_x:.2f}, {person_y:.2f}) is currently being served - SKIPPED"
            )
            return

        rospy.loginfo("=" * 60)
        rospy.loginfo("CUSTOMER DETECTED!")
        rospy.loginfo(f"  Position: ({person_x:.3f}, {person_y:.3f})")
        rospy.loginfo(f"  Timestamp: {timestamp:.2f}")
        rospy.loginfo(f"  Current queue size: {len(self.person_queue)}")
        rospy.loginfo(f"  Currently processing: {self.current_person_id}")
        rospy.loginfo("=" * 60)

        self.detection_log.append({
            'timestamp': timestamp,
            'x': person_x,
            'y': person_y,
            'action': 'detected'
        })

        # Skip if duplicate in queue
        for queued_msg in self.person_queue:
            qx = queued_msg.pose.position.x
            qy = queued_msg.pose.position.y
            distance = ((person_x - qx)**2 + (person_y - qy)**2)**0.5
            if distance < self.duplication_distance_threshold:
                rospy.logwarn(
                    f"Person at ({person_x:.2f}, {person_y:.2f}) already in queue "
                    f"(distance: {distance:.2f}m) - SKIPPED"
                )
                return

        # Skip if already processed
        for processed_key in self.processed_positions:
            if self._is_same_position((person_x, person_y), processed_key, self.position_tolerance):
                rospy.logwarn(
                    f"Person at ({person_x:.2f}, {person_y:.2f}) already processed - SKIPPED"
                )
                return

        # Cap queue size
        if len(self.person_queue) >= self.max_queue_size:
            rospy.logwarn(f"Queue full ({self.max_queue_size}), dropping oldest person")
            dropped = self.person_queue.popleft()
            rospy.loginfo(
                f"  Dropped: ({dropped.pose.position.x:.2f}, {dropped.pose.position.y:.2f})"
            )

        self.person_queue.append(msg)
        rospy.loginfo(f"Added to queue. Current queue:")
        self._print_queue()

        self.publish_status()
        self.publish_visualization()

        # FIX: Delay send by 1s so simultaneous detections all land in the queue first
        if self.navigation_ready:
            self._schedule_send()

    def _schedule_send(self):
        """Schedule a send in 10 seconds (cancel any existing timer first).
        This window allows perception to capture all persons raising hands
        before the robot starts moving towards the first one.
        """
        if self._pending_send_timer is not None:
            try:
                self._pending_send_timer.shutdown()
            except Exception:
                pass
        rospy.loginfo("Navigation ready - waiting 10s for more hand-raises before sending...")
        self._pending_send_timer = rospy.Timer(
            rospy.Duration(6.0),
            lambda e: self.send_next_person(),
            oneshot=True
        )

    # ------------------------------------------------------------------
    # Flag callbacks
    # ------------------------------------------------------------------

    def flag_out_cb(self, msg: String):
        flag = msg.data

        if flag == "home_reached":
            rospy.loginfo("=" * 60)
            rospy.loginfo("NAVIGATION: home_reached - Workflow complete")

            # Clear processed position
            if self.current_person_position:
                self.processed_positions.discard(self.current_person_position)
                rospy.loginfo(f"  Cleared processed position: {self.current_person_position}")

            self.navigation_ready = True
            self.current_person_id = None
            self.current_person_position = None

            rospy.loginfo("=" * 60)
            self._print_queue()
            self.send_next_person()

        elif flag == "customer_reached":
            rospy.loginfo(f"NAVIGATION: customer_reached - Serving customer")
            rospy.loginfo(f"  Customers waiting in queue: {len(self.person_queue)}")
            self._print_queue()
            self.navigation_ready = False

        elif flag == "delivery_complete":
            # FIX: Go directly to next customer, skip home if queue non-empty
            rospy.loginfo("NAVIGATION: delivery_complete - Checking for next customer")
            self.navigation_ready = False

            # Clear processed position of the customer just served
            if self.current_person_position:
                self.processed_positions.discard(self.current_person_position)
                rospy.loginfo(f"  Cleared processed position: {self.current_person_position}")
            self.current_person_id = None
            self.current_person_position = None

            if len(self.person_queue) > 0:
                rospy.loginfo(f"  {len(self.person_queue)} customer(s) in queue - going directly (skipping home)")
                self._print_queue()
                self.navigation_ready = True
                self.send_next_person()
            else:
                rospy.loginfo("  No customers in queue - will return home")

        elif flag == "bar_reached":
            rospy.loginfo("NAVIGATION: bar_reached - At bar, workflow continuing")
            self.navigation_ready = False

    def nav_status_cb(self, msg: String):
        status = msg.data
        if status == "idle":
            self.navigation_ready = True
            rospy.loginfo("Navigation status: IDLE")
            self.send_next_person()

    # ------------------------------------------------------------------
    # Send next person
    # ------------------------------------------------------------------

    def send_next_person(self):
        if not self.navigation_ready:
            rospy.loginfo("Navigation not ready, waiting...")
            return

        if len(self.person_queue) == 0:
            rospy.loginfo("Queue empty - waiting for new detections")
            self.publish_visualization()
            return

        next_person = self.person_queue.popleft()
        person_x = next_person.pose.position.x
        person_y = next_person.pose.position.y
        person_key = f"{person_x:.1f}_{person_y:.1f}"
        position_key = self._get_position_key(person_x, person_y)

        self.processed_positions.add(position_key)
        self.current_person_id = person_key
        self.current_person_position = position_key
        self.navigation_ready = False

        rospy.loginfo("=" * 60)
        rospy.loginfo("SENDING CUSTOMER TO NAVIGATION")
        rospy.loginfo(f"  Position: ({person_x:.3f}, {person_y:.3f})")
        rospy.loginfo(f"  Queue remaining after send: {len(self.person_queue)}")
        self._print_queue()
        rospy.loginfo(f"  Processed positions: {self.processed_positions}")
        rospy.loginfo("=" * 60)

        self.person_pub.publish(next_person)

        self.publish_status()
        self.publish_visualization()

    def request_next_customer_cb(self, msg: Bool):
        if msg.data and len(self.person_queue) > 0:
            rospy.loginfo("Received request for next customer - sending directly")
            self.navigation_ready = True
            self.send_next_person()

    def check_queue(self, event):
        if len(self.person_queue) > 0 and self.navigation_ready:
            rospy.loginfo(f"Queue check: {len(self.person_queue)} waiting, navigation ready")
            self._print_queue()

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def publish_status(self):
        status_msg = String()
        status_msg.data = f"queue_size:{len(self.person_queue)},ready:{self.navigation_ready}"
        self.status_pub.publish(status_msg)

    def publish_visualization(self):
        marker_array = MarkerArray()

        for i, q_msg in enumerate(self.person_queue):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "queue_customers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = q_msg.pose.position.x
            marker.pose.position.y = q_msg.pose.position.y
            marker.pose.position.z = 0.5
            marker.pose.orientation = q_msg.pose.orientation
            marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.4
            marker.text = f"Queue #{i+1}"
            marker_array.markers.append(marker)

            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "queue_labels"
            text_marker.id = 100 + i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = q_msg.pose.position.x
            text_marker.pose.position.y = q_msg.pose.position.y
            text_marker.pose.position.z = 1.0
            text_marker.pose.orientation = Quaternion(0, 0, 0, 1)
            text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            text_marker.scale.z = 0.3
            text_marker.text = f"Q{i+1} ({q_msg.pose.position.x:.1f}, {q_msg.pose.position.y:.1f})"
            marker_array.markers.append(text_marker)

        self.queue_viz_pub.publish(marker_array)
        rospy.loginfo(f"Published visualization: {len(self.person_queue)} customers in queue")

    def get_queue_info(self):
        info = {
            'queue_size': len(self.person_queue),
            'navigation_ready': self.navigation_ready,
            'current_person': self.current_person_id,
            'processed_positions': list(self.processed_positions),
            'persons': [],
            'detection_log': self.detection_log
        }
        for i, msg in enumerate(self.person_queue):
            info['persons'].append({
                'index': i,
                'x': msg.pose.position.x,
                'y': msg.pose.position.y
            })
        return info


def main():
    try:
        PersonQueueManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
