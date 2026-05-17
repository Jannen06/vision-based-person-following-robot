import sys
from unittest.mock import MagicMock
import os

# Set environment variables for headless operation
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Mock ultralytics and torch to avoid heavy downloads in CI
sys.modules['ultralytics'] = MagicMock()
sys.modules['ultralytics.YOLO'] = MagicMock()
sys.modules['torch'] = MagicMock()

# Mock ROS and other dependencies that aren't installed in test environment
sys.modules['rospy'] = MagicMock()
sys.modules['message_filters'] = MagicMock()
sys.modules['sensor_msgs'] = MagicMock()
sys.modules['sensor_msgs.msg'] = MagicMock()
sys.modules['geometry_msgs'] = MagicMock()
sys.modules['geometry_msgs.msg'] = MagicMock()
sys.modules['std_msgs'] = MagicMock()
sys.modules['std_msgs.msg'] = MagicMock()
sys.modules['cv_bridge'] = MagicMock()
sys.modules['tf'] = MagicMock()
sys.modules['tf.transformations'] = MagicMock()
sys.modules['tf2_ros'] = MagicMock()
sys.modules['tf2_geometry_msgs'] = MagicMock()
sys.modules['PyKDL'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()

# Create a mock YOLO class
mock_yolo = MagicMock()
sys.modules['ultralytics'].YOLO = mock_yolo

# Mock tf2_geometry_msgs functions
sys.modules['tf2_geometry_msgs'].do_transform_pose = MagicMock()
