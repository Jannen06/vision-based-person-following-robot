import rospy
from sensor_msgs.msg import LaserScan
from field_planner import FieldBasedPlanner


def test_field_planner_import():
    """Verify that field_planner is found via the auto-discovered PYTHONPATH."""
    import field_planner
    assert field_planner is not None


def test_planner_initialization():
    """Test that the planner starts and initializes parameters."""
    rospy.init_node('test_planner_init', anonymous=True)
    planner = FieldBasedPlanner()
    assert hasattr(planner, 'max_linear_vel')
    assert hasattr(planner, 'ka')
    assert planner.is_recovering is False


def test_repulsive_force_logic():
    """Test the Potential Field math logic."""
    rospy.init_node('test_math_node', anonymous=True)
    planner = FieldBasedPlanner()
    planner.p_0 = 0.5  # 50cm detection threshold
    planner.kr = 4.0   # Repulsive gain
    # Create a scan with an obstacle 20cm away, directly in front (angle 0)
    scan = LaserScan()
    scan.ranges = [0.2]
    scan.angle_min = 0.0
    scan.angle_increment = 0.1
    # Calculate force at robot heading 0
    vx, vy = planner.calculate_repulsive_force(scan, 0.0)
    # The force should be negative (pushing backward)
    assert vx < 0
    assert vy == 0


def test_stuck_detection_reset():
    """Verify that the stuck state resets correctly."""
    rospy.init_node('test_stuck_node', anonymous=True)
    planner = FieldBasedPlanner()
    planner.is_recovering = True
    planner._reset_stuck_state()
    assert planner.is_recovering is False
    assert planner.recovery_start_time is None


def test_stop_robot_logic():
    """Verify that stop_robot clears the goal and stops motion."""
    rospy.init_node('test_stop_node', anonymous=True)
    planner = FieldBasedPlanner()
    planner.goal_pose = {"x": 5.0, "y": 5.0, "theta": 0.0}
    planner.stop_robot()
    assert planner.goal_pose is None
