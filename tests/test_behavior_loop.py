import glob


def test_behavior_loop_file_exists():
    """Test that behavior_loop.py file exists in the project"""
    matches = glob.glob("**/behavior_loop.py", recursive=True)
    assert len(matches) > 0, "behavior_loop.py not found in project"


def test_all_required_scripts_exist():
    """Test that all required script files exist"""
    required_scripts = [
        "behavior_loop.py",
        "perception_node.py",
        "gesture_detector.py"
    ]

    for script in required_scripts:
        matches = glob.glob(f"**/{script}", recursive=True)
        assert len(matches) > 0, f"{script} not found in project"
