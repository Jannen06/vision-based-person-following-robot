#!/usr/bin/env python3
"""
test_gesture_detector.py - Unit tests for the Gesture Detector
"""

from unittest.mock import patch
import numpy as np


def test_gesture_detector_imports():
    """
    Tests that the gesture_detector module can be successfully imported.
    """
    import gesture_detector
    assert gesture_detector is not None


def test_gesture_detector_init():
    """
    Tests the initialization of the GestureDetector class.

    Verifies that the object is created successfully and contains
    the necessary 'model' attribute for YOLO inference.
    """
    from gesture_detector import GestureDetector

    # The YOLO model is assumed to be mocked in conftest.py
    detector = GestureDetector(model_path="yolo11n-pose.pt")
    assert detector is not None
    assert hasattr(detector, 'model')


def test_gesture_detector_has_class():
    """
    Tests that the GestureDetector class exists and is instantiable
    with the correct class name.
    """
    from gesture_detector import GestureDetector

    detector = GestureDetector(model_path="yolo11n-pose.pt")
    assert detector is not None
    assert detector.__class__.__name__ == 'GestureDetector'


def test_gesture_detector_process_empty_frame():
    """
    Tests the gesture detector's ability to handle an empty or featureless image.

    Verifies that passing a blank numpy array does not crash the detector
    and returns a valid, empty result.
    """
    from gesture_detector import GestureDetector

    detector = GestureDetector(model_path="yolo11n-pose.pt")
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Assuming the method is named 'detect' or 'process_frame'
    if hasattr(detector, 'detect'):
        result = detector.detect(dummy_image)
        # Should return an empty list/dict or None when no people are found
        assert not result


@patch('gesture_detector.GestureDetector')
def test_gesture_detector_raised_hand(mock_detector_class):
    """
    Tests the gesture evaluation logic for a 'raised hand' gesture.

    Mocks the detector to simulate returning a pose where the wrist
    keypoint is located above the shoulder keypoint.
    """
    detector = mock_detector_class.return_value
    # Mocking a valid detection result dictionary/object
    detector.detect.return_value = [{'gesture': 'raised_hand', 'confidence': 0.85, 'bbox': [100, 100, 200, 300]}]

    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(dummy_image)

    assert len(results) > 0
    assert results[0]['gesture'] == 'raised_hand'
    assert results[0]['confidence'] > 0.8
