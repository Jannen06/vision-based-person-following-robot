#!/usr/bin/env python3
"""
gesture_detector.py - YOLO-based Pose Estimation and Gesture Recognition

This module provides a robust detector that analyzes video frames to identify 
individuals raising their hands. It utilizes YOLO pose estimation to extract 
skeletal keypoints and applies geometric heuristics to confirm gestures.
"""

from math import atan2
from ultralytics import YOLO


class GestureDetector:
    """
    Analyzes video frames using YOLO pose estimation to detect hand-raise gestures.

    Unlike naive approaches that only check the most confident detection, this 
    class iterates through all detected individuals in a frame to locate anyone 
    who is actively waving or raising their hand.
    """

    TRIGGER_THRESHOLD = 2    # Consecutive frames the gesture must be held to confirm
    KEYPOINT_CONF_THRESH = 0.5

    def __init__(self, model_path='yolo11n-pose.pt'):
        """
        Initialize the gesture detector with a specific YOLO pose model.

        Args:
            model_path (str): The file path to the YOLO pose estimation weights 
                              (default: 'yolo11n-pose.pt').

        Returns:
            None
        """
        self.model = YOLO(model_path)
        self.raise_counter = 0
        self.is_gesture_active = False

    def process_frame(self, frame) -> dict:
        """
        Evaluate a single video frame to detect if any person is raising their hand.

        This method runs inference, extracts keypoints for up to 5 people, and 
        evaluates wrist and elbow positions relative to the shoulders. It includes 
        hysteresis (temporal smoothing) to require the gesture to be held for 
        multiple frames before confirming.

        Args:
            frame (numpy.ndarray): The BGR image frame captured from the camera.

        Returns:
            dict: A dictionary containing the detection results:
                - 'gesture_detected' (bool): True if the gesture is fully confirmed.
                - 'frames_held' (int): Consecutive frames the gesture has been seen.
                - 'debug_msg' (str): Which hand is raised (e.g., "Left Up", "Right Up").
                - 'nose_coords' (tuple): The (x, y) pixel coordinates of the person's nose.
                - 'person_orientation' (float or None): Calculated shoulder yaw angle in radians.
                - 'annotated_frame' (numpy.ndarray): The image with YOLO bounding boxes drawn.
        """
        results = self.model(
            frame,
            verbose=False,
            imgsz=640,
            conf=0.25,
            iou=0.5,
            max_det=5,
        )

        annotated_frame = results[0].plot()
        gesture_detected = False
        debug_msg = "None"
        nose_coords = (0, 0)
        person_orient = None
        is_any_hand_up = False
        selected_data = None

        n_people = results[0].boxes.shape[0]
        if n_people > 0:
            # Sort detections by confidence so we evaluate the clearest subjects first
            sorted_indices = results[0].boxes.conf.argsort(descending=True)

            for idx in sorted_indices:
                idx = idx.item()
                kpts = results[0].keypoints.data[idx]
                conf = results[0].boxes.conf[idx].item()

                # Map relevant skeletal keypoints
                nose = kpts[0]
                l_shoulder, r_shoulder = kpts[5],  kpts[6]
                l_elbow,    r_elbow = kpts[7],  kpts[8]
                l_wrist,    r_wrist = kpts[9],  kpts[10]

                # Extract vertical (y) coordinates for height comparisons
                ls_y, rs_y = float(l_shoulder[1]), float(r_shoulder[1])
                le_y, re_y = float(l_elbow[1]),    float(r_elbow[1])
                lw_y, rw_y = float(l_wrist[1]),    float(r_wrist[1])

                # Extract confidence scores for the arm joints
                lw_conf = float(l_wrist[2])
                rw_conf = float(r_wrist[2])
                le_conf = float(l_elbow[2])
                re_conf = float(r_elbow[2])

                # A hand is considered "raised" if the wrist or elbow is physically higher
                # (lower y-coordinate value) than the shoulder, provided the detection is confident.
                l_wrist_up = (lw_conf > 0.3) and (lw_y < ls_y - 10)
                l_elbow_up = (le_conf > 0.5) and (le_y < ls_y - 10)
                r_wrist_up = (rw_conf > 0.3) and (rw_y < rs_y - 10)
                r_elbow_up = (re_conf > 0.5) and (re_y < rs_y - 10)

                is_left_up = l_wrist_up or l_elbow_up
                is_right_up = r_wrist_up or r_elbow_up
                is_up = is_left_up or is_right_up

                # Estimate the person's physical orientation based on the angle between their shoulders
                orientation = None
                if l_shoulder[2] > 0.5 and r_shoulder[2] > 0.5:
                    orientation = atan2(
                        float(r_shoulder[1]) - float(l_shoulder[1]),
                        float(r_shoulder[0]) - float(l_shoulder[0]),
                    )

                # Package the data for this specific individual
                person_data = {
                    "conf":       conf,
                    "nose_coords": (int(nose[0]), int(nose[1])),
                    "debug_str":  f"P{idx}({conf:.2f}): LW={lw_y:.0f} RW={rw_y:.0f} Sh={ls_y:.0f}",
                    "is_up":      is_up,
                    "orientation": orientation,
                }

                # If this person is gesturing, they become our primary target. Stop searching.
                if is_up:
                    selected_data = person_data
                    is_any_hand_up = True
                    debug_msg = ("Left Up" if is_left_up else "") + \
                        ("Right Up" if is_right_up else "")
                    print(f">>> PERSON FOUND: {person_data['debug_str']} <<<")
                    break

                # Keep the most confident person as a fallback for debugging if no one waves
                if selected_data is None:
                    selected_data = person_data

        if selected_data:
            nose_coords = selected_data["nose_coords"]
            person_orient = selected_data["orientation"]
            if not is_any_hand_up:
                print(f" Top candidate (no gesture): {selected_data['debug_str']}")

        # Apply temporal hysteresis to prevent flickering detections caused by dropped frames
        if is_any_hand_up:
            self.raise_counter = min(self.raise_counter + 1, self.TRIGGER_THRESHOLD)
        else:
            self.raise_counter = max(self.raise_counter - 1, 0)

        # Update the active gesture state based on whether the threshold has been met
        if self.raise_counter >= self.TRIGGER_THRESHOLD:
            gesture_detected = True
            self.is_gesture_active = True
            print(f"!!! GESTURE CONFIRMED → {nose_coords} !!!")
        else:
            self.is_gesture_active = False

        return {
            "gesture_detected":  gesture_detected,
            "frames_held":       self.raise_counter,
            "debug_msg":         debug_msg,
            "nose_coords":       nose_coords,
            "person_orientation": person_orient,
            "annotated_frame":   annotated_frame,
        }
