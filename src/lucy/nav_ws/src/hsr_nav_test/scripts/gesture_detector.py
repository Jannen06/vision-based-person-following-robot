from math import atan2
from ultralytics import YOLO


class GestureDetector:
    """Detects gestures from video frames using YOLO pose estimation."""

    # 2 Frames is enough to confirm intent without being glitchy
    TRIGGER_THRESHOLD = 2

    KEYPOINT_CONF_THRESH = 0.5

    def __init__(self, model_path='yolo11n-pose.pt'):
        self.model = YOLO(model_path)
        self.raise_counter = 0
        self.is_gesture_active = False

    def process_frame(self, frame):
        """
        Process frame. 
        CRITICAL CHANGE: Now iterates through ALL people to find the waver,
        instead of just checking the person with the highest confidence.
        """
        results = self.model(
            frame,
            verbose=False,
            imgsz=640,
            conf=0.25,
            iou=0.5,
            max_det=5  # Allow detecting up to 5 people
        )

        gesture_detected = False
        debug_msg = "None"
        nose_coords = (0, 0)
        person_orientation = None
        annotated_frame = results[0].plot()

        # We need to find the "Best Candidate".
        # Priority 1: Someone raising their hand.
        # Priority 2: If no one is raising hand, the person with highest confidence (for debug).

        selected_data = None
        is_any_hand_up = False

        if results[0].boxes.shape[0] > 0:
            # Sort people by confidence (descending) so we check best detections first
            # But we won't stop until we find a hand OR run out of people
            sorted_indices = results[0].boxes.conf.argsort(descending=True)

            for idx in sorted_indices:
                idx = idx.item()  # Convert tensor to int

                # --- EXTRACT DATA FOR THIS PERSON ---
                kpts = results[0].keypoints.data[idx]
                conf = results[0].boxes.conf[idx].item()

                # Map Keypoints
                nose = kpts[0]
                l_shoulder, r_shoulder = kpts[5], kpts[6]
                l_elbow, r_elbow = kpts[7], kpts[8]
                l_wrist, r_wrist = kpts[9], kpts[10]

                # Coordinates
                float(nose[1])
                ls_y, rs_y = float(l_shoulder[1]), float(r_shoulder[1])
                le_y, re_y = float(l_elbow[1]), float(r_elbow[1])
                lw_y, rw_y = float(l_wrist[1]), float(r_wrist[1])

                # Confidences
                lw_conf, rw_conf = float(l_wrist[2]), float(r_wrist[2])
                le_conf, re_conf = float(l_elbow[2]), float(r_elbow[2])

                # --- CHECK GESTURE ---
                # Check Left
                l_wrist_up = (lw_conf > 0.3) and (lw_y < ls_y - 10)
                l_elbow_up = (le_conf > 0.5) and (le_y < ls_y - 10)
                is_left_up = l_wrist_up or l_elbow_up

                # Check Right
                r_wrist_up = (rw_conf > 0.3) and (rw_y < rs_y - 10)
                r_elbow_up = (re_conf > 0.5) and (re_y < rs_y - 10)
                is_right_up = r_wrist_up or r_elbow_up

                current_is_up = is_left_up or is_right_up

                # Store this person's data
                person_data = {
                    "conf": conf,
                    "nose_coords": (int(nose[0]), int(nose[1])),
                    "debug_str": f"P{idx}({conf:.2f}): L_W={lw_y:.0f} R_W={rw_y:.0f} Shldr={ls_y:.0f}",
                    "is_up": current_is_up,
                    "orientation": None
                }

                # Calc Orientation
                if l_shoulder[2] > 0.5 and r_shoulder[2] > 0.5:
                    person_data["orientation"] = atan2(
                        float(r_shoulder[1]) - float(l_shoulder[1]),
                        float(r_shoulder[0]) - float(l_shoulder[0])
                    )

                # LOGIC:
                # If this person is waving, THEY are the winner. Stop searching.
                if current_is_up:
                    selected_data = person_data
                    is_any_hand_up = True
                    if is_left_up:
                        debug_msg = "Left Up"
                    if is_right_up:
                        debug_msg = "Right Up"
                    print(f">>> FOUND WAVER: {person_data['debug_str']} <<<")
                    break

                # If this is the very first (highest conf) person, store them as fallback
                if selected_data is None:
                    selected_data = person_data

        # --- PROCESS WINNER ---
        if selected_data:
            nose_coords = selected_data["nose_coords"]
            person_orientation = selected_data["orientation"]

            # Print debug for the selected person (waver or fallback)
            if not is_any_hand_up:
                print(f"--- Top Candidate (No Gesture): {selected_data['debug_str']}")

        # --- HYSTERESIS ---
        if is_any_hand_up:
            self.raise_counter = min(self.raise_counter + 1, self.TRIGGER_THRESHOLD)
        else:
            self.raise_counter = max(self.raise_counter - 1, 0)

        # Final Decision
        if self.raise_counter >= self.TRIGGER_THRESHOLD:
            gesture_detected = True
            self.is_gesture_active = True
            print(f"!!! GESTURE CONFIRMED (Target: {nose_coords}) !!!")
        else:
            self.is_gesture_active = False

        return {
            "gesture_detected": gesture_detected,
            "frames_held": self.raise_counter,
            "debug_msg": debug_msg,
            "nose_coords": nose_coords,
            "person_orientation": person_orientation,
            "annotated_frame": annotated_frame
        }
