from math import atan2
from ultralytics import YOLO
import numpy as np


class GestureDetector:
    """
    Detects hand-raise gestures from video frames using YOLO pose estimation.
    Iterates ALL detected people to find a waver (not just highest-confidence).
    """

    TRIGGER_THRESHOLD    = 2    # frames gesture must be held to confirm
    KEYPOINT_CONF_THRESH = 0.5

    def __init__(self, model_path='yolo11n-pose.pt'):
        self.model           = YOLO(model_path)
        self.raise_counter   = 0
        self.is_gesture_active = False

    def process_frame(self, frame) -> dict:
        results = self.model(
            frame,
            verbose=False,
            imgsz=640,
            conf=0.25,
            iou=0.5,
            max_det=5,
        )

        annotated_frame  = results[0].plot()
        gesture_detected = False
        debug_msg        = "None"
        nose_coords      = (0, 0)
        person_orient    = None
        is_any_hand_up   = False
        selected_data    = None

        n_people = results[0].boxes.shape[0]
        if n_people > 0:
            sorted_indices = results[0].boxes.conf.argsort(descending=True)

            for idx in sorted_indices:
                idx  = idx.item()
                kpts = results[0].keypoints.data[idx]
                conf = results[0].boxes.conf[idx].item()

                nose                              = kpts[0]
                l_shoulder, r_shoulder            = kpts[5],  kpts[6]
                l_elbow,    r_elbow               = kpts[7],  kpts[8]
                l_wrist,    r_wrist               = kpts[9],  kpts[10]

                ls_y, rs_y = float(l_shoulder[1]), float(r_shoulder[1])
                le_y, re_y = float(l_elbow[1]),    float(r_elbow[1])
                lw_y, rw_y = float(l_wrist[1]),    float(r_wrist[1])

                lw_conf = float(l_wrist[2])
                rw_conf = float(r_wrist[2])
                le_conf = float(l_elbow[2])
                re_conf = float(r_elbow[2])

                l_wrist_up = (lw_conf > 0.3) and (lw_y < ls_y - 10)
                l_elbow_up = (le_conf > 0.5) and (le_y < ls_y - 10)
                r_wrist_up = (rw_conf > 0.3) and (rw_y < rs_y - 10)
                r_elbow_up = (re_conf > 0.5) and (re_y < rs_y - 10)

                is_left_up  = l_wrist_up or l_elbow_up
                is_right_up = r_wrist_up or r_elbow_up
                is_up       = is_left_up or is_right_up

                orientation = None
                if l_shoulder[2] > 0.5 and r_shoulder[2] > 0.5:
                    orientation = atan2(
                        float(r_shoulder[1]) - float(l_shoulder[1]),
                        float(r_shoulder[0]) - float(l_shoulder[0]),
                    )

                person_data = {
                    "conf":       conf,
                    "nose_coords": (int(nose[0]), int(nose[1])),
                    "debug_str":  f"P{idx}({conf:.2f}): LW={lw_y:.0f} RW={rw_y:.0f} Sh={ls_y:.0f}",
                    "is_up":      is_up,
                    "orientation": orientation,
                }

                if is_up:
                    selected_data   = person_data
                    is_any_hand_up  = True
                    debug_msg       = ("Left Up" if is_left_up else "") + \
                                      ("Right Up" if is_right_up else "")
                    print(f">>> PERSON FOUND: {person_data['debug_str']} <<<")
                    break

                if selected_data is None:
                    selected_data = person_data

        if selected_data:
            nose_coords   = selected_data["nose_coords"]
            person_orient = selected_data["orientation"]
            if not is_any_hand_up:
                print(f" Top candidate (no gesture): {selected_data['debug_str']}")

        # Hysteresis counter
        if is_any_hand_up:
            self.raise_counter = min(self.raise_counter + 1, self.TRIGGER_THRESHOLD)
        else:
            self.raise_counter = max(self.raise_counter - 1, 0)

        # FIX: is_gesture_active reflects the confirmed state, reset when counter drops
        if self.raise_counter >= self.TRIGGER_THRESHOLD:
            gesture_detected       = True
            self.is_gesture_active = True
            print(f"!!! GESTURE CONFIRMED → {nose_coords} !!!")
        else:
            self.is_gesture_active = False  # was missing in original

        return {
            "gesture_detected":  gesture_detected,
            "frames_held":       self.raise_counter,
            "debug_msg":         debug_msg,
            "nose_coords":       nose_coords,
            "person_orientation": person_orient,
            "annotated_frame":   annotated_frame,
        }