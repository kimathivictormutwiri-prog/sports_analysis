import numpy as np
import cv2
from collections import defaultdict


class FeatureExtractor:
    """Extracts player movement features and team assignments from detections."""

    def __init__(self, fps=5):
        """Initialize the extractor with target fps and empty player history."""
        self.fps = fps
        self.player_history = defaultdict(list)

    def assign_team(self, frame, detections):
        """Assign team labels to player detections based on jersey colour clustering."""
        person_indices = []
        hues = []

        for index, detection in enumerate(detections):
            if detection["class_id"] != 0:
                continue

            x1, y1, x2, y2 = [int(v) for v in detection["bbox"]]
            crop = frame[y1:y2, x1:x2]
            avg_bgr = cv2.mean(crop)[:3]
            bgr_pixel = np.uint8([[list(avg_bgr)]])
            hsv = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
            hues.append(float(hsv[0][0][0]))
            person_indices.append(index)

        if len(hues) >= 2:
            hue_array = np.array(hues, dtype=np.float64)
            center_0 = hue_array.min()
            center_1 = hue_array.max()

            for _ in range(10):
                dist_0 = np.abs(hue_array - center_0)
                dist_1 = np.abs(hue_array - center_1)
                labels = (dist_1 < dist_0).astype(int)
                if np.any(labels == 0):
                    center_0 = hue_array[labels == 0].mean()
                if np.any(labels == 1):
                    center_1 = hue_array[labels == 1].mean()

            midpoint = (center_0 + center_1) / 2

            for person_index, hue in zip(person_indices, hues):
                detections[person_index]["team"] = 0 if hue < midpoint else 1
        elif len(hues) == 1:
            detections[person_indices[0]]["team"] = 0

        return detections

    def update_history(self, frame_index, detections):
        """Append frame position data to the history for each tracked detection."""
        for detection in detections:
            entry = {
                "frame_index": frame_index,
                "center": detection["center"],
            }
            if "team" in detection:
                entry["team"] = detection["team"]
            self.player_history[detection["track_id"]].append(entry)

    def compute_speed_distance(self, track_id, pixel_to_meter=0.05):
        """Compute total distance, average speed, and max speed for a tracked player."""
        history = self.player_history[track_id]
        if len(history) < 2:
            return {
                "total_distance_m": 0.0,
                "avg_speed_mps": 0.0,
                "max_speed_mps": 0.0,
            }

        total_distance_m = 0.0
        speeds = []

        for i in range(1, len(history)):
            c0 = history[i - 1]["center"]
            c1 = history[i]["center"]
            dist_px = np.sqrt((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2)
            dist_m = dist_px * pixel_to_meter
            total_distance_m += dist_m
            speeds.append(dist_m * self.fps)

        return {
            "total_distance_m": total_distance_m,
            "avg_speed_mps": float(np.mean(speeds)),
            "max_speed_mps": float(np.max(speeds)),
        }

    def compute_fatigue_index(self, track_id):
        """Compare early and late movement speeds to estimate fatigue percentage."""
        history = self.player_history[track_id]
        if len(history) < 6:
            return {
                "early_speed_mps": 0.0,
                "late_speed_mps": 0.0,
                "fatigue_pct": 0.0,
            }

        third = len(history) // 3
        early_history = history[:third]
        late_history = history[-third:]
        pixel_to_meter = 0.05
        early_speeds = []
        late_speeds = []

        for i in range(1, len(early_history)):
            c0 = early_history[i - 1]["center"]
            c1 = early_history[i]["center"]
            dist_px = np.sqrt((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2)
            early_speeds.append(dist_px * pixel_to_meter * self.fps)

        for i in range(1, len(late_history)):
            c0 = late_history[i - 1]["center"]
            c1 = late_history[i]["center"]
            dist_px = np.sqrt((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2)
            late_speeds.append(dist_px * pixel_to_meter * self.fps)

        early_speed_mps = float(np.mean(early_speeds)) if early_speeds else 0.0
        late_speed_mps = float(np.mean(late_speeds)) if late_speeds else 0.0

        if early_speed_mps == 0:
            fatigue_pct = 0.0
        else:
            fatigue_pct = ((early_speed_mps - late_speed_mps) / early_speed_mps) * 100

        return {
            "early_speed_mps": early_speed_mps,
            "late_speed_mps": late_speed_mps,
            "fatigue_pct": fatigue_pct,
        }
