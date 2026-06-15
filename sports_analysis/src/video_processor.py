import cv2
import os


class VideoProcessor:
    """Processes sports videos by extracting and preprocessing frames."""

    def __init__(self, video_path, output_dir, fps=5):
        """Initialize the processor with video path, output directory, and target fps."""
        self.video_path = video_path
        self.output_dir = output_dir
        self.fps = fps

    def get_video_info(self):
        """Return metadata about the video including duration, fps, and resolution."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / original_fps if original_fps else 0

        info = {
            "filename": os.path.basename(self.video_path),
            "duration_seconds": duration_seconds,
            "original_fps": original_fps,
            "resolution": f"{width}x{height}",
            "total_frames": total_frames,
        }

        for key, value in info.items():
            print(f"{key}: {value}")

        cap.release()
        return info

    def extract_frames(self):
        """Extract frames from the video at the target fps and save them as JPEG files."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / self.fps)
        saved_count = 0
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                saved_count += 1
                filename = f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(os.path.join(self.output_dir, filename), frame)
                if saved_count % 50 == 0:
                    print(f"Saved {saved_count} frames")

            frame_index += 1

        cap.release()
        return saved_count

    def preprocess_frame(self, frame):
        """Resize a frame to 640x640 pixels."""
        return cv2.resize(frame, (640, 640))
