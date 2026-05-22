"""Logo removal task (placeholder)."""
from videowipe.tasks.base import BaseTask


class DelogoTask(BaseTask):
    """Remove channel logos from video. Placeholder implementation."""

    def process_video(self, reader, frame_info, mask, output_dir, video_path=""):
        print("Logo removal is not yet implemented.")
        return None
