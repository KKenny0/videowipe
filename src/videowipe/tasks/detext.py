"""Subtitle removal task — delegates inpainting to an Inpainter."""
from videowipe.tasks.base import BaseTask


class DetextTask(BaseTask):
    """Remove hardcoded subtitles from video.

    The inpainting loop lives in the Inpainter injected by WipeEngine (default
    ``STTNInpainter``). This task only assembles the :class:`InpaintJob` from
    the prepared reader/mask and returns the output path.
    """

    def process_video(self, reader, frame_info, mask, output_dir: str,
                      video_path: str = "", progress=None,
                      frame_mask=None) -> str:
        from videowipe.inpainters.base import InpaintJob

        metrics = self._bm.get("timing", {}) if isinstance(self._bm, dict) else {}
        job = InpaintJob(
            video_path=video_path,
            mask=mask,
            output_dir=output_dir,
            fps=frame_info["fps"],
            frame_count=frame_info["len"],
            width=frame_info["W_ori"],
            height=frame_info["H_ori"],
            dual=self.dual,
            gap=self.gap,
            output_suffix=getattr(self, "output_suffix", "detext"),
            reader=reader,
            progress=progress,
            metrics=metrics,
            mask_path=getattr(self, "mask_path", None),
            feather_radius=getattr(self, "feather_radius", 0),
            frame_mask=frame_mask,
        )
        outcome = self.inpainter.inpaint(job)
        self.backend_label = outcome.backend
        return outcome.output_path
