from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

from adapters.validation import (
    ensure_nonempty_images,
    is_video_file,
)


@dataclass
class PreparedInput:
    source_path: Path
    input_type: str          # "image_folder" or "video"
    frames_dir: Path
    frame_count: int


class InputAdapter:
    def __init__(self, work_root: Path, *, max_frames: int | None = None):
        self.work_root = Path(work_root)
        self.max_frames = max_frames

    def prepare(self, input_path: Path) -> PreparedInput:
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        if input_path.is_dir():
            return self._prepare_from_image_folder(input_path)

        if is_video_file(input_path):
            return self._prepare_from_video(input_path)

        raise ValueError(
            f"Unsupported input path: {input_path}\n"
            f"Expected a folder with images or a video file."
        )

    def _prepare_from_image_folder(self, folder: Path) -> PreparedInput:
        images = ensure_nonempty_images(folder)
        if self.max_frames is not None and self.max_frames > 0:
            images = images[: self.max_frames]

        frames_dir = self.work_root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for img in images:
            shutil.copy2(img, frames_dir / img.name)

        copied_images = ensure_nonempty_images(frames_dir)

        return PreparedInput(
            source_path=folder,
            input_type="image_folder",
            frames_dir=frames_dir,
            frame_count=len(copied_images),
        )

    def _prepare_from_video(self, video_path: Path) -> PreparedInput:
        frames_dir = self.work_root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        output_pattern = frames_dir / "frame_%06d.png"

        self._extract_video_frames_ffmpeg(
            video_path,
            output_pattern,
            max_frames=self.max_frames,
        )

        images = ensure_nonempty_images(frames_dir)

        return PreparedInput(
            source_path=video_path,
            input_type="video",
            frames_dir=frames_dir,
            frame_count=len(images),
        )

    @staticmethod
    def _extract_video_frames_ffmpeg(
        video_path: Path,
        output_pattern: Path,
        *,
        max_frames: int | None = None,
    ) -> None:
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
        ]

        if max_frames is not None and max_frames > 0:
            ffmpeg_cmd.extend(["-vframes", str(max_frames)])

        ffmpeg_cmd.extend([
            str(output_pattern),
        ])

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and ensure it is available in PATH."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "ffmpeg failed while extracting frames.\n"
                f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
            ) from e