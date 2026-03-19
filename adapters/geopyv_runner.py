from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from collections import Counter

import cv2
import numpy as np


@dataclass
class RunResult:
    success: bool
    message: str
    results_dir: Path


class GeopyvRunner:
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def check_environment(self) -> None:
        try:
            import geopyv  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Unable to import geopyv. Check your environment and installation."
            ) from e

    def _list_image_candidates(self, frames_dir: Path) -> list[Path]:
        allowed = {".png", ".jpg", ".jpeg"}
        files = sorted(
            [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed]
        )
        if len(files) < 2:
            raise FileNotFoundError(
                f"Expected at least 2 image files in {frames_dir}, found {len(files)}."
            )
        return files

    def _infer_common_name_and_format(self, image_files: list[Path]) -> tuple[str, str]:
        suffixes = [p.suffix.lower() for p in image_files]
        file_format = Counter(suffixes).most_common(1)[0][0]

        same_ext_files = [p for p in image_files if p.suffix.lower() == file_format]
        if len(same_ext_files) < 2:
            raise RuntimeError(
                f"Could not infer a dominant file format with at least 2 files. "
                f"Formats found: {sorted(set(suffixes))}"
            )

        stems = [p.stem for p in same_ext_files[:2]]

        # Take everything before the trailing number block.
        prefixes = []
        for stem in stems:
            m = re.match(r"^(.*?)(\d+)$", stem)
            if m:
                prefixes.append(m.group(1))
            else:
                prefixes.append(stem)

        if prefixes[0] == prefixes[1]:
            common_name = prefixes[0]
        else:
            # Fallback: longest common prefix
            a, b = prefixes[0], prefixes[1]
            i = 0
            while i < min(len(a), len(b)) and a[i] == b[i]:
                i += 1
            common_name = a[:i]

        if common_name == "":
            raise RuntimeError(
                "Could not infer a non-empty common filename prefix for geopyv."
            )

        return common_name, file_format

    def _detect_first_frame(self, frames_dir: Path, common_name: str, file_format: str) -> Path:
        candidates = sorted(frames_dir.glob(f"{common_name}*{file_format}"))
        if len(candidates) < 2:
            raise FileNotFoundError(
                f"Expected at least 2 frames matching {common_name}*{file_format} in {frames_dir}"
            )
        return candidates[0]

    def _build_full_image_boundary(self, frame_path: Path):
        import geopyv as gp

        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")

        height, width = image.shape[:2]

        nodes = np.array(
            [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ],
            dtype=float,
        )

        boundary_obj = gp.geometry.region.Path(
            nodes=nodes,
            option="F",
            hard=True,
            compensate=True,
        )

        return boundary_obj, {"width": int(width), "height": int(height), "nodes": nodes.tolist()}

    def initialise_sequence(
        self,
        frames_dir: Path,
        *,
        target_nodes: int = 1000,
        save_by_reference: bool = False,
    ):
        import geopyv as gp

        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        image_files = self._list_image_candidates(frames_dir)
        common_name, file_format = self._infer_common_name_and_format(image_files)
        first_frame = self._detect_first_frame(frames_dir, common_name, file_format)
        boundary_obj, boundary_info = self._build_full_image_boundary(first_frame)

        sequence = gp.sequence.Sequence(
            image_dir=str(frames_dir),
            common_name=common_name,
            file_format=file_format,
            target_nodes=target_nodes,
            boundary_obj=boundary_obj,
            exclusion_objs=[],
            save_by_reference=save_by_reference,
            mesh_dir=str(self.results_dir),
            ID="pipeline_sequence",
        )
        return sequence, boundary_info, common_name, file_format

    def run(self, frames_dir: Path) -> RunResult:
        frames_dir = Path(frames_dir)

        sequence, boundary_info, common_name, file_format = self.initialise_sequence(frames_dir)

        summary = {
            "sequence_type": type(sequence).__name__,
            "image_dir": str(frames_dir),
            "initialised": True,
            "boundary_info": boundary_info,
            "inferred_common_name": common_name,
            "inferred_file_format": file_format,
        }

        if hasattr(sequence, "data"):
            summary["data_keys"] = list(sequence.data.keys())
            file_settings = sequence.data.get("file_settings", {})
            summary["file_settings"] = {
                "image_dir": file_settings.get("image_dir"),
                "common_name": file_settings.get("common_name"),
                "file_format": file_settings.get("file_format"),
                "save_by_reference": file_settings.get("save_by_reference"),
                "mesh_dir": file_settings.get("mesh_dir"),
            }

            images = file_settings.get("images")
            if images is not None:
                summary["image_count_detected_by_geopyv"] = len(images)
                summary["first_images"] = list(images[:5])

            mesh_settings = sequence.data.get("mesh_settings", {})
            boundary = mesh_settings.get("boundary_obj")
            if boundary is not None and hasattr(boundary, "data"):
                summary["boundary_shape"] = boundary.data.get("shape")

        summary_file = self.results_dir / "sequence_initialisation_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return RunResult(
            success=True,
            message="Geopyv sequence initialised successfully with automatic image boundary.",
            results_dir=self.results_dir,
        )