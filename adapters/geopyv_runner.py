from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from collections import Counter
import os

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
        prefixes = []
        for stem in stems:
            m = re.match(r"^(.*?)(\d+)$", stem)
            prefixes.append(m.group(1) if m else stem)

        if prefixes[0] == prefixes[1]:
            common_name = prefixes[0]
        else:
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

    @staticmethod
    def _running_in_wsl() -> bool:
        try:
            return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
        except Exception:
            return False

    def _manual_setup(self, first_image_path: Path):
        import geopyv as gp
        import cv2

        # ==========================================================
        # MANUAL PARAMETERS TO EDIT
        # Coordinates are in image pixels: [x, y]
        # ==========================================================

        image = cv2.imread(str(first_image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read first image: {first_image_path}")

        height, width = image.shape[:2]
        min_dim = min(height, width)
        if min_dim < 100:
            raise RuntimeError(
                f"Image resolution too small for robust geopyv solve ({width}x{height})."
            )

        margin_x = max(20.0, width * 0.10)
        margin_y = max(20.0, height * 0.10)

        x0 = margin_x
        x1 = max(margin_x + 20.0, width - margin_x)
        y0 = margin_y
        y1 = max(margin_y + 20.0, height - margin_y)

        boundary_nodes = np.asarray(
            [[x0, y0], [x0, y1], [x1, y1], [x1, y0]],
            dtype=float,
        )

        seed_coord = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=float)

        template_radius = int(max(10, min(50, min_dim // 25)))
        target_nodes = int(max(80, min(220, (width * height) // 90000)))
        seed_tolerance = 0.6
        tolerance = 0.5

        # Optional exclusion regions inside the valid domain.
        exclusion_objs = []
        # Example:
        # exclusion_objs.append(
        #     gp.geometry.region.Circle(
        #         centre=np.asarray([2500.0, 1600.0]),
        #         radius=120.0,
        #         size=20.0,
        #         option="F",
        #         hard=True,
        #     )
        # )

        boundary_obj = gp.geometry.region.Path(
            nodes=boundary_nodes,
            hard=False,
        )

        template = gp.templates.Circle(template_radius)

        setup = {
            "image_width": int(width),
            "image_height": int(height),
            "boundary_nodes": boundary_nodes,
            "boundary_obj": boundary_obj,
            "exclusion_objs": exclusion_objs,
            "seed_coord": seed_coord,
            "template": template,
            "template_radius": template_radius,
            "target_nodes": target_nodes,
            "seed_tolerance": seed_tolerance,
            "tolerance": tolerance,
        }
        return setup

    def initialise_sequence(
        self,
        frames_dir: Path,
        *,
        save_by_reference: bool = False,
    ):
        import geopyv as gp

        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        image_files = self._list_image_candidates(frames_dir)
        common_name, file_format = self._infer_common_name_and_format(image_files)
        setup = self._manual_setup(image_files[0])

        sequence = gp.sequence.Sequence(
            image_dir=str(frames_dir),
            common_name=common_name,
            file_format=file_format,
            target_nodes=setup["target_nodes"],
            boundary_obj=setup["boundary_obj"],
            exclusion_objs=setup["exclusion_objs"],
            save_by_reference=save_by_reference,
            mesh_dir=str(self.results_dir),
            ID="pipeline_sequence",
        )
        return sequence, setup, common_name, file_format

    def run(self, frames_dir: Path, *, force_conservative_mode: bool = True) -> RunResult:
        frames_dir = Path(frames_dir)

        sequence, setup, common_name, file_format = self.initialise_sequence(frames_dir)

        # Conservative defaults reduce native crashes under WSL/high thread pressure.
        conservative_mode = force_conservative_mode or self._running_in_wsl()
        sequential = bool(conservative_mode)
        sync = not sequential
        max_iterations = 20 if conservative_mode else 30
        adaptive_iterations = 0 if conservative_mode else 1
        alpha = 0.4 if conservative_mode else 0.5

        if conservative_mode:
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        solved = sequence.solve(
            seed_coord=setup["seed_coord"],
            template=setup["template"],
            max_norm=1e-5,
            max_iterations=max_iterations,
            adaptive_iterations=adaptive_iterations,
            method="ICGN",
            mesh_order=2,
            subset_order=1,
            tolerance=setup["tolerance"],
            seed_tolerance=setup["seed_tolerance"],
            alpha=alpha,
            guide=True,
            override=False,
            sequential=sequential,
            sync=sync,
            dense=False,
        )

        summary = {
            "sequence_type": type(sequence).__name__,
            "image_dir": str(frames_dir),
            "initialised": True,
            "solved": bool(solved),
            "inferred_common_name": common_name,
            "inferred_file_format": file_format,
            "boundary_nodes": setup["boundary_nodes"].tolist(),
            "seed_coord": setup["seed_coord"].tolist(),
            "template_radius": int(setup["template_radius"]),
            "target_nodes": int(setup["target_nodes"]),
            "seed_tolerance": float(setup["seed_tolerance"]),
            "tolerance": float(setup["tolerance"]),
            "exclusion_count": len(setup["exclusion_objs"]),
            "image_width": setup["image_width"],
            "image_height": setup["image_height"],
            "conservative_mode": conservative_mode,
            "sequential": sequential,
            "sync": sync,
            "max_iterations": max_iterations,
            "adaptive_iterations": adaptive_iterations,
            "alpha": alpha,
        }

        if hasattr(sequence, "data"):
            summary["data_keys"] = list(sequence.data.keys())
            summary["sequence_data"] = {
                "solved": sequence.data.get("solved"),
                "unsolvable": sequence.data.get("unsolvable"),
                "sync": sequence.data.get("sync"),
                "dense": sequence.data.get("dense"),
            }

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

        summary_file = self.results_dir / "sequence_solve_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        message = (
            "Geopyv sequence solved successfully."
            if solved
            else "Geopyv sequence initialised but solve did not complete successfully."
        )

        return RunResult(
            success=bool(solved),
            message=message,
            results_dir=self.results_dir,
        )
