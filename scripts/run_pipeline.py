from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path, PureWindowsPath
import os
import re
import sys
import faulthandler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.input_adapter import InputAdapter
from adapters.geopyv_runner import GeopyvRunner


def running_in_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except Exception:
        return False


def looks_like_windows_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value))


def windows_to_wsl_path(value: str) -> str:
    win_path = PureWindowsPath(value)
    drive = win_path.drive.rstrip(":").lower()

    if not drive:
        return value

    tail_parts = list(win_path.parts[1:])  # skip "C:\\"
    return str(Path("/mnt") / drive / Path(*tail_parts))


def normalize_input_path(raw_path: str) -> Path:
    candidate = raw_path.strip()

    if running_in_wsl() and looks_like_windows_path(candidate):
        converted = windows_to_wsl_path(candidate)
        print(f"[path] Windows path detected under WSL.")
        print(f"[path] Converted: {candidate}  ->  {converted}")
        candidate = converted

    # Espande anche "~" se mai servisse
    return Path(os.path.expanduser(candidate)).resolve()


def make_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal pipeline for geopyv-enhanced"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input video file or folder containing image frames",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory where pipeline runs will be stored",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: auto-safe on WSL)",
    )
    parser.add_argument(
        "--conservative",
        action="store_true",
        help="Enable conservative geopyv settings (safer on WSL/native extensions)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Safer defaults for non-GUI and native-extension stability.
    os.environ.setdefault("MPLBACKEND", "Agg")
    if running_in_wsl():
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    max_frames = args.max_frames
    if max_frames is None and running_in_wsl():
        max_frames = 120

    input_path = normalize_input_path(args.input)
    runs_dir = Path(args.runs_dir).resolve()
    run_dir = make_run_dir(runs_dir)

    faulthandler_path = run_dir / "logs" / "faulthandler.log"
    fh_file = faulthandler_path.open("w", encoding="utf-8")
    faulthandler.enable(file=fh_file, all_threads=True)

    try:
        adapter = InputAdapter(work_root=run_dir, max_frames=max_frames)
        prepared = adapter.prepare(input_path)

        runner = GeopyvRunner(results_dir=run_dir / "results")
        runner.check_environment()
        result = runner.run(
            prepared.frames_dir,
            force_conservative_mode=args.conservative or running_in_wsl(),
        )

        run_info = {
            "timestamp": datetime.now().isoformat(),
            "input_path": str(prepared.source_path),
            "input_type": prepared.input_type,
            "frames_dir": str(prepared.frames_dir),
            "frame_count": prepared.frame_count,
            "max_frames": max_frames,
            "conservative": bool(args.conservative or running_in_wsl()),
            "success": result.success,
            "message": result.message,
            "results_dir": str(result.results_dir),
            "faulthandler_log": str(faulthandler_path),
        }

        run_info_path = run_dir / "run_info.json"
        run_info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")

        print("\n=== Pipeline completed ===")
        print(f"Run directory : {run_dir}")
        print(f"Input type    : {prepared.input_type}")
        print(f"Frames count  : {prepared.frame_count}")
        print(f"Results dir   : {result.results_dir}")
        print(f"Message       : {result.message}")
        print(f"Faulthandler  : {faulthandler_path}")
    finally:
        faulthandler.disable()
        fh_file.close()


if __name__ == "__main__":
    main()