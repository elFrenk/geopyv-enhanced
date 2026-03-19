from pathlib import Path

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_VIDEO_EXTENSIONS


def list_image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {folder}")

    images = sorted(
        [p for p in folder.iterdir() if is_image_file(p)]
    )
    return images


def ensure_nonempty_images(folder: Path) -> list[Path]:
    images = list_image_files(folder)
    if not images:
        raise ValueError(
            f"No valid image files found in folder: {folder}\n"
            f"Allowed extensions: {sorted(VALID_IMAGE_EXTENSIONS)}"
        )
    return images