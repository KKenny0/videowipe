"""Auto-download model weights from GitHub Releases."""
from __future__ import annotations

import os
import urllib.request
from typing import Optional

from tqdm import tqdm

_WEIGHTS_DIR = os.environ.get(
    "VIDEOWIPE_WEIGHTS_DIR",
    os.path.expanduser("~/.videowipe/weights"),
)

_RELEASE_BASE = (
    "https://github.com/KKenny0/videowipe/releases/download"
)

_MANUAL_DOWNLOAD_MSG = (
    "Automatic download failed. Download the weight file manually from:\n"
    "  https://github.com/KKenny0/videowipe/releases\n"
    "Place it in ~/.videowipe/weights/ or set VIDEOWIPE_WEIGHTS_DIR."
)


def get_weights_dir() -> str:
    os.makedirs(_WEIGHTS_DIR, exist_ok=True)
    return _WEIGHTS_DIR


def ensure_weight(filename: str, version: str = "v0.1") -> str:
    """Return the local path to a weight file, downloading if necessary.

    Returns the absolute path to the weight file on disk.
    """
    weights_dir = get_weights_dir()
    local_path = os.path.join(weights_dir, filename)
    if os.path.isfile(local_path):
        return local_path

    url = f"{_RELEASE_BASE}/{version}/{filename}"
    print(f"Downloading {filename} from {url}")

    try:
        _download_with_progress(url, local_path)
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise RuntimeError(
            f"Failed to download {filename}: {e}\n{_MANUAL_DOWNLOAD_MSG}"
        ) from e

    return local_path


def _download_with_progress(url: str, dest: str) -> None:
    tmp_path = dest + ".tmp"

    class _Progress(tqdm):
        def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with _Progress(unit="B", unit_scale=True, desc=os.path.basename(dest)) as pbar:
        urllib.request.urlretrieve(
            url, tmp_path, reporthook=pbar.update_to
        )

    os.replace(tmp_path, dest)
