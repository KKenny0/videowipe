"""Inspect VideoWipe distribution artifacts without third-party packages."""
from __future__ import annotations

import argparse
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


REQUIRED_WHEEL_SUFFIXES = {
    "videowipe/__init__.py",
    "videowipe/api.py",
    "videowipe/errors.py",
    "videowipe/engine.py",
    "videowipe/propainter_wipe.py",
}
FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}
MAX_SDIST_BYTES = 2 * 1024 * 1024
ALLOWED_SDIST_ROOT_FILES = {
    ".gitignore",
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "README_CN.md",
    "pyproject.toml",
}
ALLOWED_SDIST_PREFIXES = ("scripts/", "src/", "tests/")


def _reject_noise(names):
    noisy = [
        name for name in names
        if any(part in FORBIDDEN_PARTS for part in Path(name).parts)
        or name.endswith((".pyc", ".log"))
    ]
    if noisy:
        raise SystemExit(f"generated noise found in package: {noisy[:5]}")


def _validated_sdist_files(members):
    roots = set()
    files = []
    for member in members:
        name = member.name.replace("\\", "/")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe path in sdist: {member.name}")
        if not path.parts:
            raise SystemExit("empty path in sdist")
        roots.add(path.parts[0])
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported member type in sdist: {member.name}")
        if member.isfile():
            if len(path.parts) < 2:
                raise SystemExit(f"file outside sdist root: {member.name}")
            files.append(PurePosixPath(*path.parts[1:]).as_posix())
    if len(roots) != 1:
        raise SystemExit(f"sdist must have one top-level directory: {sorted(roots)}")
    return files


def verify(dist_dir: Path) -> None:
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("expected exactly one wheel and one sdist")

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = archive.namelist()
        _reject_noise(wheel_names)
        missing = [
            suffix for suffix in REQUIRED_WHEEL_SUFFIXES
            if not any(name.endswith(suffix) for name in wheel_names)
        ]
        if missing:
            raise SystemExit(f"wheel is missing required files: {missing}")
        metadata_name = next(
            name for name in wheel_names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode("utf-8")
        if "GNU General Public License v3 (GPLv3)" not in metadata:
            raise SystemExit("wheel metadata does not declare GPL-3.0")
        if not any(name.endswith(".dist-info/licenses/LICENSE") for name in wheel_names):
            raise SystemExit("wheel does not include LICENSE")

    with tarfile.open(sdists[0], "r:gz") as archive:
        members = archive.getmembers()
        sdist_names = [member.name for member in members]
        sdist_files = _validated_sdist_files(members)
        _reject_noise(sdist_names)
        if "LICENSE" not in sdist_files:
            raise SystemExit("sdist does not include LICENSE")
        disallowed = [
            name for name in sdist_files
            if name not in ALLOWED_SDIST_ROOT_FILES
            and not name.startswith(ALLOWED_SDIST_PREFIXES)
        ]
        if disallowed:
            raise SystemExit(f"sdist contains disallowed paths: {disallowed[:10]}")
        tracked = set(subprocess.check_output(
            ["git", "ls-files"], text=True, encoding="utf-8"
        ).splitlines())
        unexpected = [
            name for name in sdist_files
            if name != "PKG-INFO" and name not in tracked
        ]
        if unexpected:
            raise SystemExit(
                f"sdist contains files outside git tracked state: {unexpected[:10]}"
            )
    if sdists[0].stat().st_size > MAX_SDIST_BYTES:
        raise SystemExit(
            f"sdist exceeds {MAX_SDIST_BYTES} bytes: {sdists[0].stat().st_size}"
        )

    print(f"verified wheel: {wheels[0].name} ({len(wheel_names)} entries)")
    print(f"verified sdist: {sdists[0].name} ({len(sdist_names)} entries)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", nargs="?", default="dist", type=Path)
    args = parser.parse_args()
    verify(args.dist_dir)


if __name__ == "__main__":
    main()
