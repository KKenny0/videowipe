"""Bounded STTN crop predictions; optional acceleration, never an output store."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import uuid
import warnings
import zipfile

import numpy as np

TASK_LIMIT = 512 * 1024 * 1024
TOTAL_LIMIT = 2 * 1024 * 1024 * 1024
# ponytail: one local server process; add cross-process locking before multi-worker support.
_write_lock = threading.Lock()


class PredictionCache:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError("prediction cache directory cannot be a symlink")
        self.directory = self.directory.resolve()
        self.identity = identity
        self.hits = self.misses = self.corrupt = self.writes = 0

    def key(self, first, last, top, bottom):
        payload = dict(self.identity, frames=[first, last], crop=[top, bottom])
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def read(self, key, count):
        path = self.directory / f"{key}.npz"
        if not path.exists():
            self.misses += 1
            return None
        try:
            if path.is_symlink():
                raise ValueError("symlink cache entry")
            with zipfile.ZipFile(path) as archive:
                if len(archive.namelist()) != 3 or set(archive.namelist()) != {"pixels.npy", "valid.npy", "checksum.npy"}:
                    raise ValueError("unknown cache entries")
                if sum(i.file_size for i in archive.infolist()) > count * (640*120*3 + 1) + 4096:
                    raise ValueError("oversized cache entry")
                # Validate array headers before numpy can allocate their claimed shapes.
                for name, shape, dtype in (("pixels.npy", (count, 120, 640, 3), np.uint8),
                                           ("valid.npy", (count,), np.bool_),
                                           ("checksum.npy", (32,), np.uint8)):
                    with archive.open(name) as member:
                        version = np.lib.format.read_magic(member)
                        if version not in ((1, 0), (2, 0)):
                            raise ValueError("unsupported prediction array format")
                        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                       else np.lib.format.read_array_header_2_0)
                        actual_shape, fortran, actual_dtype = read_header(member, max_header_size=1024)
                        if actual_shape != shape or actual_dtype != np.dtype(dtype) or fortran:
                            raise ValueError("invalid prediction array header")
                        if archive.getinfo(name).file_size != member.tell() + int(np.prod(shape)) * actual_dtype.itemsize:
                            raise ValueError("invalid prediction array size")
            with np.load(path, allow_pickle=False) as stored:
                pixels, valid, checksum = stored['pixels'], stored['valid'], stored['checksum']
            if pixels.shape != (count, 120, 640, 3) or pixels.dtype != np.uint8:
                raise ValueError("invalid prediction shape or dtype")
            if valid.shape != (count,) or valid.dtype != np.bool_:
                raise ValueError("invalid prediction frame markers")
            if checksum.shape != (32,) or checksum.dtype != np.uint8:
                raise ValueError("invalid prediction checksum")
            if hashlib.sha256(pixels.tobytes() + valid.tobytes()).digest() != checksum.tobytes():
                raise ValueError("prediction checksum mismatch")
            self.hits += 1
            return [pixels[i] if valid[i] else None for i in range(count)]
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            self.corrupt += 1
            self.misses += 1
            warnings.warn(f"Corrupt prediction cache ignored: {path.name}", RuntimeWarning)
            return None

    def write(self, key, frames):
        temporary = None
        try:
            pixels = np.zeros((len(frames), 120, 640, 3), dtype=np.uint8)
            valid = np.asarray([frame is not None for frame in frames], dtype=np.bool_)
            for i, frame in enumerate(frames):
                if frame is not None:
                    if frame.shape != pixels.shape[1:] or frame.dtype != np.uint8:
                        raise ValueError("invalid STTN prediction")
                    pixels[i] = frame
            checksum = np.frombuffer(hashlib.sha256(pixels.tobytes() + valid.tobytes()).digest(), dtype=np.uint8)
            with _write_lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                # Web layout is jobs/<id>/predictions. Count only this feature's
                # sibling task directories; inputs and successful videos are never evicted.
                directories = {self.directory}
                directories.update(p.resolve() for p in self.directory.parent.parent.glob(f"*/{self.directory.name}")
                                   if p.is_dir() and not p.is_symlink())
                sizes = {directory: sum(p.stat().st_size for p in directory.glob('*')
                                         if p.suffix in {'.npz', '.tmp'} and p.is_file() and not p.is_symlink()) for directory in directories}
                expected = pixels.nbytes + valid.nbytes + 4096
                if sizes[self.directory] + expected > TASK_LIMIT or sum(sizes.values()) + expected > TOTAL_LIMIT:
                    return
                temporary = self.directory / f".{uuid.uuid4().hex}.tmp"
                with temporary.open('xb') as stream:
                    np.savez(stream, pixels=pixels, valid=valid, checksum=checksum)
                    stream.flush()
                    os.fsync(stream.fileno())
                destination = self.directory / f"{key}.npz"
                if destination.is_symlink():
                    return
                os.replace(temporary, destination)
                self.writes += 1
        except OSError:
            warnings.warn("Prediction cache write skipped; processing continues", RuntimeWarning)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
