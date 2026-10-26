"""Prediction readers and atomic work files."""

import hashlib
import itertools
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def save_part(path, **arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def slices(lo, hi):
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def tiles(shape, size):
    for start in itertools.product(*(range(0, n, size) for n in shape)):
        lo = np.array(start)
        yield lo, np.minimum(lo + size, shape)


def file_signature(path):
    """Detect replacement or modification without reading a second volume copy."""
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for item in paths:
        stat = item.stat()
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


class Predictions:
    """Local NPY or Zarr array with two channels, returned in V,H order."""

    def __init__(self, path, key=None, order="VH"):
        self.path = Path(path).resolve(strict=True)
        if self.path.suffix == ".npy":
            if key:
                raise ValueError("--array is only valid for a Zarr group")
            self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        else:
            try:
                import zarr
            except ImportError as exc:
                raise ValueError("Install avf-spline-generator[zarr] to read Zarr") from exc
            self.array = zarr.open(str(self.path), mode="r")
            if key:
                try:
                    self.array = self.array[key]
                except KeyError as exc:
                    raise ValueError(f"Zarr array not found: {key}") from exc
        if not hasattr(self.array, "shape") or len(self.array.shape) != 4:
            raise ValueError("Predictions must be an array with shape (2, Z, Y, X)")
        if self.array.shape[0] != 2 or min(self.array.shape[1:]) < 2:
            raise ValueError("Expected two channels and at least 2 voxels along each spatial axis")
        self.shape = tuple(self.array.shape[1:])
        self.dtype = np.dtype(self.array.dtype)
        if self.dtype != np.uint8 and self.dtype.kind != "f":
            raise ValueError("Predictions must be uint8 [0,255] or floating point [0,1]")
        self.reverse = order == "HV"

    def read(self, lo, hi, *, quantized=True):
        block = np.asarray(self.array[(slice(None),) + slices(lo, hi)])
        if self.reverse:
            block = block[::-1]
        if self.dtype.kind == "f":
            if not np.isfinite(block).all() or (block < 0).any() or (block > 1).any():
                raise ValueError("Predictions contain NaN, infinity or values outside [0,1]")
            block = block.astype("f4")
            if quantized:
                block = np.rint(block * 255).astype("u1")
        return np.ascontiguousarray(block)


class NativeSampler:
    """Trilinear sampling with a bounded cache of native prediction chunks."""

    def __init__(self, predictions, cache_mib=128, chunk=64):
        self.predictions = predictions
        self.shape = np.array(predictions.shape)
        self.chunk = chunk
        self.limit = cache_mib * 1024**2
        self.cache = OrderedDict()
        self.bytes = 0

    def sample(self, points):
        q = np.clip(points[:, ::-1], 0, self.shape - 1)
        lo = np.floor(q).astype("i8")
        frac = q - lo
        values = np.zeros((len(q), 2), "f4")
        grid = (self.shape + self.chunk - 1) // self.chunk
        for corner in itertools.product((0, 1), repeat=3):
            corner = np.array(corner)
            ij = np.minimum(lo + corner, self.shape - 1)
            weight = np.prod(np.where(corner, frac, 1 - frac), axis=1)
            chunks = ij // self.chunk
            keys = (chunks[:, 0] * grid[1] + chunks[:, 1]) * grid[2] + chunks[:, 2]
            order = np.argsort(keys, kind="stable")
            boundaries = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1, len(order)]
            for a, b in zip(boundaries[:-1], boundaries[1:]):
                if a == b:
                    continue
                ids = order[a:b]
                key = tuple(chunks[ids[0]])
                block = self.cache.pop(key, None)
                if block is None:
                    start = np.array(key) * self.chunk
                    block = self.predictions.read(start, np.minimum(start + self.chunk, self.shape))
                    self.bytes += block.nbytes
                self.cache[key] = block
                while self.bytes > self.limit and len(self.cache) > 1:
                    _, removed = self.cache.popitem(last=False)
                    self.bytes -= removed.nbytes
                local = ij[ids] % self.chunk
                values[ids] += block[(slice(None),) + tuple(local.T)].T * weight[ids, None]
        return values
