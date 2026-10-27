"""Lossless storage adapter for the unchanged ARGON numerical kernels.

Ported from the running resident_block_store/benchmark_resident_argon backend:
global coordinates, bounded tile slots, optional region pinning, and lossless
coverage eviction. Zarr replaces the deployment-specific .blosc file layout.
Parallel smoothing receives an immutable presence snapshot, never this cache.
"""

import itertools
import math
from pathlib import Path

import numpy as np
from numba import objmode, types
from numba.experimental import jitclass
from scipy.ndimage import map_coordinates

from .storage import slices

_CACHES = {}
_TOKENS = itertools.count()


def dependencies():
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise ValueError("Install avf-spline-generator[compressed] for compressed storage") from exc
    if not zarr.__version__.startswith("2."):
        raise ValueError("Compressed storage requires Zarr 2.x")
    return zarr, Blosc


def work_array(folder, name, dtype, shape, *, create=False):
    zarr, Blosc = dependencies()
    path = Path(folder) / (name + ".zarr")
    fill = 128 if name in ("nx", "ny") else 0
    chunks = ((1,) if len(shape) == 4 else ()) + (128, 128, 128)
    if create:
        return zarr.open_array(
            str(path),
            mode="w",
            shape=tuple(shape),
            dtype=dtype,
            chunks=chunks,
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
            fill_value=fill,
            write_empty_chunks=False,
        )
    result = zarr.open_array(str(path), mode="r+", write_empty_chunks=False)
    if (
        result.shape != tuple(shape)
        or result.dtype != np.dtype(dtype)
        or result.chunks != chunks
        or result.fill_value != fill
    ):
        raise ValueError(f"Invalid compressed work array: {path.name}")
    return result


def _load(token, tile):
    return _CACHES[token].load(tile)


def _field_type(dtype, chunk=128):
    @jitclass(
        [
            ("data", dtype[::1]),
            ("index", types.int32[::1]),
            ("shape", types.UniTuple(types.int64, 3)),
            ("grid", types.UniTuple(types.int64, 3)),
            ("chunk", types.int64),
            ("token", types.int64),
            ("writable", types.boolean),
        ]
    )
    class Field:
        def __init__(self, data, index, shape, grid, chunk, token, writable):
            self.data, self.index, self.shape, self.grid = data, index, shape, grid
            self.chunk, self.token, self.writable = chunk, token, writable

        def __getitem__(self, ijk):
            z, y, x = ijk
            c = chunk  # Compile-time constant, as in the production 128³ adapter.
            tile = ((z // c) * self.grid[1] + y // c) * self.grid[2] + x // c
            slot = self.index[tile]
            if slot < 0:
                token = self.token
                with objmode(loaded="int64"):
                    loaded = _load(token, tile)
                slot = loaded
            at = ((z % c) * c + y % c) * c + x % c
            return self.data[slot * c**3 + at]

        def __setitem__(self, ijk, value):
            if not self.writable:
                raise RuntimeError("Prediction fields are read-only during tracing")
            z, y, x = ijk
            c = chunk
            tile = ((z // c) * self.grid[1] + y // c) * self.grid[2] + x // c
            slot = self.index[tile]
            if slot < 0:
                token = self.token
                with objmode(loaded="int64"):
                    loaded = _load(token, tile)
                slot = loaded
            at = ((z % c) * c + y % c) * c + x % c
            self.data[slot * c**3 + at] = value

    return Field


U8, U16 = _field_type(types.uint8), _field_type(types.uint16)
_FIELD_TYPES = {128: (U8, U16)}


@jitclass(
    [
        ("data", types.uint8[::1]),
        ("index", types.int32[::1]),
        ("shape", types.UniTuple(types.int64, 3)),
        ("grid", types.UniTuple(types.int64, 3)),
        ("chunk", types.int64),
    ]
)
class StaticPresence:
    def __init__(self, data, index, shape, grid, chunk):
        self.data, self.index, self.shape = data, index, shape
        self.grid, self.chunk = grid, chunk

    def __getitem__(self, ijk):
        z, y, x = ijk
        c = self.chunk
        tile = ((z // c) * self.grid[1] + y // c) * self.grid[2] + x // c
        slot = self.index[tile]
        if slot < 0:
            raise RuntimeError("Smoothing read outside its immutable presence snapshot")
        return self.data[slot * c**3 + ((z % c) * c + y % c) * c + x % c]


class ResidentFields:
    """Serial paged access; mutable coverage is a cache, not a restart journal."""

    def __init__(self, cent, presence, axes, coverage, cache_mib=1024, chunk=128):
        self.sources = [cent, presence, *axes, *coverage]
        self.shape = tuple(int(n) for n in presence.shape)
        self.chunk = chunk
        self.grid = tuple((n + chunk - 1) // chunk for n in self.shape)
        self.volume = chunk**3
        self.limit = int(cache_mib * 1024**2)
        # One uint16 and six uint8 fields per slot, as in the production cache.
        self.slots = min(math.prod(self.grid), self.limit // (8 * self.volume))
        if self.slots < 1:
            raise ValueError("Field cache must fit at least one 16 MiB tracing tile")
        self.data = [np.empty(self.slots * self.volume, a.dtype) for a in self.sources]
        self.index = np.full(math.prod(self.grid), -1, "i4")
        self.slot_tiles = np.full(self.slots, -1, "i8")
        self.used = self.hand = self.loads = self.evictions = 0
        self.pinned = set()
        self.token = next(_TOKENS)
        _CACHES[self.token] = self

    def region(self, tile):
        z, rem = divmod(int(tile), self.grid[1] * self.grid[2])
        y, x = divmod(rem, self.grid[2])
        lo = np.array((z, y, x)) * self.chunk
        hi = np.minimum(lo + self.chunk, self.shape)
        return slices(lo, hi), slices(np.zeros(3, "i8"), hi - lo)

    def load(self, tile):
        tile = int(tile)
        if self.index[tile] >= 0:
            return int(self.index[tile])
        if self.used < self.slots:
            slot = self.used
            self.used += 1
        else:
            for _ in range(self.slots):
                slot = self.hand
                self.hand = (self.hand + 1) % self.slots
                if int(self.slot_tiles[slot]) not in self.pinned:
                    break
            else:
                raise RuntimeError("All tracing cache slots are pinned")
            old = int(self.slot_tiles[slot])
            region, inner = self.region(old)
            for array, data in zip(self.sources[4:], self.data[4:]):
                array[region] = data[slot * self.volume : (slot + 1) * self.volume].reshape(
                    (self.chunk,) * 3
                )[inner]
            self.index[old] = -1
            self.evictions += 1
        self.slot_tiles[slot] = tile
        region, inner = self.region(tile)
        for array, data in zip(self.sources, self.data):
            block = data[slot * self.volume : (slot + 1) * self.volume].reshape((self.chunk,) * 3)
            block.fill(0)
            block[inner] = array[region]
        self.index[tile] = slot
        self.loads += 1
        return slot

    def arrays(self):
        if self.chunk not in _FIELD_TYPES:
            _FIELD_TYPES[self.chunk] = (
                _field_type(types.uint8, self.chunk),
                _field_type(types.uint16, self.chunk),
            )
        u8, u16 = _FIELD_TYPES[self.chunk]
        return [
            (u16 if a.dtype == np.uint16 else u8)(
                a, self.index, self.shape, self.grid, self.chunk, self.token, i >= 4
            )
            for i, a in enumerate(self.data)
        ]

    def pin_region(self, region, size=1024):
        grid = tuple((n + size - 1) // size for n in self.shape)
        z, rem = divmod(int(region), grid[1] * grid[2])
        y, x = divmod(rem, grid[2])
        lo = np.array((z, y, x)) * size
        hi = np.minimum(lo + size, self.shape)
        ids = [
            (z * self.grid[1] + y) * self.grid[2] + x
            for z, y, x in itertools.product(
                *(
                    range(int(a // self.chunk), int((b + self.chunk - 1) // self.chunk))
                    for a, b in zip(lo, hi)
                )
            )
        ]
        self.pinned.clear()
        # A small cache remains valid: demand paging replaces region pinning.
        if len(ids) < self.slots or len(ids) == self.slots == math.prod(self.grid):
            self.pinned.update(ids)
            for tile in ids:
                self.load(tile)

    def presence_snapshot(self, points):
        q = points[:, ::-1]
        keys = set()
        for delta in itertools.product((-2, 2), repeat=3):
            ij = np.floor(np.clip(q + np.array(delta), 0, np.array(self.shape) - 1)).astype("i8")
            ij //= self.chunk
            keys.update(
                map(int, np.unique((ij[:, 0] * self.grid[1] + ij[:, 1]) * self.grid[2] + ij[:, 2]))
            )
        if len(keys) * self.volume > self.limit:
            raise RuntimeError("Smoothing snapshot exceeds --field-cache-mib; increase that limit")
        data = np.zeros(len(keys) * self.volume, "u1")
        index = np.full(len(self.index), -1, "i4")
        for slot, tile in enumerate(sorted(keys)):
            target = data[slot * self.volume : (slot + 1) * self.volume]
            resident = int(self.index[tile])
            if resident >= 0:
                target[:] = self.data[1][resident * self.volume : (resident + 1) * self.volume]
            else:
                region, inner = self.region(tile)
                target.reshape((self.chunk,) * 3)[inner] = self.sources[1][region]
            index[tile] = slot
        return StaticPresence(data, index, self.shape, self.grid, self.chunk)

    def close(self):
        # Restart always rebuilds coverage from committed raw curves. Deliberately
        # do not persist an uncommitted tail as if it were a checkpoint.
        _CACHES.pop(self.token, None)


def sample_tracing(tracing, points, chunk=64):
    """Use the dense backend's SciPy interpolator on bounded one-voxel-halo tiles."""
    shape = np.array(tracing.shape[1:])
    q = points[:, ::-1]
    # Clip only the tile address. Preserve the original sampling coordinate so
    # SciPy's nearest-boundary arithmetic is the same as on the full array.
    ij = np.floor(np.clip(q, 0, shape - 1)).astype("i8") // chunk
    grid = (shape + chunk - 1) // chunk
    keys = (ij[:, 0] * grid[1] + ij[:, 1]) * grid[2] + ij[:, 2]
    order = np.argsort(keys, kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1, len(order)]
    result = np.empty((2, len(q)), "f4")
    for a, b in zip(cuts[:-1], cuts[1:]):
        if a == b:
            continue
        ids = order[a:b]
        lo = ij[ids[0]] * chunk
        hi = np.minimum(lo + chunk + 1, shape)
        block = tracing[(slice(None),) + slices(lo, hi)]
        for channel in range(2):
            result[channel, ids] = map_coordinates(
                block[channel],
                (q[ids] - lo).T,
                order=1,
                prefilter=False,
                mode="nearest",
                output="f4",
            )
    return result
