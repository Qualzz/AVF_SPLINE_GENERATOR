"""Prepare disk-backed tracing fields and spatially grouped seeds."""

import itertools
import json
import logging

import numpy as np
from scipy.ndimage import gaussian_filter

from . import core
from .storage import save_json, slices, tiles

LOG = logging.getLogger(__name__)


def mapped(folder, name, dtype, shape, *, create=False):
    path = folder / (name + ".npy")
    if create:
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(shape))
    result = np.load(path, mmap_mode="r+", allow_pickle=False)
    if result.shape != tuple(shape) or result.dtype != np.dtype(dtype):
        raise ValueError(f"Invalid work array: {path.name}")
    return result


def prepare_tracing(native, supplied, folder):
    shape = tuple(n // 2 for n in native.shape)
    marker = folder / "tracing-ready.json"
    if marker.exists():
        return mapped(folder, "tracing", "u1", (2,) + shape)
    tracing = mapped(folder, "tracing", "u1", (2,) + shape, create=True)
    for lo, hi in tiles(shape, 64):
        if supplied is not None:
            block = supplied.read(lo, hi)
        else:
            block = native.read(lo * 2, hi * 2, quantized=False).astype("f4")
            pooled = np.zeros((2,) + tuple(hi - lo), "f4")
            # Match the eight additions in a 2x2x2 average-pooling kernel.
            # A multi-axis mean may round differently at uint8 half steps.
            for dz, dy, dx in itertools.product((0, 1), repeat=3):
                pooled += block[:, dz::2, dy::2, dx::2]
            block = pooled / 8
            if native.dtype.kind == "f":
                block *= 255
            block = np.rint(block).clip(0, 255).astype("u1")
        tracing[(slice(None),) + slices(lo, hi)] = block
    tracing.flush()
    save_json(marker, {"complete": True})
    return tracing


def tensor_axes(block, inner):
    presence = block[0].astype("u2") + block[1]
    dominant = block[1] > block[0]
    shape = presence[inner].shape
    xx = np.full(shape, 128, "u1")
    yy = xx.copy()
    for family in (0, 1):
        ids = np.flatnonzero((presence[inner] >= 40) & (dominant[inner] == family))
        if not len(ids):
            continue
        grad = np.gradient(gaussian_filter(block[family].astype("f4") / 255, 0.9))
        tensor = np.empty((len(ids), 3, 3), "f4")
        for u in range(3):
            for v in range(u, 3):
                values = gaussian_filter(grad[u] * grad[v], 1.5)[inner].reshape(-1)[ids]
                tensor[:, u, v] = values
                tensor[:, v, u] = values
        for start in range(0, len(ids), 250000):
            _, vectors = np.linalg.eigh(tensor[start : start + 250000])
            xyz = vectors[:, :, 0][:, ::-1]
            xyz *= np.where(xyz[:, 2:3] < 0, -1, 1)
            indices = ids[start : start + 250000]
            xx.reshape(-1)[indices] = np.rint(xyz[:, 0] * 127 + 128).clip(1, 255).astype("u1")
            yy.reshape(-1)[indices] = np.rint(xyz[:, 1] * 127 + 128).clip(1, 255).astype("u1")
    return xx, yy


def prepare_axes(tracing, folder):
    shape = tracing.shape[1:]
    marker = folder / "axes-ready.json"
    create = not marker.exists()
    axes = [mapped(folder, name, "u1", shape, create=create) for name in ("nx", "ny")]
    if not create:
        return axes
    jobs = list(tiles(shape, 256))
    for index, (start, end) in enumerate(jobs):
        lo, hi = np.maximum(0, start - 16), np.minimum(shape, end + 16)
        inner = slices(start - lo, end - lo)
        block = tracing[(slice(None),) + slices(lo, hi)]
        xx, yy = tensor_axes(block, inner)
        for output, values in zip(axes, (xx, yy)):
            output[slices(start, end)] = values
        LOG.info("Direction field: %d/%d tiles", index + 1, len(jobs))
    for output in axes:
        output.flush()
    save_json(marker, {"complete": True})
    return axes


def centrality_block(presence, nx, ny, device):
    if device == "cpu":
        return core.stage1_centrality_and_ridge(presence, nx, ny, 2)
    from numba import cuda

    from .cuda import centrality_kernel, seed_kernel

    arrays = [cuda.to_device(np.ascontiguousarray(a)) for a in (presence, nx, ny)]
    cent = cuda.to_device(np.zeros(presence.shape, "u2"))
    seeds = cuda.to_device(np.zeros(presence.shape, "u1"))
    blocks = tuple((n + 7) // 8 for n in presence.shape[::-1])
    centrality_kernel[blocks, (8, 8, 8)](*arrays, cent, seeds, 2)
    seed_kernel[blocks, (8, 8, 8)](*arrays, cent, seeds, 2)
    return cent.copy_to_host(), seeds.copy_to_host()


def prepare_family(tracing, axes, family, folder, device):
    folder.mkdir(exist_ok=True)
    shape = tracing.shape[1:]
    marker = folder / "fields-ready.json"
    create = not marker.exists()
    pres = mapped(folder, "presence", "u1", shape, create=create)
    cent = mapped(folder, "centrality", "u2", shape, create=create)
    seeds = mapped(folder, "seeds", "u1", shape, create=create)
    if not create:
        return pres, cent, seeds
    for lo, hi in tiles(shape, 128):
        block = tracing[(slice(None),) + slices(lo, hi)]
        own, other = block[family - 1], block[2 - family]
        p = np.minimum(own.astype("u2") + other, 255).astype("u1")
        p[own < other] = 0
        pres[slices(lo, hi)] = p
    pres.flush()
    # Even tile origins preserve checkerboard seed parity. Eight halo voxels
    # cover both the centrality stencil and the subsequent ridge test.
    jobs = list(tiles(shape, 128))
    for index, (start, end) in enumerate(jobs):
        lo, hi = np.maximum(0, start - 8), np.minimum(shape, end + 8)
        region, inner = slices(lo, hi), slices(start - lo, end - lo)
        cc, ss = centrality_block(pres[region], axes[0][region], axes[1][region], device)
        cent[slices(start, end)] = cc[inner]
        seeds[slices(start, end)] = ss[inner]
        LOG.info("%s centrality: %d/%d tiles", "VH"[family - 1], index + 1, len(jobs))
    cent.flush()
    seeds.flush()
    save_json(marker, {"complete": True})
    return pres, cent, seeds


def prepare_seeds(cent, seeds, folder, region_size=1024):
    marker = folder / "seed-order.json"
    if marker.exists():
        meta = json.loads(marker.read_text())
        return mapped(folder, "coordinates", "i4", (meta["total"], 3)), meta
    shape = seeds.shape
    counts = [0, 0]
    for z in range(0, shape[0], 16):
        slab = seeds[z : z + 16]
        for i, priority in enumerate((2, 1)):
            counts[i] += int(np.count_nonzero(slab == priority))
    total = sum(counts)
    coordinates = mapped(folder, "coordinates", "i4", (total, 3), create=True)
    raw = np.empty((total, 3), "i4")
    values = np.empty(total, "u2")
    at = 0
    for count, priority in zip(counts, (2, 1)):
        begin = at
        for z in range(0, shape[0], 16):
            zz, yy, xx = np.where(seeds[z : z + 16] == priority)
            n = len(zz)
            raw[at : at + n] = np.column_stack((zz + z, yy, xx))
            values[at : at + n] = cent[zz + z, yy, xx]
            at += n
        # Sort each seed priority by descending centrality.
        order = np.argsort(values[begin : begin + count])[::-1]
        coordinates[begin:at] = raw[begin:at][order]
    del raw, values
    grid = (np.array(shape) + region_size - 1) // region_size
    regions = coordinates.astype("i8") // region_size
    keys = (regions[:, 0] * grid[1] + regions[:, 1]) * grid[2] + regions[:, 2]
    del regions
    order = np.argsort(keys, kind="stable")
    region_counts = np.bincount(keys, minlength=int(np.prod(grid)))
    first_counts = np.bincount(keys[: counts[0]], minlength=len(region_counts))
    # Copy through a separate array so grouping cannot overwrite unread seeds.
    grouped = mapped(folder, "grouped", "i4", (total, 3), create=True)
    for start in range(0, total, 1000000):
        grouped[start : start + 1000000] = coordinates[order[start : start + 1000000]]
    for start in range(0, total, 1000000):
        coordinates[start : start + 1000000] = grouped[start : start + 1000000]
    coordinates.flush()
    del grouped
    (folder / "grouped.npy").unlink()
    starts = np.r_[0, np.cumsum(region_counts)]
    meta = {"total": total, "starts": starts.tolist(), "primary_counts": first_counts.tolist()}
    save_json(marker, meta)
    return coordinates, meta
