"""Continuous tracing with independent family coverage and restartable batches."""

import json
import logging
import time

import numpy as np
from scipy.ndimage import map_coordinates

from . import core
from .prepare import mapped, prepare_family, prepare_seeds
from .storage import save_json, save_part

LOG = logging.getLogger(__name__)
BATCH = 16


def valid_segments(points, tracing, family, native):
    delta = np.diff(points, axis=0)
    counts = np.maximum(1, np.ceil(np.linalg.norm(delta, axis=1) / 0.25).astype(int))
    valid = np.ones(len(delta), bool)
    for n in np.unique(counts):
        all_ids = np.flatnonzero(counts == n)
        # Bound native sampling allocations even for a long fibre.
        for start in range(0, len(all_ids), max(1, 32768 // (int(n) + 1))):
            ids = all_ids[start : start + max(1, 32768 // (int(n) + 1))]
            t = np.linspace(0, 1, int(n) + 1, dtype="f4")
            q = (points[ids, None] + delta[ids, None] * t[None, :, None]).reshape(-1, 3)
            values = [
                map_coordinates(
                    field, q[:, ::-1].T, order=1, prefilter=False, mode="nearest", output="f4"
                )
                for field in tracing
            ]
            good = values[family - 1] + 1e-4 >= values[2 - family]
            v = native.sample(q * 2 + 0.5)
            good &= v[:, family - 1] + 1e-4 >= v[:, 2 - family]
            valid[ids] = good.reshape(len(ids), -1).all(axis=1)
    return valid


def pack(curves):
    points = np.concatenate(curves).astype("f4") if curves else np.empty((0, 3), "f4")
    offsets = np.r_[0, np.cumsum([len(curve) for curve in curves])].astype("i8")
    return points, offsets


def finish(points, offsets, tracing, presence, family, native):
    pieces = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        p = points[a:b]
        valid = valid_segments(p, tracing, family, native)
        edges = np.diff(np.r_[False, valid, False].astype("i1"))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            pieces.append(p[start : end + 1].copy())
    raw, raw_offsets = pack(pieces)
    if not pieces:
        return raw, raw_offsets
    smoothed, counts = core.fit_and_resample_splines(
        raw,
        raw_offsets,
        presence,
        min_final_len=0.0,
        target_step=0.8,
        smooth_iters=10,
        max_dev=0.3,
    )
    ids = np.flatnonzero(counts >= 2)
    fitted_offsets = np.r_[0, np.cumsum(counts[ids])].astype("i8")
    fitted, _ = core.evaluate_catmull_rom_splines(
        smoothed,
        raw_offsets,
        ids,
        fitted_offsets,
        shape=presence.shape,
    )
    for index, source in enumerate(ids):
        a, b = fitted_offsets[index : index + 2]
        candidate = fitted[a:b]
        if valid_segments(candidate, tracing, family, native).all():
            pieces[source] = candidate.copy()
    return pack(pieces)


def trace_family(tracing, axes, family, folder, native, device):
    progress_path = folder / "progress.json"
    state = {"next_seed": 0, "parts": 0, "complete": False}
    if progress_path.exists():
        state = json.loads(progress_path.read_text())
        if state["complete"]:
            return
    presence, cent, seeds = prepare_family(tracing, axes, family, folder, device)
    coordinates, catalog = prepare_seeds(cent, seeds, folder)
    if not 0 <= state["next_seed"] <= catalog["total"]:
        raise ValueError("Invalid saved seed position")
    coverage = [
        mapped(folder, n, "u1", presence.shape, create=True)
        for n in ("coverage", "coverage_nx", "coverage_ny")
    ]
    # Rebuild from committed raw curves. Mutable coverage may contain an
    # interrupted batch and must never be used as the restart checkpoint.
    for index in range(state["parts"]):
        with np.load(folder / f"part-{index:08d}.npz", allow_pickle=False) as data:
            points, tangents, offsets = data["raw_points"], data["tangents"], data["raw_offsets"]
        for a, b in zip(offsets[:-1], offsets[1:]):
            core.mark_curve_coverage(
                *coverage,
                cent,
                presence,
                *axes,
                points[a:b],
                tangents[a:b],
                int(b - a),
                2.05,
            )
    if state["parts"]:
        LOG.info("%s: restored %d committed batches", "VH"[family - 1], state["parts"])
    # Each seed can produce at most 65,537 points before geometry splitting;
    # splitting shares endpoints, so reserve twice that bound.
    points = np.empty((BATCH * 65537 * 2 + 1, 3), "f4")
    tangents = np.empty_like(points)
    offsets = np.empty(BATCH * 65537 + 1, "i8")
    last_commit = last_log = time.monotonic()
    for region, (start, end) in enumerate(zip(catalog["starts"][:-1], catalog["starts"][1:])):
        begin = max(start, state["next_seed"])
        if begin >= end:
            continue
        primary_end = start + catalog["primary_counts"][region]
        # These are views of one global coordinate catalogue. Coverage and
        # prediction arrays cover the entire volume, including other regions.
        sz, sy, sx = (coordinates[:, i] for i in range(3))
        while begin < end:
            stop = min(begin + BATCH, end)
            offsets[0] = 0
            cursor, count = core.trace_seed_range(
                cent,
                presence,
                *axes,
                sz,
                sy,
                sx,
                *coverage,
                points,
                tangents,
                offsets,
                0,
                0,
                begin,
                stop,
                min_raw_len=7.5,
                step_size=0.75,
                max_low_steps=3,
                core_r=2.05,
                n_pass1=primary_end,
            )
            if count:
                final_points, final_offsets = finish(
                    points[:cursor],
                    offsets[: count + 1],
                    tracing,
                    presence,
                    family,
                    native,
                )
                save_part(
                    folder / f"part-{state['parts']:08d}.npz",
                    raw_points=points[:cursor],
                    tangents=tangents[:cursor],
                    raw_offsets=offsets[: count + 1],
                    points=final_points,
                    offsets=final_offsets,
                )
                state["parts"] += 1
            state["next_seed"] = stop
            now = time.monotonic()
            if count or stop == end or now - last_commit >= 10:
                save_json(progress_path, state)
                last_commit = now
            if stop == end or now - last_log >= 1:
                LOG.info(
                    "%s seeds: %d/%d; %d curves in batch",
                    "VH"[family - 1],
                    stop,
                    catalog["total"],
                    count,
                )
                last_log = now
            begin = stop
    state["complete"] = True
    save_json(progress_path, state)


def iter_curves(work, families):
    for family in families:
        folder = work / f"family-{family}"
        state = json.loads((folder / "progress.json").read_text())
        if not state["complete"]:
            raise ValueError("Cannot export an unfinished tracing run")
        for index in range(state["parts"]):
            with np.load(folder / f"part-{index:08d}.npz", allow_pickle=False) as data:
                points, offsets = data["points"], data["offsets"]
            for a, b in zip(offsets[:-1], offsets[1:]):
                yield family, points[a:b]
