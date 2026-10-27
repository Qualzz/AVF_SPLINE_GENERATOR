import hashlib
import json
import sqlite3

import numpy as np
import pytest
from numba import njit, set_num_threads
from scipy.ndimage import map_coordinates

from avf_spline_generator import core
from avf_spline_generator.cli import parser, run
from avf_spline_generator.compressed import ResidentFields, sample_tracing, work_array
from avf_spline_generator.prepare import prepare_seeds

set_num_threads(2)


def fields(tmp_path, shape=(25, 18, 35), chunk=8, slots=2):
    rng = np.random.default_rng(17)
    names = ("centrality", "presence", "nx", "ny", "coverage", "coverage_nx", "coverage_ny")
    arrays, expected = [], []
    for i, name in enumerate(names):
        dtype = "u2" if i == 0 else "u1"
        a = work_array(tmp_path, name, dtype, shape, create=True)
        b = rng.integers(0, 256, shape, dtype=dtype) if i < 4 else np.zeros(shape, dtype)
        a[:] = b
        arrays.append(a)
        expected.append(b)
    cache = ResidentFields(
        arrays[0],
        arrays[1],
        arrays[2:4],
        arrays[4:],
        cache_mib=slots * 8 * chunk**3 / 1024**2,
        chunk=chunk,
    )
    return cache, arrays, expected


@njit
def read_write(arrays, coords):
    values = np.empty((len(coords), 7), np.int64)
    cent, pres, nx, ny, c0, c1, c2 = arrays
    for i in range(len(coords)):
        z, y, x = coords[i]
        values[i, 0] = cent[z, y, x]
        values[i, 1] = pres[z, y, x]
        values[i, 2] = nx[z, y, x]
        values[i, 3] = ny[z, y, x]
        c0[z, y, x] = i % 200 + 1
        c1[z, y, x] = i % 200 + 2
        c2[z, y, x] = i % 200 + 3
        values[i, 4] = c0[z, y, x]
        values[i, 5] = c1[z, y, x]
        values[i, 6] = c2[z, y, x]
    return values


def test_lossless_eviction_global_addressing_and_pin(tmp_path):
    cache, arrays, expected = fields(tmp_path)
    coords = np.array([(0, 0, 0), (24, 17, 34), (8, 8, 8), (16, 0, 32), (1, 1, 1)], "i8")
    try:
        got = read_write(tuple(cache.arrays()), coords)
        for i, q in enumerate(coords):
            for c in range(4):
                assert got[i, c] == expected[c][tuple(q)]
            for c in range(3):
                expected[c + 4][tuple(q)] = i % 200 + c + 1
        assert cache.evictions > 0
        cache.pin_region(0, size=8)
        pinned = int(cache.index[0])
        wrapped = cache.arrays()
        for q in coords[::-1]:
            for c in range(7):
                assert wrapped[c][tuple(q)] == expected[c][tuple(q)]
            assert cache.index[0] == pinned
        assert sum(a.nbytes for a in cache.data) <= cache.limit
    finally:
        cache.close()


def test_sparse_teravoxel_shape_does_not_allocate_dense_storage(tmp_path):
    # One trillion logical voxels. Only one populated chunk is written.
    array = work_array(tmp_path, "presence", "u1", (10000, 10000, 10000), create=True)
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 4096
    array[9999, 9999, 9999] = 255
    assert array[0, 0, 0] == 0 and array[9999, 9999, 9999] == 255
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 20000
    array[9999, 9999, 9999] = 0
    assert array.nchunks_initialized == 0


def test_window_sampling_exact_at_boundaries(tmp_path):
    rng = np.random.default_rng(9)
    dense = rng.integers(0, 256, (2, 31, 27, 153), dtype="u1")
    stored = work_array(tmp_path, "tracing", "u1", dense.shape, create=True)
    stored[:] = dense
    points = rng.uniform(-3, (157, 32, 36), (900, 3)).astype("f4")
    points = np.vstack(
        [points, [[63.999, 2, 2], [64, 26, 30], [127.5, 2, 3], [152, 26, 30]]]
    ).astype("f4")
    expected = np.array(
        [
            map_coordinates(
                a, points[:, ::-1].T, order=1, prefilter=False, mode="nearest", output="f4"
            )
            for a in dense
        ]
    )
    assert np.array_equal(sample_tracing(stored, points), expected)


def test_parallel_smoothing_immutable_snapshot_matches_dense(tmp_path):
    cache, arrays, expected = fields(tmp_path, shape=(40, 24, 24), chunk=8)
    arrays[1][:] = 255
    expected[1][:] = 255
    points = np.array([[10 + 0.1 * np.sin(z), 10, z] for z in np.linspace(1, 38, 80)], "f4")
    offsets = np.array([0, len(points)], "i8")
    # Snapshot has its own bound; allow all relevant tiles, keep runtime cache tiny.
    cache.limit = 1024 * 1024
    snapshot = cache.presence_snapshot(points)
    try:
        a = core.fit_and_resample_splines(points, offsets, expected[1], min_final_len=0)
        b = core.fit_and_resample_splines(points, offsets, snapshot, min_final_len=0)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
    finally:
        cache.close()


def test_seed_order_identical_with_ties_and_empty_volume(tmp_path):
    rng = np.random.default_rng(5)
    for empty in (False, True):
        shape = (33, 17, 71)
        cent = rng.integers(0, 5, shape, dtype="u2")
        seeds = rng.integers(0, 3, shape, dtype="u1") if not empty else np.zeros(shape, "u1")
        d, c = tmp_path / f"d{empty}", tmp_path / f"c{empty}"
        d.mkdir()
        c.mkdir()
        cc = work_array(c, "centrality", "u2", shape, create=True)
        cc[:] = cent
        ss = work_array(c, "seeds", "u1", shape, create=True)
        ss[:] = seeds
        a, ma = prepare_seeds(cent, seeds, d, region_size=16)
        b, mb = prepare_seeds(cc, ss, c, region_size=16)
        assert np.array_equal(a, b) and ma == mb


def args_for(path, work, output, storage, resume=False):
    args = [
        "--predictions",
        str(path),
        "--voxel-size-um",
        "8.64",
        "--work-dir",
        str(work),
        "--output",
        str(output),
        "--threads",
        "2",
        "--storage",
        storage,
        "--field-cache-mib",
        "16",
    ]
    return parser().parse_args(args + (["--resume"] if resume else []))


def fixture_predictions(path):
    # Two disjoint families, both crossing the 128 and 256 tracing-tile seams.
    z, y, x = np.ogrid[:96, :48, :600]
    arrays = []
    for cz, cy in ((28, 12), (68, 34)):
        signal = np.exp(-((z - cz) ** 2 + (y - cy) ** 2) / 9) * np.ones((1, 1, 600))
        signal[:, :, :8] = 0
        signal[:, :, -8:] = 0
        arrays.append(np.rint(signal * 255).astype("u1"))
    np.save(path, np.stack(arrays))


def rows(path):
    with sqlite3.connect(path) as db:
        assert db.execute("pragma integrity_check").fetchone() == ("ok",)
        return {
            name: db.execute(f"SELECT * FROM {name} ORDER BY id").fetchall()
            for name in ("fibers", "blocks", "block_bounds")
        }


def test_pipeline_dense_compressed_and_interrupted_resume(tmp_path, monkeypatch):
    import avf_spline_generator.tracing as tracing

    path = tmp_path / "input.npy"
    fixture_predictions(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    dense_work, compressed_work = tmp_path / "dense", tmp_path / "compressed"
    dense_out, compressed_out = tmp_path / "dense.afv", tmp_path / "compressed.afv"
    run(args_for(path, dense_work, dense_out, "dense"))
    original = tracing.save_json
    interrupted = []

    def commit_then_interrupt(p, state):
        original(p, state)
        if (
            p.name == "progress.json"
            and state["parts"] >= 1
            and not state["complete"]
            and not interrupted
        ):
            interrupted.append(True)
            raise KeyboardInterrupt

    monkeypatch.setattr(tracing, "save_json", commit_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(args_for(path, compressed_work, compressed_out, "compressed"))
    monkeypatch.setattr(tracing, "save_json", original)
    # Simulate arbitrary mutable coverage left by an uncommitted batch. Restart
    # must discard it and reconstruct only the committed raw curves.
    for name in ("coverage", "coverage_nx", "coverage_ny"):
        dirty = work_array(compressed_work / "family-1", name, "u1", (48, 24, 300))
        dirty[:] = 255
    resumed = args_for(path, compressed_work, compressed_out, "compressed", resume=True)
    resumed.field_cache_mib = 32  # Cache size may change on resume; geometry must not.
    run(resumed)
    assert rows(dense_out) == rows(compressed_out)
    assert len(rows(compressed_out)["fibers"]) >= 2
    for family in (1, 2):
        d, c = dense_work / f"family-{family}", compressed_work / f"family-{family}"
        assert json.loads((d / "progress.json").read_text()) == json.loads(
            (c / "progress.json").read_text()
        )
        for part in d.glob("part-*.npz"):
            with np.load(part) as a, np.load(c / part.name) as b:
                assert a.files == b.files
                for name in a.files:
                    assert np.array_equal(a[name], b[name]), name
    for fibre in rows(compressed_out)["fibers"]:
        # min_x/max_x columns; paths must actually cross chunk boundaries.
        assert fibre[6] - fibre[5] > 500
    assert not list(compressed_work.rglob("presence.npy"))
    assert not list(compressed_work.rglob("centrality.npy"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="settings changed"):
        run(args_for(path, compressed_work, tmp_path / "bad.afv", "dense", resume=True))


def test_resume_legacy_dense_manifest(tmp_path):
    path = tmp_path / "input.npy"
    np.save(path, np.zeros((2, 16, 16, 16), "u1"))
    work = tmp_path / "dense"
    run(args_for(path, work, tmp_path / "first.afv", "dense"))
    manifest = json.loads((work / "run.json").read_text())
    manifest["version"] = "0.1.0"
    del manifest["storage"]
    (work / "run.json").write_text(json.dumps(manifest))
    run(args_for(path, work, tmp_path / "second.afv", "dense", resume=True))
    assert rows(tmp_path / "first.afv") == rows(tmp_path / "second.afv")


def test_continuous_path_crosses_1024_seed_region(tmp_path):
    shape = (12, 12, 1100)
    z, y, _ = np.indices(shape)
    ridge = np.exp(-((z - 6) ** 2 + (y - 6) ** 2) / 3)
    dense = [
        np.rint(ridge * 2000).astype("u2"),
        np.rint(ridge * 255).astype("u1"),
        np.full(shape, 255, "u1"),
        np.full(shape, 128, "u1"),
    ]
    dense += [np.zeros(shape, "u1") for _ in range(3)]
    names = ("centrality", "presence", "nx", "ny", "coverage", "coverage_nx", "coverage_ny")
    stored = []
    for name, a in zip(names, dense):
        b = work_array(tmp_path, name, a.dtype, shape, create=True)
        b[:] = a
        stored.append(b)
    seeds = (np.array([6, 6]), np.array([6, 6]), np.array([512, 1050]))

    def trace(arrays):
        points = np.empty((20000, 3), "f4")
        tangents = np.empty_like(points)
        offsets = np.zeros(100, "i8")
        cent, pres, nx, ny, *cov = arrays
        end, count = core.trace_seed_range(
            cent, pres, nx, ny, *seeds, *cov, points, tangents, offsets, 0, 0, 0, 2, n_pass1=2
        )
        return points[:end].copy(), tangents[:end].copy(), offsets[: count + 1].copy()

    expected = trace(dense)
    cache = ResidentFields(stored[0], stored[1], stored[2:4], stored[4:], cache_mib=32)
    try:
        cache.pin_region(0)
        got = trace(cache.arrays())
        assert all(np.array_equal(a, b) for a, b in zip(expected, got))
        assert len(got[2]) == 2  # Second seed is covered by the first global path.
        assert got[0][:, 0].min() < 128 and got[0][:, 0].max() > 1024
        assert cache.evictions > 0
    finally:
        cache.close()
