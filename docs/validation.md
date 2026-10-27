# Validation of compressed storage

The reference is the published dense generator at
[`3c8b73628ca56cb28042b9b2e79439b2e1c693d2`](https://github.com/Qualzz/AVF_SPLINE_GENERATOR/commit/3c8b73628ca56cb28042b9b2e79439b2e1c693d2).
`core.py`, `cuda.py` and `afv.py` are unchanged. Their hashes and the measured
real-data results are in [real-crop-validation.json](real-crop-validation.json).

## Automated regression cases

Eight CPU tests cover:

1. Exact global addressing and mutable coverage round trips with forced eviction
   and a pinned tile, including partial edge tiles.
2. A logical trillion-voxel Zarr field: metadata-only when empty, one small
   compressed chunk when populated, and removal when reset to empty.
3. Exact SciPy interpolation at internal chunk boundaries and outside the volume.
4. Parallel smoothing against an immutable paged snapshot versus the dense field.
5. Identical ordered seeds with tied priorities, multiple regions and no seeds.
6. A full dense/compressed AFV pipeline, interrupted after a committed batch.
   Saved mutable coverage is deliberately corrupted; resume must rebuild it and
   produce identical raw/fitted curves and AFV geometry. The cache budget changes
   on resume. Changing storage backend is rejected.
7. Resuming a dense work directory with the original 0.1.0 manifest format.
8. A continuous path extending past the 1024-voxel seed-region boundary under
   forced eviction. The second seed is suppressed by the first path's global
   coverage. Both paths and coverage decisions match the dense kernel.

Run with `pip install '.[test]'` and `python -m pytest -q`. CI runs the suite on
Linux with Python 3.10 and 3.12. These are correctness and regression tests,
not a full-scroll speed benchmark.

## Real PHerc1447 prediction crop

Tested on 27 October 2026 using checkpoint 29,220 predictions:

- Native Z,Y,X origin: **4090, 6344, 3766**.
- Native shape: **96 × 128 × 384**, at **8.64 µm**.
- Tracing shape: **48 × 64 × 192**, using the original prepooled predictions.
- CPU centrality, two CPU threads, both H and V families.
- **184 final fibres, 9,946 points** in both modes.

Prepared prediction/direction/centrality/seed fields, ordered seeds, raw paths,
fitted paths and all AFV fibre/block/spatial-index records matched exactly. Input
file hashes were unchanged. The comparison is of geometry records, not the
random UUID in each AFV container.

| Work files after completion | Dense | Compressed |
|---|---:|---:|
| Total logical file bytes | 11,204,436 | 2,962,989 |
| Voxel field files within that total | 10,618,752 | 2,377,066 |

That is approximately **3.8× less scratch file data on this crop**. These numbers
include seed catalogues, saved curves and work metadata in the total, but exclude
inputs, output AFVs and filesystem overhead. Coverage is an expendable runtime
cache: resident coverage need not be flushed when the run ends, since restart
rebuilds it from committed curves.

The forced-eviction real-data run used only **16 MiB**, one tracing tile. This
intentionally causes heavy cache thrashing and is not a sensible speed benchmark
or a recommended full-scroll setting. A second check with a 64 MiB budget
held both tiles resident and reproduced the same AFV geometry; its warm run took
2.04 seconds. These small-crop timings do not predict full-scroll throughput.
The JSON records both configurations. See [storage.md](storage.md) for memory guidance.

## Limits of the evidence

The portable backend has not been validated by rerunning a complete scroll.
The real-crop check is not a measurement of full-scroll throughput, peak RAM,
compression ratio or biological correctness. CUDA code is unchanged, but these
storage-parity tests use CPU centrality. Different dependency versions or changing
the centrality backend can affect numerical results independently of storage.

The source of the resident-cache design is the local PHerc1447 production
implementation (`resident_block_store.py`, `benchmark_resident_argon.py`,
`trace_resident_blocks.py`). Deployment-specific source locations, networking,
credentials and multi-GPU validation code are not dependencies of this package.
The portable integration uses Zarr v2 rather than those deployment-specific
compressed file readers. It does not introduce a new fibre tracing method.
