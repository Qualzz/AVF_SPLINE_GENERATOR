# Compressed working storage

This is a storage backend for the existing generator, not a different fibre
tracing algorithm. `core.py`, the centrality CUDA kernels, the geometry parameters,
seed ordering and the AFV writer are unchanged from release 0.1.0 (commit
`3c8b736`). The implementation ports the resident-tile addressing, coverage
spilling and immutable smoothing snapshots used by the dual-3090 PHerc1447 run.
It replaces that deployment's custom files and machine connections with local
Zarr v2 stores. No FUSE mount, privileged service, second computer or GPU is needed.

## Run on one machine

Install `pip install '.[compressed]'` in the repository, then:

```bash
avf-spline-generator \
  --predictions predictions.zarr --array semantic \
  --tracing-predictions predictions.zarr --tracing-array tracing \
  --voxel-size-um 8.64 \
  --storage compressed --field-cache-mib 16384 \
  --threads 8 \
  --work-dir tracing-work --output fibers.afv
```

8.64 µm is an example: use the actual native voxel size of your scan. Omit the
prepooled-array options if you only have native probabilities; the same pooling
and quantization as the dense backend will be used. CPU centrality is the default;
`--device cuda` remains available for a single supported NVIDIA GPU.

16,384 MiB is a **field-cache budget**, not total process RAM. The default is
1,024 MiB. Smaller caches use the same data and equations but can cause more I/O.
Change this budget on `--resume` if needed; the storage backend itself must stay
the same. Keep the inputs immutable throughout the run.

## What is stored

All full-volume voxel fields are created directly as losslessly compressed Zarr
v2 arrays, with 128³ spatial chunks and Blosc/Zstandard + bitshuffle:

- The two tracing probability channels (one channel per chunk).
- The two encoded direction fields (empty value 128).
- Per-family presence, centrality and seed-class fields.
- The three mutable per-family coverage fields.

Chunks containing only their declared empty value are omitted. The logical shape
still covers the entire scroll, including empty space, but that does not allocate
a dense file. There is no intermediate conversion through a full-size `.npy` copy.
The required disk space depends on occupancy and compressibility; **there is no
fixed compression factor**. Input probabilities, output AFVs and saved curves
also need space.

The coordinate catalogue stays as a dense `(number_of_seeds, 3)` int32 array,
12 bytes per seed. Sorting temporarily creates a second such array. The original
sorting routine also uses RAM proportional to the seed count (roughly 64 bytes
per seed). This backend addresses full-volume scratch storage; it does not make
seed sorting a bounded-memory external sort.

## How a fibre crosses a storage boundary

The existing Numba kernels still read global `(z, y, x)` coordinates. The storage
adapter translates each access to a 128³ tile and a cache slot. A missing tile is
loaded losslessly; evicting a tile saves its mutable coverage to its Zarr store.
Presence, centrality and directions remain read-only during tracing.

A seed region contains 1024³ **tracing** voxels (twice the native voxel size).
When the cache can hold the region plus overflow, its tiles are pinned in RAM.
With less memory, the same algorithm runs with demand paging. A fibre continues
immediately beyond the seed region; storage boundaries do not terminate paths,
create endpoints, bridge gaps or trigger a later merge.

The seven tracing fields occupy 8 bytes per cached voxel. A 128³ tile therefore
uses 16 MiB across these fields. A full 1024³ pinned region uses 8 GiB; choose a
larger cache if you want pinning and room for paths outside it. A small global
index (4 bytes per logical tile) maps tiles to slots.

Tracing is serial because the coverage state is shared. The existing parallel
smoother receives an immutable snapshot of the presence tiles around its curves,
so another read cannot evict data underneath a smoothing worker. This snapshot
may add up to another `--field-cache-mib` of memory. If it exceeds that limit the
run stops with an explicit message; increase the budget and resume. Tensor
preparation, curve buffers, native sampling and seed sorting use additional RAM.

## Restart and reproducibility

Committed raw/fitted batches and their seed cursor remain the restart record.
Coverage is rebuilt from the committed raw curves; partially updated cache tiles
are never trusted as a checkpoint. An interrupted batch is retraced. Completed
preparation phases are reused; a preparation phase interrupted before its ready
marker is rebuilt. Output publication and AFV integrity checks are unchanged.

The dense and compressed modes have the same seed order in this repository.
Tests compare prepared fields, ordered seeds, raw paths, fitted paths and AFV
geometry records with exact equality on the same backend and dependency versions.
AFV whole-file hashes can differ because each export has a new UUID.

This is different from comparing an older globally sorted deployment to a
spatially sorted deployment: changing seed order can change coverage decisions.
Neither storage compression nor these tests establish physical fibre accuracy.
CPU versus CUDA numerical differences near thresholds remain as documented in
the main README; use the same preparation backend when comparing storage modes.

Run the regression tests with:

```bash
pip install '.[test]'
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 python -m pytest -q
```

The tests include global reads and coverage writes under forced eviction,
non-multiple-of-128 edges, immutable parallel smoothing, interpolation at chunk
boundaries, tied seed priorities, empty seed catalogues, a trillion-voxel sparse
array, and a complete dense-versus-compressed AFV run with an injected interruption.
