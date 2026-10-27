# AVF_SPLINE_GENERATOR

Convert horizontal/vertical fibre probability volumes into 3D splines with
medial-ridge tracing. Outputs an **AFV** file with native voxel coordinates,
H/V labels, fibre lengths and a spatial index.

Fibre prediction model: [AVF Fiber 9µm V2 on Hugging Face](https://huggingface.co/Qualzz20/afv_fiber_9um_v2).
Use its V/H probability outputs as input to this generator.

## Installation

Python 3.10 or newer:

```bash
git clone https://github.com/Qualzz/AVF_SPLINE_GENERATOR.git
cd AVF_SPLINE_GENERATOR
python -m venv .venv
source .venv/bin/activate
pip install .
```

For Zarr inputs, use `pip install '.[zarr]'`. NumPy, SciPy, Numba and filelock
are the required dependencies. A GPU is optional.

For full scrolls with limited scratch disk space, install
`pip install '.[compressed]'` and use `--storage compressed` (see below).

## Inputs

- A NumPy `.npy` array or local Zarr v2 array with shape **`(2, Z, Y, X)`**.
- Channel order **V, H** by default; use `--channel-order HV` to reverse it.
- Floating-point probabilities in **[0, 1]**, or **uint8 in [0, 255]**.
  Logits and class-ID masks are not accepted as probability inputs.
- The **native isotropic voxel size**, in micrometres.

The generator traces on a grid at twice the native voxel size. By default,
it averages each 2×2×2 block, then quantizes to uint8. For floating-point inputs,
pooling happens before quantization. Pooling already quantized uint8 values can
produce slightly different results; supply the original prepooled array with
`--tracing-predictions` when it is available. Its spatial shape must be the native
shape divided by two, rounded down. Unpaired planes on odd-sized axes are excluded
from the tracing grid. The native input remains available for family validation.

The input arrays must remain unchanged while a run or its resumed run is active.

## Generate splines

```bash
avf-spline-generator \
  --predictions probabilities.npy \
  --voxel-size-um 9 \
  --output fibers.afv \
  --work-dir spline-work
```

For a Zarr group containing native `semantic` and prepooled `tracing` arrays:

```bash
avf-spline-generator \
  --predictions predictions.zarr --array semantic \
  --tracing-predictions predictions.zarr --tracing-array tracing \
  --voxel-size-um 9 \
  --output fibers.afv \
  --work-dir spline-work
```

For a crop, declare its native **X Y Z** origin and the full volume's **Z Y X** shape:

```bash
avf-spline-generator \
  --predictions crop.npy --voxel-size-um 9 \
  --origin-xyz 1000 2000 3000 \
  --volume-shape-zyx 12000 8000 8000 \
  --output crop-fibers.afv --work-dir crop-work
```

Use `--coordinate-space` to attach a scan identifier to the AFV coordinate frame.
Output positions are transformed as:

```text
native_XYZ = tracing_XYZ * 2 + 0.5 + origin_XYZ
```

## Options

| Option | Default | Purpose |
|---|---|---|
| `--family both/V/H` | `both` | Trace either or both predicted families. |
| `--min-length-mm` | `0` | Inclusive length cutoff at export. |
| `--threads` | Up to 8 | CPU threads for parallel preparation and fitting. |
| `--device cpu/cuda` | `cpu` | Centrality and ridge-detection backend. |
| `--cuda-device` | `0` | Device index within the visible CUDA devices. |
| `--native-cache-mib` | `128` | Native prediction chunk cache. |
| `--storage dense/compressed` | `dense` | Dense NPY or sparse lossless Zarr working fields. |
| `--field-cache-mib` | `1024` | Compressed-mode tracing cache, minimum 16 MiB; not total process RAM. |
| `--resume` | Off | Continue a run in the same work directory. |

`--device cuda` requires an NVIDIA driver and a CUDA toolkit supported by the
installed Numba version. Fibre tracing itself runs on the CPU. CPU and CUDA can
differ slightly near numerical decision thresholds; a resumed run must use the
same backend. The first invocation includes Numba compilation time.

## Tracing behaviour

Directions are estimated from the least-eigenvalue axis of a smoothed structure
tensor of the probability fields. The model's direction head is not required.
Seeds follow medial ridges and are visited by 1024³-voxel regions, with primary
seeds before secondary seeds and descending centrality within each group.

Each seed is traced in both directions through the **global** prediction arrays.
A curve can cross any number of preparation tiles or seed regions. Each family
has its own coverage map to suppress retracing nearby paths of the same direction.

The local tracing rules use approximately 30% presence, 0.75 tracing-voxel steps,
55° turn checks and local self-return checks. Valid portions of a rejected curve
are retained. Raw paths require at least 7.5 tracing voxels of arc length for primary
seeds, or 16 for secondary seeds. These internal eligibility rules still apply when
the export cutoff is zero.

Segments are checked at intervals of at most 0.25 tracing voxels against both the
tracing and native probabilities: the assigned family's probability must be at
least the other family's probability. Failed segments split the curve. Fitting
uses constrained smoothing and Catmull–Rom resampling; if the fitted path fails the
family check, the validated raw path is retained. This probability check is not a
proof of physical fibre identity or sheet membership.

There is no endpoint bridging or final cross-fibre merging. A tracing half-path
has a capacity of 32,768 points; reaching it raises an error rather than exporting
a silently shortened fibre. The default export adds no further length cutoff.

## Large volumes and resuming

### Compressed working fields

Use the same command with `--storage compressed` to avoid full-size dense scratch
arrays. Preparation writes directly into sparse Zarr v2 arrays with lossless
Blosc/Zstandard compression. Empty chunks are omitted. Tracing reads through a
bounded RAM cache and saves evicted coverage blocks without loss. Fibres continue
across chunk boundaries using the same tracing kernels and seed order.

```bash
pip install '.[compressed]'
avf-spline-generator \
  --predictions predictions.zarr --array semantic \
  --tracing-predictions predictions.zarr --tracing-array tracing \
  --voxel-size-um 8.64 \
  --storage compressed --field-cache-mib 16384 \
  --output fibers.afv --work-dir spline-work
```

Use your scan's native voxel size. This runs on one machine; a GPU is optional.
The example allocates up to 16 GiB to the tracing-field cache. The default is
1 GiB; smaller caches may perform more I/O. Parallel smoothing can require an
additional immutable presence snapshot up to the same budget. Preparation and
seed sorting use additional RAM. The compression ratio depends on the data;
no fixed disk reduction is promised.

The seed coordinate catalogue and saved curves still require disk space. Seed
sorting retains the RAM requirements below. See [storage details](docs/storage.md)
for layout, memory accounting, continuation and parity tests.

### Dense working fields

With the default `--storage dense`, working fields are memory-mapped on disk. They require approximately
**18 bytes per tracing voxel** for both families, in addition to input data,
seed catalogues, saved curve batches and the AFV output. A tracing grid contains
about one eighth as many voxels as the native volume. These working files are
uncompressed; allow enough disk space before starting a full-volume run.

Seed sorting uses RAM proportional to the seed count (budget roughly 64 bytes
per seed). OS page caching and structure-tensor preparation use additional RAM.
Memory mapping does not guarantee that a full scroll will fit every machine or
run at the same speed as a small crop.

Completed raw/fitted batches are saved in the work directory. After an interruption,
repeat the command with `--resume`. Coverage is rebuilt from committed raw paths;
an interrupted batch is retraced. Input paths, shapes, settings and file
size/modification-time signatures are checked. These signatures do not replace
the requirement to keep inputs immutable.

Keep `--storage` unchanged when resuming. The compressed cache budget may be
changed on resume. Existing dense work directories from release 0.1.0 remain
supported; converting an in-progress dense work directory is not implemented.

The final AFV is published only after SQLite integrity and spatial-index checks.
An existing output file is never overwritten. To export a different length cutoff,
use the same work directory with `--resume`, a new output path and
`--min-length-mm`. Remove the work directory manually when it is no longer needed.

## Scope

This package consumes predictions; it does not run the neural network or read CT.
The output is prediction-derived geometry. CT-based exterior filtering is not
applied. Model weights, training, surface reconstruction and deployment services
are separate from this tool.

## Validation

`pip install '.[test]'` followed by `python -m pytest -q` runs the storage-parity
and restart tests. The numerical tracing core and AFV writer are unchanged from
the dense release. Validation scope and measurements are recorded in
[docs/validation.md](docs/validation.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
