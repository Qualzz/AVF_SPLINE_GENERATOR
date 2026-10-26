"""Command-line entry point."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from filelock import FileLock
from numba import get_num_threads, set_num_threads

from . import __version__
from .afv import export_afv
from .prepare import prepare_axes, prepare_tracing
from .storage import NativeSampler, Predictions, file_signature, save_json
from .tracing import iter_curves, trace_family

LOG = logging.getLogger(__name__)


def parser():
    p = argparse.ArgumentParser(description="Trace H/V probability volumes into AFV splines.")
    p.add_argument("--predictions", type=Path, required=True, help="Native (2,Z,Y,X) NPY or Zarr")
    p.add_argument("--array", help="Array key inside a native Zarr group")
    p.add_argument("--channel-order", choices=("VH", "HV"), default="VH")
    p.add_argument("--tracing-predictions", type=Path, help="Optional prepooled 2x array")
    p.add_argument("--tracing-array", help="Array key inside the prepooled Zarr group")
    p.add_argument("--voxel-size-um", type=float, required=True, help="Isotropic native voxel size")
    p.add_argument("--output", type=Path, required=True, help="Destination .afv")
    p.add_argument("--work-dir", type=Path, required=True, help="Persistent scratch directory")
    p.add_argument("--resume", action="store_true", help="Resume the same immutable inputs")
    p.add_argument("--family", choices=("both", "V", "H"), default="both")
    p.add_argument("--min-length-mm", type=float, default=0, help="AFV export cutoff (inclusive)")
    p.add_argument("--origin-xyz", type=int, nargs=3, default=(0, 0, 0), metavar=("X", "Y", "Z"))
    p.add_argument(
        "--volume-shape-zyx",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        help="Full native domain; required for a nonzero crop origin",
    )
    p.add_argument("--coordinate-space", help="Optional AFV scan identifier")
    p.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Centrality backend; tracing always runs on CPU",
    )
    p.add_argument("--cuda-device", type=int, default=0, help="Visible CUDA device index")
    p.add_argument("--threads", type=int, default=min(8, get_num_threads()))
    p.add_argument("--native-cache-mib", type=int, default=128)
    p.add_argument("--version", action="version", version=__version__)
    return p


def run(args):
    if (
        not np.isfinite(args.voxel_size_um)
        or args.voxel_size_um <= 0
        or not np.isfinite(args.min_length_mm)
        or args.min_length_mm < 0
    ):
        raise ValueError("Voxel size must be positive and minimum length nonnegative")
    if min(args.origin_xyz) < 0 or args.native_cache_mib < 1 or args.cuda_device < 0:
        raise ValueError("Invalid origin, cache size or CUDA device")
    set_num_threads(args.threads)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    native = Predictions(args.predictions, args.array, args.channel_order)
    supplied = None
    if args.tracing_array and not args.tracing_predictions:
        raise ValueError("--tracing-array requires --tracing-predictions")
    if args.tracing_predictions:
        supplied = Predictions(args.tracing_predictions, args.tracing_array, args.channel_order)
        if supplied.shape != tuple(n // 2 for n in native.shape):
            raise ValueError("Prepooled tracing shape must equal native shape // 2")
    elif native.dtype == np.uint8:
        LOG.warning(
            "Pooling quantized uint8 inputs. Supply prepooled predictions to preserve "
            "pooling-before-quantization values."
        )
    if min(native.shape) < 8:
        raise ValueError("Native shape must be at least (8,8,8)")
    if any(n % 2 for n in native.shape):
        LOG.warning("The final unpaired voxel plane on odd-sized axes is excluded from tracing")
    if any(args.origin_xyz) and args.volume_shape_zyx is None:
        raise ValueError("A nonzero origin requires --volume-shape-zyx")
    domain = args.volume_shape_zyx or list(native.shape)
    if any(n + o > d for n, o, d in zip(native.shape, args.origin_xyz[::-1], domain)):
        raise ValueError("Prediction crop lies outside the declared native volume")
    paths = [native.path] + ([supplied.path] if supplied is not None else [])
    work = args.work_dir.resolve()
    output = args.output.resolve()
    for path in paths:
        if path == output or path == work or path in work.parents or work in path.parents:
            raise ValueError("Inputs must be separate from the work directory and output")
        if path.is_dir() and path in output.parents:
            raise ValueError("Output cannot be inside an input Zarr store")
    if args.device == "cuda":
        from numba import cuda

        if not cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu or configure a CUDA toolkit")
        cuda.select_device(args.cuda_device)
    work.mkdir(parents=True, exist_ok=True)
    with FileLock(str(work / ".lock"), timeout=0):
        config = {
            "version": __version__,
            "predictions": str(native.path),
            "array": args.array,
            "tracing_predictions": str(supplied.path) if supplied is not None else None,
            "tracing_array": args.tracing_array,
            "channel_order": args.channel_order,
            "shape": list(native.shape),
            "device": args.device,
            "family": args.family,
            "voxel_size_um": args.voxel_size_um,
            "origin_xyz": list(args.origin_xyz),
            "volume_shape_zyx": list(domain),
            "coordinate_space": args.coordinate_space,
            "input_signatures": [file_signature(path) for path in paths],
        }
        manifest = work / "run.json"
        if manifest.exists():
            if not args.resume:
                raise ValueError("Work directory already contains a run; use --resume")
            if json.loads(manifest.read_text()) != config:
                raise ValueError("Inputs or tracing settings changed; use a new work directory")
        else:
            if args.resume:
                raise ValueError("No run to resume in this work directory")
            if any(p.name != ".lock" for p in work.iterdir()):
                raise ValueError("A new run requires an empty work directory")
            save_json(manifest, config)
        LOG.info("Preparing tracing predictions")
        tracing = prepare_tracing(native, supplied, work)
        axes = prepare_axes(tracing, work)
        sampler = NativeSampler(native, cache_mib=args.native_cache_mib)
        families = (1, 2) if args.family == "both" else ({"V": 1, "H": 2}[args.family],)
        for family in families:
            trace_family(tracing, axes, family, work / f"family-{family}", sampler, args.device)
        summary = export_afv(
            iter_curves(work, families),
            output,
            voxel_um=args.voxel_size_um,
            origin_xyz=args.origin_xyz,
            volume_shape=domain,
            minimum_mm=args.min_length_mm,
            coordinate_space=args.coordinate_space,
        )
        LOG.info(
            "Wrote %d fibres (%d points) to %s",
            summary["fiber_count"],
            summary["point_count"],
            output,
        )
        return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parser().parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        LOG.error("Interrupted. Run the same command with --resume to continue.")
        raise SystemExit(130) from None
    except (ValueError, OSError, RuntimeError) as exc:
        LOG.error("%s", exc)
        raise SystemExit(1) from None
