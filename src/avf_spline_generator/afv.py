"""Stream native-coordinate polylines to the AFV SQLite container."""

import json
import os
import sqlite3
import tempfile
import uuid

import numpy as np

SCHEMA = """
PRAGMA application_id=0x56434643;
PRAGMA user_version=1;
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE fibers(id INTEGER PRIMARY KEY,name TEXT NOT NULL,family TEXT NOT NULL,
 point_count INTEGER NOT NULL CHECK(point_count>=2),length REAL NOT NULL,
 min_x REAL NOT NULL,max_x REAL NOT NULL,min_y REAL NOT NULL,max_y REAL NOT NULL,
 min_z REAL NOT NULL,max_z REAL NOT NULL,annotation TEXT NOT NULL);
CREATE TABLE blocks(id INTEGER PRIMARY KEY,fiber_id INTEGER NOT NULL REFERENCES fibers(id),
 first_segment INTEGER NOT NULL,points BLOB NOT NULL,UNIQUE(fiber_id,first_segment));
CREATE VIRTUAL TABLE block_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);
"""


def dumps(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def bounds(points):
    return tuple(float(v) for pair in zip(points.min(0), points.max(0)) for v in pair)


def export_afv(
    curves,
    destination,
    *,
    voxel_um,
    origin_xyz,
    volume_shape,
    minimum_mm=0.0,
    coordinate_space=None,
):
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".afv-", suffix=".partial", dir=destination.parent)
    os.close(fd)
    db = None
    try:
        db = sqlite3.connect(temporary)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA cache_size=-32768")
        db.executescript(SCHEMA)
        frame = {
            "coordinate_base_shape_zyx": list(volume_shape),
            "vc_open_data_source_coordinate_level": 0,
            "vc_open_data_source_coordinate_scale_factor": 1,
            "vc_open_data_source_original_resolution": voxel_um,
        }
        if coordinate_space:
            frame["vc_open_data_coordinate_space"] = coordinate_space
        root = {
            "type": "vc3d_fiber_collection",
            "version": 1,
            "scale": 1,
            **frame,
            "coordinate_order": "XYZ",
            "coordinate_units": "voxel",
            "voxel_size_mm": voxel_um / 1000,
        }
        meta = {
            "complete": False,
            "uuid": str(uuid.uuid4()),
            "root": root,
            "frame": frame,
            "geometry": "XYZ binary64 little-endian native L0",
            "block_segments": 256,
            "source": "AVF_SPLINE_GENERATOR",
            "tracing": {
                "scale": 2,
                "center_offset": 0.5,
                "origin_xyz": list(origin_xyz),
                "minimum_length_mm": minimum_mm,
                "family_locked": True,
                "ct_exterior_filter": False,
            },
        }
        db.executemany("INSERT INTO metadata VALUES(?,?)", [(k, dumps(v)) for k, v in meta.items()])
        count = point_count = block_count = 0
        for family, points in curves:
            q = points.astype("<f8") * 2 + 0.5 + np.array(origin_xyz, "f8")
            if (
                len(q) < 2
                or not np.isfinite(q).all()
                or (q < 0).any()
                or (q >= np.array(volume_shape[::-1])).any()
            ):
                raise ValueError("Spline contains invalid native coordinates")
            length = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
            mm = length * voxel_um / 1000
            if length <= 0 or mm < minimum_mm:
                continue
            count += 1
            label = {1: "V", 2: "H"}[family]
            name = f"fiber_{count}.json"
            annotation = {
                "type": "vc3d_fiber",
                "version": 1,
                "sequence": count,
                "filename": name,
                "hv_classification": {"manual_tag": label},
                "classification_source": "model_prediction_family_locked_tracing",
                "control_points": [q[0].tolist(), q[-1].tolist()],
                "length_voxels": length,
                "length_mm": mm,
            }
            db.execute(
                "INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (count, name, label, len(q), length, *bounds(q), dumps(annotation)),
            )
            for first in range(0, len(q) - 1, 256):
                block = q[first : first + 257].astype("<f8", copy=False)
                block_count += 1
                db.execute(
                    "INSERT INTO blocks VALUES(?,?,?,?)",
                    (block_count, count, first, block.tobytes()),
                )
                db.execute(
                    "INSERT INTO block_bounds VALUES(?,?,?,?,?,?,?)", (block_count, *bounds(block))
                )
            point_count += len(q)
            if count % 1000 == 0:
                db.commit()
        db.execute("CREATE INDEX fibers_length ON fibers(length DESC,id)")
        summary = {
            "complete": True,
            "fiber_count": count,
            "point_count": point_count,
            "block_count": block_count,
        }
        db.executemany(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)",
            [(k, dumps(v)) for k, v in summary.items()],
        )
        db.commit()
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError("AFV integrity check failed")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("AFV contains invalid block references")
        if db.execute("SELECT rtreecheck('block_bounds')").fetchone()[0] != "ok":
            raise RuntimeError("AFV spatial index check failed")
        db.close()
        db = None
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        # A hard link publishes the complete file without overwriting a path
        # that another process may have created while tracing was running.
        os.link(temporary, destination)
        return summary
    finally:
        if db is not None:
            db.close()
        os.unlink(temporary)
