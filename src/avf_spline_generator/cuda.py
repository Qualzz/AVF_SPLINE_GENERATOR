"""CUDA centrality and ridge detection; tracing remains on the CPU."""

import math

import numpy as np
from numba import cuda


@cuda.jit(device=True, fastmath=False)
def decode_dir(nx_val, ny_val):
    dx = (np.float32(nx_val) - 128.0) / 127.0
    dy = (np.float32(ny_val) - 128.0) / 127.0
    r2 = dx * dx + dy * dy
    if r2 > 1.0:
        inv = np.float32(1.0) / math.sqrt(r2)
        dx *= inv
        dy *= inv
        dz = np.float32(0.0)
    else:
        dz = math.sqrt(np.float32(1.0) - r2)
    return (dx, dy, dz)


@cuda.jit(device=True, fastmath=False)
def trilin_scalar_u8(arr, x, y, z):
    D, H, W = arr.shape
    if x < 0.0:
        x = np.float32(0.0)
    elif x > W - 1.001:
        x = np.float32(W - 1.001)
    if y < 0.0:
        y = np.float32(0.0)
    elif y > H - 1.001:
        y = np.float32(H - 1.001)
    if z < 0.0:
        z = np.float32(0.0)
    elif z > D - 1.001:
        z = np.float32(D - 1.001)
    ix = int(x)
    iy = int(y)
    iz = int(z)
    fx = np.float32(x - ix)
    fy = np.float32(y - iy)
    fz = np.float32(z - iz)
    p00 = (1.0 - fx) * arr[iz, iy, ix] + fx * arr[iz, iy, ix + 1]
    p01 = (1.0 - fx) * arr[iz, iy + 1, ix] + fx * arr[iz, iy + 1, ix + 1]
    p10 = (1.0 - fx) * arr[iz + 1, iy, ix] + fx * arr[iz + 1, iy, ix + 1]
    p11 = (1.0 - fx) * arr[iz + 1, iy + 1, ix] + fx * arr[iz + 1, iy + 1, ix + 1]
    return (1.0 - fz) * ((1.0 - fy) * p00 + fy * p01) + fz * ((1.0 - fy) * p10 + fy * p11)


@cuda.jit(device=True, fastmath=False)
def trilin_scalar_u16(arr, x, y, z):
    D, H, W = arr.shape
    if x < 0.0:
        x = np.float32(0.0)
    elif x > W - 1.001:
        x = np.float32(W - 1.001)
    if y < 0.0:
        y = np.float32(0.0)
    elif y > H - 1.001:
        y = np.float32(H - 1.001)
    if z < 0.0:
        z = np.float32(0.0)
    elif z > D - 1.001:
        z = np.float32(D - 1.001)
    ix = int(x)
    iy = int(y)
    iz = int(z)
    fx = np.float32(x - ix)
    fy = np.float32(y - iy)
    fz = np.float32(z - iz)
    p00 = (1.0 - fx) * arr[iz, iy, ix] + fx * arr[iz, iy, ix + 1]
    p01 = (1.0 - fx) * arr[iz, iy + 1, ix] + fx * arr[iz, iy + 1, ix + 1]
    p10 = (1.0 - fx) * arr[iz + 1, iy, ix] + fx * arr[iz + 1, iy, ix + 1]
    p11 = (1.0 - fx) * arr[iz + 1, iy + 1, ix] + fx * arr[iz + 1, iy + 1, ix + 1]
    return np.float32(0.01) * (
        (1.0 - fz) * ((1.0 - fy) * p00 + fy * p01) + fz * ((1.0 - fy) * p10 + fy * p11)
    )


@cuda.jit(device=True, fastmath=False)
def ortho_basis(ux, uy, uz):
    if abs(uz) < 0.85:
        e1x = -uy
        e1y = ux
        e1z = np.float32(0.0)
    else:
        e1x = np.float32(0.0)
        e1y = -uz
        e1z = uy
    n1 = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    e1x /= n1
    e1y /= n1
    e1z /= n1
    e2x = uy * e1z - uz * e1y
    e2y = uz * e1x - ux * e1z
    e2z = ux * e1y - uy * e1x
    return (e1x, e1y, e1z, e2x, e2y, e2z)


@cuda.jit(fastmath=False)
def centrality_kernel(pres, nx, ny, cent, is_seed, seed_stride):
    x, y, z = cuda.grid(3)
    D, H, W = pres.shape
    if x >= W or y >= H or z >= D:
        return
    cos_a = (
        np.float32(1.0),
        np.float32(0.70710678),
        np.float32(0.0),
        np.float32(-0.70710678),
        np.float32(-1.0),
        np.float32(-0.70710678),
        np.float32(0.0),
        np.float32(0.70710678),
    )
    sin_a = (
        np.float32(0.0),
        np.float32(0.70710678),
        np.float32(1.0),
        np.float32(0.70710678),
        np.float32(0.0),
        np.float32(-0.70710678),
        np.float32(-1.0),
        np.float32(-0.70710678),
    )
    p0 = np.float32(pres[z, y, x])
    if p0 < 64.0:
        return
    ux, uy, uz = decode_dir(nx[z, y, x], ny[z, y, x])
    p_p = trilin_scalar_u8(pres, x + ux, y + uy, z + uz)
    p_m = trilin_scalar_u8(pres, x - ux, y - uy, z - uz)
    p_par = 0.5 * p0 + 0.25 * p_p + 0.25 * p_m
    e1x, e1y, e1z, e2x, e2y, e2z = ortho_basis(ux, uy, uz)
    c_sum = np.float32(0.0)
    for k in range(8):
        dx = cos_a[k] * e1x + sin_a[k] * e2x
        dy = cos_a[k] * e1y + sin_a[k] * e2y
        dz = cos_a[k] * e1z + sin_a[k] * e2z
        for r, wr in ((1.0, 0.5), (1.8, 0.28), (2.7, 0.14), (3.6, 0.06)):
            xx = min(max(x + r * dx, 0.0), W - 1.001)
            yy = min(max(y + r * dy, 0.0), H - 1.001)
            zz = min(max(z + r * dz, 0.0), D - 1.001)
            pr = trilin_scalar_u8(pres, xx, yy, zz)
            ux2, uy2, uz2 = decode_dir(
                nx[int(zz + 0.5), int(yy + 0.5), int(xx + 0.5)],
                ny[int(zz + 0.5), int(yy + 0.5), int(xx + 0.5)],
            )
            dr = abs(ux * ux2 + uy * uy2 + uz * uz2)
            if dr > 0.85:
                c_sum += wr / 8.0 * pr * ((dr - 0.85) / 0.15)
    cent[z, y, x] = np.uint16(min(max((p_par + c_sum) * 100.0 + 0.5, 0.0), 65535.0))


@cuda.jit(fastmath=False)
def seed_kernel(pres, nx, ny, cent, is_seed, seed_stride):
    x, y, z = cuda.grid(3)
    D, H, W = pres.shape
    if x < 1 or y < 1 or z < 1 or (x >= W - 1) or (y >= H - 1) or (z >= D - 1):
        return
    cos_a = (
        np.float32(1.0),
        np.float32(0.70710678),
        np.float32(0.0),
        np.float32(-0.70710678),
        np.float32(-1.0),
        np.float32(-0.70710678),
        np.float32(0.0),
        np.float32(0.70710678),
    )
    sin_a = (
        np.float32(0.0),
        np.float32(0.70710678),
        np.float32(1.0),
        np.float32(0.70710678),
        np.float32(0.0),
        np.float32(-0.70710678),
        np.float32(-1.0),
        np.float32(-0.70710678),
    )
    if pres[z, y, x] < 79:
        return
    if seed_stride == 2 and x + y + z & 1 != 0:
        return
    c0 = np.float32(cent[z, y, x]) * np.float32(0.01)
    ux, uy, uz = decode_dir(nx[z, y, x], ny[z, y, x])
    e1x, e1y, e1z, e2x, e2y, e2z = ortho_basis(ux, uy, uz)
    n_max = 0
    for k in range(4):
        dx = cos_a[k] * e1x + sin_a[k] * e2x
        dy = cos_a[k] * e1y + sin_a[k] * e2y
        dz = cos_a[k] * e1z + sin_a[k] * e2z
        cp = trilin_scalar_u16(cent, x + 1.15 * dx, y + 1.15 * dy, z + 1.15 * dz)
        cm = trilin_scalar_u16(cent, x - 1.15 * dx, y - 1.15 * dy, z - 1.15 * dz)
        cp2 = trilin_scalar_u16(cent, x + 2.0 * dx, y + 2.0 * dy, z + 2.0 * dz)
        cm2 = trilin_scalar_u16(cent, x - 2.0 * dx, y - 2.0 * dy, z - 2.0 * dz)
        if c0 >= cp and c0 >= cm and (c0 >= 0.5 * (cp2 + cm2) + 2.0):
            n_max += 1
    if n_max == 4:
        is_seed[z, y, x] = 2
    elif n_max == 3 and c0 >= 185.0:
        is_seed[z, y, x] = 1
