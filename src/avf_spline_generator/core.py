"""Medial-ridge tracing, local geometry checks and constrained spline fitting."""

import numpy as np
from numba import njit, prange


@njit(inline="always")
def decode_dir(nx_val, ny_val):
    dx = (np.float32(nx_val) - 128.0) / 127.0
    dy = (np.float32(ny_val) - 128.0) / 127.0
    r2 = dx * dx + dy * dy
    if r2 > 1.0:
        inv = np.float32(1.0) / np.sqrt(r2)
        dx *= inv
        dy *= inv
        dz = np.float32(0.0)
    else:
        dz = np.sqrt(np.float32(1.0) - r2)
    return (dx, dy, dz)


@njit(inline="always")
def encode_dir(ux, uy, uz):
    if uz < 0.0:
        ux = -ux
        uy = -uy
        uz = -uz
    b_nx = int(min(max(ux * 127.0 + 128.5, 1.0), 255.0))
    b_ny = int(min(max(uy * 127.0 + 128.5, 1.0), 255.0))
    return (np.uint8(b_nx), np.uint8(b_ny))


@njit(inline="always")
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


@njit(inline="always")
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


@njit(inline="always")
def trilin_dir_aligned(pres, nx, ny, x, y, z, ref_x, ref_y, ref_z):
    D, H, W = pres.shape
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
    vx = np.float32(0.0)
    vy = np.float32(0.0)
    vz = np.float32(0.0)
    for dz_i in range(2):
        wz = 1.0 - fz if dz_i == 0 else fz
        zz = iz + dz_i
        for dy_i in range(2):
            wy = wz * (1.0 - fy if dy_i == 0 else fy)
            yy = iy + dy_i
            for dx_i in range(2):
                w = wy * (1.0 - fx if dx_i == 0 else fx)
                xx = ix + dx_i
                pw = w * (np.float32(pres[zz, yy, xx]) + 10.0)
                ux, uy, uz = decode_dir(nx[zz, yy, xx], ny[zz, yy, xx])
                if ux * ref_x + uy * ref_y + uz * ref_z < 0.0:
                    ux = -ux
                    uy = -uy
                    uz = -uz
                vx += pw * ux
                vy += pw * uy
                vz += pw * uz
    n = np.sqrt(vx * vx + vy * vy + vz * vz)
    if n > 1e-08:
        vx /= n
        vy /= n
        vz /= n
    return (vx, vy, vz)


@njit(inline="always")
def ortho_basis(ux, uy, uz):
    if abs(uz) < 0.85:
        e1x = -uy
        e1y = ux
        e1z = np.float32(0.0)
    else:
        e1x = np.float32(0.0)
        e1y = -uz
        e1z = uy
    n1 = np.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    e1x /= n1
    e1y /= n1
    e1z /= n1
    e2x = uy * e1z - uz * e1y
    e2y = uz * e1x - ux * e1z
    e2z = ux * e1y - uy * e1x
    return (e1x, e1y, e1z, e2x, e2y, e2z)


@njit(inline="always")
def recenter_normal_plane(cent, pres, nx, ny, x, y, z, tx, ty, tz, max_shift, n_iter):
    D, H, W = cent.shape
    e1x, e1y, e1z, e2x, e2y, e2z = ortho_basis(tx, ty, tz)
    cx, cy, cz = (x, y, z)
    u1_arr = np.array(
        [0.0, 0.55, 0.0, -0.55, 0.0, 1.15, 0.81317, 0.0, -0.81317, -1.15, -0.81317, 0.0, 0.81317],
        dtype=np.float32,
    )
    u2_arr = np.array(
        [0.0, 0.0, 0.55, 0.0, -0.55, 0.0, 0.81317, 1.15, 0.81317, 0.0, -0.81317, -1.15, -0.81317],
        dtype=np.float32,
    )
    r2_arr = u1_arr * u1_arr + u2_arr * u2_arr
    exp_arr = np.exp(-0.5 * r2_arr)
    vals = np.empty(13, dtype=np.float32)
    for _ in range(n_iter):
        c_min = np.float32(1000000000.0)
        c_max = np.float32(-1000000000.0)
        for i in range(13):
            px = cx + u1_arr[i] * e1x + u2_arr[i] * e2x
            py = cy + u1_arr[i] * e1y + u2_arr[i] * e2y
            pz = cz + u1_arr[i] * e1z + u2_arr[i] * e2z
            if (
                px < 0.0
                or px >= W - 1.001
                or py < 0.0
                or (py >= H - 1.001)
                or (pz < 0.0)
                or (pz >= D - 1.001)
            ):
                vals[i] = 0.0
            else:
                ux, uy, uz = decode_dir(
                    nx[int(pz + 0.5), int(py + 0.5), int(px + 0.5)],
                    ny[int(pz + 0.5), int(py + 0.5), int(px + 0.5)],
                )
                dot = abs(ux * tx + uy * ty + uz * tz)
                if dot < 0.82:
                    vals[i] = 0.0
                else:
                    vals[i] = trilin_scalar_u16(cent, px, py, pz) * dot
            if vals[i] < c_min:
                c_min = vals[i]
            if vals[i] > c_max:
                c_max = vals[i]
        if c_max - c_min < 0.5:
            break
        w_sum = np.float32(0.0)
        du1 = np.float32(0.0)
        du2 = np.float32(0.0)
        for i in range(13):
            ex = vals[i] - c_min
            w = ex * ex * exp_arr[i]
            w_sum += w
            du1 += w * u1_arr[i]
            du2 += w * u2_arr[i]
        if w_sum > 1e-06:
            du1 /= w_sum
            du2 /= w_sum
            cx += du1 * e1x + du2 * e2x
            cy += du1 * e1y + du2 * e2y
            cz += du1 * e1z + du2 * e2z
    dx = cx - x
    dy = cy - y
    dz = cz - z
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    if dist > max_shift:
        s = max_shift / dist
        cx = x + dx * s
        cy = y + dy * s
        cz = z + dz * s
    return (cx, cy, cz)


@njit(inline="always")
def is_covered_same_dir(cov_cnt, cov_nx, cov_ny, x, y, z, tx, ty, tz):
    D, H, W = cov_cnt.shape
    ix = int(x + 0.5)
    iy = int(y + 0.5)
    iz = int(z + 0.5)
    if ix < 0 or ix >= W or iy < 0 or (iy >= H) or (iz < 0) or (iz >= D):
        return False
    c = cov_cnt[iz, iy, ix]
    if c == 0:
        return False
    if c >= 2:
        return True
    ux, uy, uz = decode_dir(cov_nx[iz, iy, ix], cov_ny[iz, iy, ix])
    return abs(ux * tx + uy * ty + uz * tz) > 0.86


@njit(parallel=True, cache=True)
def stage1_centrality_and_ridge(pres, nx, ny, seed_stride=2):
    D, H, W = pres.shape
    cent = np.zeros((D, H, W), dtype=np.uint16)
    cos_a = np.array(
        [1.0, 0.70710678, 0.0, -0.70710678, -1.0, -0.70710678, 0.0, 0.70710678], dtype=np.float32
    )
    sin_a = np.array(
        [0.0, 0.70710678, 1.0, 0.70710678, 0.0, -0.70710678, -1.0, -0.70710678], dtype=np.float32
    )
    for z in prange(D):
        for y in range(H):
            for x in range(W):
                p0 = np.float32(pres[z, y, x])
                if p0 < 64.0:
                    continue
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
    is_seed = np.zeros((D, H, W), dtype=np.uint8)
    for z in prange(1, D - 1):
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if pres[z, y, x] < 79:
                    continue
                if seed_stride == 2 and x + y + z & 1 != 0:
                    continue
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
    return (cent, is_seed)


@njit(cache=True)
def trace_half_buf(
    cent,
    pres,
    nx,
    ny,
    cov_cnt,
    cov_nx,
    cov_ny,
    x0,
    y0,
    z0,
    tx0,
    ty0,
    tz0,
    pts,
    tans,
    step_size=0.75,
    max_low_steps=3,
):
    D, H, W = pres.shape
    max_steps = len(pts)
    x, y, z = (x0, y0, z0)
    tx, ty, tz = (tx0, ty0, tz0)
    n_pts = 0
    low_pres_run = 0
    cov_run = 0
    for s in range(max_steps):
        mx = x + 0.5 * step_size * tx
        my = y + 0.5 * step_size * ty
        mz = z + 0.5 * step_size * tz
        if (
            mx < 0.25
            or mx >= W - 1.25
            or my < 0.25
            or (my >= H - 1.25)
            or (mz < 0.25)
            or (mz >= D - 1.25)
        ):
            break
        ux, uy, uz = trilin_dir_aligned(pres, nx, ny, mx, my, mz, tx, ty, tz)
        dot_u = ux * tx + uy * ty + uz * tz
        if dot_u < 0.5735764363510462:
            crossed = False
            for look_d in (1.5, 2.2):
                lx = x + look_d * tx
                ly = y + look_d * ty
                lz = z + look_d * tz
                if (
                    lx >= 0.25
                    and lx < W - 1.25
                    and (ly >= 0.25)
                    and (ly < H - 1.25)
                    and (lz >= 0.25)
                    and (lz < D - 1.25)
                ):
                    lp = trilin_scalar_u8(pres, lx, ly, lz)
                    lux, luy, luz = trilin_dir_aligned(pres, nx, ny, lx, ly, lz, tx, ty, tz)
                    if lp >= 72.5 and lux * tx + luy * ty + luz * tz >= 0.5735764363510462:
                        crossed = True
                        break
            if crossed:
                nx_p = x + step_size * tx
                ny_p = y + step_size * ty
                nz_p = z + step_size * tz
                ntx, nty, ntz = (tx, ty, tz)
                ux, uy, uz = (tx, ty, tz)
            else:
                break
        else:
            ntx = 0.45 * tx + 0.55 * ux
            nty = 0.45 * ty + 0.55 * uy
            ntz = 0.45 * tz + 0.55 * uz
            nn = np.sqrt(ntx * ntx + nty * nty + ntz * ntz)
            ntx /= nn
            nty /= nn
            ntz /= nn
            px = x + step_size * ntx
            py = y + step_size * nty
            pz = z + step_size * ntz
            if (
                px < 0.25
                or px >= W - 1.25
                or py < 0.25
                or (py >= H - 1.25)
                or (pz < 0.25)
                or (pz >= D - 1.25)
            ):
                break
            nx_p, ny_p, nz_p = recenter_normal_plane(
                cent, pres, nx, ny, px, py, pz, ntx, nty, ntz, 0.42, 2
            )
        dx = nx_p - x
        dy = ny_p - y
        dz = nz_p - z
        dl = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dl < 0.25 or dl > 1.45:
            break
        dx /= dl
        dy /= dl
        dz /= dl
        if dx * tx + dy * ty + dz * tz < 0.5735764363510462:
            break
        if n_pts >= 4:
            chx = nx_p - pts[n_pts - 4, 0]
            chy = ny_p - pts[n_pts - 4, 1]
            chz = nz_p - pts[n_pts - 4, 2]
            chl = np.sqrt(chx * chx + chy * chy + chz * chz)
            if chl > 1e-05:
                if (chx * dx + chy * dy + chz * dz) / chl < 0.5735764363510462:
                    break
        else:
            sx_d = nx_p - x0
            sy_d = ny_p - y0
            sz_d = nz_p - z0
            sl = np.sqrt(sx_d * sx_d + sy_d * sy_d + sz_d * sz_d)
            if sl > 1e-05:
                if (sx_d * tx0 + sy_d * ty0 + sz_d * tz0) / sl < 0.5735764363510462:
                    break
        if n_pts >= 4:
            v2x = nx_p - pts[n_pts - 4, 0]
            v2y = ny_p - pts[n_pts - 4, 1]
            v2z = nz_p - pts[n_pts - 4, 2]
            l2 = np.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
            if n_pts >= 9:
                v1x = pts[n_pts - 4, 0] - pts[n_pts - 9, 0]
                v1y = pts[n_pts - 4, 1] - pts[n_pts - 9, 1]
                v1z = pts[n_pts - 4, 2] - pts[n_pts - 9, 2]
            else:
                v1x = pts[n_pts - 4, 0] - x0
                v1y = pts[n_pts - 4, 1] - y0
                v1z = pts[n_pts - 4, 2] - z0
            l1 = np.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
            if l1 > 1e-05 and l2 > 1e-05:
                if (v1x * v2x + v1y * v2y + v1z * v2z) / (l1 * l2) < 0.5735764363510462:
                    break
        if n_pts >= 9:
            w2x = nx_p - pts[n_pts - 9, 0]
            w2y = ny_p - pts[n_pts - 9, 1]
            w2z = nz_p - pts[n_pts - 9, 2]
            wl2 = np.sqrt(w2x * w2x + w2y * w2y + w2z * w2z)
            if n_pts >= 19:
                w1x = pts[n_pts - 9, 0] - pts[n_pts - 19, 0]
                w1y = pts[n_pts - 9, 1] - pts[n_pts - 19, 1]
                w1z = pts[n_pts - 9, 2] - pts[n_pts - 19, 2]
            else:
                w1x = pts[n_pts - 9, 0] - x0
                w1y = pts[n_pts - 9, 1] - y0
                w1z = pts[n_pts - 9, 2] - z0
            wl1 = np.sqrt(w1x * w1x + w1y * w1y + w1z * w1z)
            if wl1 > 1e-05 and wl2 > 1e-05:
                if (w1x * w2x + w1y * w2y + w1z * w2z) / (wl1 * wl2) < 0.5735764363510462:
                    break
        p_new = trilin_scalar_u8(pres, nx_p, ny_p, nz_p)
        p_mid = trilin_scalar_u8(pres, 0.5 * (x + nx_p), 0.5 * (y + ny_p), 0.5 * (z + nz_p))
        if p_new < 76.5 or p_mid < 76.5:
            if p_new < 52.5 or p_mid < 52.5:
                break
            low_pres_run += 1
            if low_pres_run > max_low_steps:
                break
        else:
            low_pres_run = 0
        if is_covered_same_dir(cov_cnt, cov_nx, cov_ny, nx_p, ny_p, nz_p, ntx, nty, ntz):
            cov_run += 1
            if cov_run >= 2:
                break
        else:
            cov_run = 0
        loop_hit = False
        for j in range(0, n_pts - 5):
            ddx = nx_p - pts[j, 0]
            ddy = ny_p - pts[j, 1]
            ddz = nz_p - pts[j, 2]
            if ddx * ddx + ddy * ddy + ddz * ddz < 9.0:
                loop_hit = True
                break
        if loop_hit:
            break
        tx = 0.5 * ntx + 0.3 * dx + 0.2 * ux
        ty = 0.5 * nty + 0.3 * dy + 0.2 * uy
        tz = 0.5 * ntz + 0.3 * dz + 0.2 * uz
        tn = np.sqrt(tx * tx + ty * ty + tz * tz)
        tx /= tn
        ty /= tn
        tz /= tn
        x, y, z = (nx_p, ny_p, nz_p)
        pts[n_pts, 0] = x
        pts[n_pts, 1] = y
        pts[n_pts, 2] = z
        tans[n_pts, 0] = tx
        tans[n_pts, 1] = ty
        tans[n_pts, 2] = tz
        n_pts += 1
    while n_pts > 0 and cov_run > 0:
        n_pts -= 1
        cov_run -= 1
    while n_pts > 0:
        if trilin_scalar_u8(pres, pts[n_pts - 1, 0], pts[n_pts - 1, 1], pts[n_pts - 1, 2]) < 76.5:
            n_pts -= 1
        else:
            break
    return n_pts


@njit(cache=True)
def mark_curve_coverage(
    cov_cnt, cov_nx, cov_ny, cent, pres, nx, ny, pts, tangents, n_total, core_r=2.05
):
    D, H, W = cov_cnt.shape
    core_r2 = np.float32(core_r * core_r)
    for i in range(n_total):
        px, py, pz = (pts[i, 0], pts[i, 1], pts[i, 2])
        tx, ty, tz = (tangents[i, 0], tangents[i, 1], tangents[i, 2])
        bnx, bny = encode_dir(tx, ty, tz)
        c0 = trilin_scalar_u16(cent, px, py, pz)
        ix0 = int(px + 0.5)
        iy0 = int(py + 0.5)
        iz0 = int(pz + 0.5)
        for dz_i in range(-3, 4):
            zz = iz0 + dz_i
            if zz < 0 or zz >= D:
                continue
            for dy_i in range(-3, 4):
                yy = iy0 + dy_i
                if yy < 0 or yy >= H:
                    continue
                for dx_i in range(-3, 4):
                    xx = ix0 + dx_i
                    if xx < 0 or xx >= W:
                        continue
                    vx = xx - px
                    vy = yy - py
                    vz = zz - pz
                    par = vx * tx + vy * ty + vz * tz
                    if abs(par) > 0.9:
                        continue
                    perp2 = vx * vx + vy * vy + vz * vz - par * par
                    if perp2 > 3.5 * 3.5:
                        continue
                    ux, uy, uz = decode_dir(nx[zz, yy, xx], ny[zz, yy, xx])
                    if abs(ux * tx + uy * ty + uz * tz) <= 0.85:
                        continue
                    if perp2 <= core_r2:
                        suppress = True
                    else:
                        cq = np.float32(cent[zz, yy, xx]) * np.float32(0.01)
                        cm1 = trilin_scalar_u16(
                            cent, 0.6 * px + 0.4 * xx, 0.6 * py + 0.4 * yy, 0.6 * pz + 0.4 * zz
                        )
                        cm2 = trilin_scalar_u16(
                            cent,
                            0.35 * px + 0.65 * xx,
                            0.35 * py + 0.65 * yy,
                            0.35 * pz + 0.65 * zz,
                        )
                        suppress = c0 + 2.0 >= cm1 and cm1 + 3.0 >= cm2 and (cm2 + 3.0 >= cq)
                    if suppress:
                        if cov_cnt[zz, yy, xx] == 0:
                            cov_cnt[zz, yy, xx] = 1
                            cov_nx[zz, yy, xx] = bnx
                            cov_ny[zz, yy, xx] = bny
                        else:
                            ux0, uy0, uz0 = decode_dir(cov_nx[zz, yy, xx], cov_ny[zz, yy, xx])
                            if abs(ux0 * tx + uy0 * ty + uz0 * tz) <= 0.85:
                                cov_cnt[zz, yy, xx] = 2


@njit(parallel=True, cache=True)
def fit_and_resample_splines(
    pts, offsets, pres, min_final_len=16.0, target_step=0.8, smooth_iters=10, max_dev=0.3
):
    n_curves = len(offsets) - 1
    smoothed = pts.copy()
    new_counts = np.zeros(n_curves, dtype=np.int64)
    for c in prange(n_curves):
        a = offsets[c]
        b = offsets[c + 1]
        m = b - a
        if m < 4:
            continue
        buf = np.empty((m, 3), dtype=np.float32)
        for i in range(m):
            buf[i, 0] = pts[a + i, 0]
            buf[i, 1] = pts[a + i, 1]
            buf[i, 2] = pts[a + i, 2]
        for _ in range(smooth_iters):
            for i in range(1, m - 1):
                if i == 1 or i == m - 2:
                    nx_p = 0.25 * buf[i - 1, 0] + 0.5 * buf[i, 0] + 0.25 * buf[i + 1, 0]
                    ny_p = 0.25 * buf[i - 1, 1] + 0.5 * buf[i, 1] + 0.25 * buf[i + 1, 1]
                    nz_p = 0.25 * buf[i - 1, 2] + 0.5 * buf[i, 2] + 0.25 * buf[i + 1, 2]
                else:
                    lap = 0.5 * (buf[i - 1] + buf[i + 1])
                    bih = (-buf[i - 2] + 4.0 * buf[i - 1] + 4.0 * buf[i + 1] - buf[i + 2]) / 6.0
                    nx_p = 0.5 * buf[i, 0] + 0.3 * lap[0] + 0.2 * bih[0]
                    ny_p = 0.5 * buf[i, 1] + 0.3 * lap[1] + 0.2 * bih[1]
                    nz_p = 0.5 * buf[i, 2] + 0.3 * lap[2] + 0.2 * bih[2]
                ox, oy, oz = (pts[a + i, 0], pts[a + i, 1], pts[a + i, 2])
                dx = nx_p - ox
                dy = ny_p - oy
                dz = nz_p - oz
                d = np.sqrt(dx * dx + dy * dy + dz * dz)
                if d > max_dev:
                    s = max_dev / d
                    nx_p = ox + dx * s
                    ny_p = oy + dy * s
                    nz_p = oz + dz * s
                p_orig = trilin_scalar_u8(pres, ox, oy, oz)
                p_new = trilin_scalar_u8(pres, nx_p, ny_p, nz_p)
                if p_orig >= 76.5 and p_new < max(77.0, p_orig - 6.0):
                    nx_p = 0.5 * (ox + buf[i, 0])
                    ny_p = 0.5 * (oy + buf[i, 1])
                    nz_p = 0.5 * (oz + buf[i, 2])
                buf[i, 0] = nx_p
                buf[i, 1] = ny_p
                buf[i, 2] = nz_p
        arc = np.float32(0.0)
        for i in range(m):
            smoothed[a + i, 0] = buf[i, 0]
            smoothed[a + i, 1] = buf[i, 1]
            smoothed[a + i, 2] = buf[i, 2]
            if i > 0:
                dx = buf[i, 0] - buf[i - 1, 0]
                dy = buf[i, 1] - buf[i - 1, 1]
                dz = buf[i, 2] - buf[i - 1, 2]
                arc += np.sqrt(dx * dx + dy * dy + dz * dz)
        if arc >= min_final_len:
            n_out = max(int(arc / target_step + 0.5) + 1, 2)
            new_counts[c] = n_out
    return (smoothed, new_counts)


@njit(parallel=True, cache=True)
def evaluate_catmull_rom_splines(
    smoothed, offsets, kept_indices, new_offsets, shape=(512, 512, 512)
):
    n_kept = len(kept_indices)
    total_pts = new_offsets[n_kept]
    out_pts = np.empty((total_pts, 3), dtype=np.float32)
    out_lengths = np.empty(n_kept, dtype=np.float32)
    for idx in prange(n_kept):
        c = kept_indices[idx]
        a = offsets[c]
        b = offsets[c + 1]
        m = b - a
        n_out = new_offsets[idx + 1] - new_offsets[idx]
        out_start = new_offsets[idx]
        cum_s = np.empty(m, dtype=np.float32)
        cum_s[0] = 0.0
        for i in range(m - 1):
            dx = smoothed[a + i + 1, 0] - smoothed[a + i, 0]
            dy = smoothed[a + i + 1, 1] - smoothed[a + i, 1]
            dz = smoothed[a + i + 1, 2] - smoothed[a + i, 2]
            cum_s[i + 1] = cum_s[i] + np.sqrt(dx * dx + dy * dy + dz * dz)
        tot_s = cum_s[m - 1]
        out_pts[out_start] = smoothed[a]
        out_pts[out_start + n_out - 1] = smoothed[b - 1]
        seg_k = 0
        for j in range(1, n_out - 1):
            target_s = np.float32(j) / np.float32(n_out - 1) * tot_s
            while seg_k < m - 2 and cum_s[seg_k + 1] < target_s:
                seg_k += 1
            ds = cum_s[seg_k + 1] - cum_s[seg_k]
            u = (target_s - cum_s[seg_k]) / max(ds, np.float32(1e-06))
            u2 = u * u
            u3 = u2 * u
            i0 = a + max(0, seg_k - 1)
            i1 = a + seg_k
            i2 = a + seg_k + 1
            i3 = a + min(m - 1, seg_k + 2)
            w0 = -0.5 * u3 + u2 - 0.5 * u
            w1 = 1.5 * u3 - 2.5 * u2 + 1.0
            w2 = -1.5 * u3 + 2.0 * u2 + 0.5 * u
            w3 = 0.5 * u3 - 0.5 * u2
            for d in range(3):
                val = (
                    w0 * smoothed[i0, d]
                    + w1 * smoothed[i1, d]
                    + w2 * smoothed[i2, d]
                    + w3 * smoothed[i3, d]
                )
                out_pts[out_start + j, d] = min(
                    max(val, np.float32(0.0)), np.float32(shape[2 - d] - 1)
                )
        poly_len = np.float32(0.0)
        for j in range(n_out - 1):
            dx = out_pts[out_start + j + 1, 0] - out_pts[out_start + j, 0]
            dy = out_pts[out_start + j + 1, 1] - out_pts[out_start + j, 1]
            dz = out_pts[out_start + j + 1, 2] - out_pts[out_start + j, 2]
            poly_len += np.sqrt(dx * dx + dy * dy + dz * dz)
        out_lengths[idx] = poly_len
    return (out_pts, out_lengths)


@njit(cache=True)
def first_curve_split(pts, n_pts):
    if n_pts < 2:
        return 0
    for i in range(n_pts):
        px, py, pz = (pts[i, 0], pts[i, 1], pts[i, 2])
        for j in range(i + 7, n_pts):
            dx = pts[j, 0] - px
            dy = pts[j, 1] - py
            dz = pts[j, 2] - pz
            if dx * dx + dy * dy + dz * dz < 10.24:
                return (i + j) // 2
    for i in range(1, n_pts - 1):
        u1x = pts[i, 0] - pts[i - 1, 0]
        u1y = pts[i, 1] - pts[i - 1, 1]
        u1z = pts[i, 2] - pts[i - 1, 2]
        u2x = pts[i + 1, 0] - pts[i, 0]
        u2y = pts[i + 1, 1] - pts[i, 1]
        u2z = pts[i + 1, 2] - pts[i, 2]
        ul1 = np.sqrt(u1x * u1x + u1y * u1y + u1z * u1z)
        ul2 = np.sqrt(u2x * u2x + u2y * u2y + u2z * u2z)
        if ul1 > 1e-05 and ul2 > 1e-05:
            if (u1x * u2x + u1y * u2y + u1z * u2z) / (ul1 * ul2) < 0.5735764363510462:
                return i
    s4 = 5
    if n_pts >= 2 * s4 + 1:
        for i in range(s4, n_pts - s4):
            v1x = pts[i, 0] - pts[i - s4, 0]
            v1y = pts[i, 1] - pts[i - s4, 1]
            v1z = pts[i, 2] - pts[i - s4, 2]
            v2x = pts[i + s4, 0] - pts[i, 0]
            v2y = pts[i + s4, 1] - pts[i, 1]
            v2z = pts[i + s4, 2] - pts[i, 2]
            l1 = np.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
            l2 = np.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
            if l1 > 1e-05 and l2 > 1e-05:
                if (v1x * v2x + v1y * v2y + v1z * v2z) / (l1 * l2) < 0.5735764363510462:
                    return i
    s8 = 10
    if n_pts >= 2 * s8 + 1:
        for i in range(s8, n_pts - s8):
            w1x = pts[i, 0] - pts[i - s8, 0]
            w1y = pts[i, 1] - pts[i - s8, 1]
            w1z = pts[i, 2] - pts[i - s8, 2]
            w2x = pts[i + s8, 0] - pts[i, 0]
            w2y = pts[i + s8, 1] - pts[i, 1]
            w2z = pts[i + s8, 2] - pts[i, 2]
            wl1 = np.sqrt(w1x * w1x + w1y * w1y + w1z * w1z)
            wl2 = np.sqrt(w2x * w2x + w2y * w2y + w2z * w2z)
            if wl1 > 1e-05 and wl2 > 1e-05:
                if (w1x * w2x + w1y * w2y + w1z * w2z) / (wl1 * wl2) < 0.5735764363510462:
                    return i
    cum_s = np.empty(n_pts, dtype=np.float32)
    cum_s[0] = 0.0
    for i in range(n_pts - 1):
        dx = pts[i + 1, 0] - pts[i, 0]
        dy = pts[i + 1, 1] - pts[i, 1]
        dz = pts[i + 1, 2] - pts[i, 2]
        cum_s[i + 1] = cum_s[i] + np.sqrt(dx * dx + dy * dy + dz * dz)
    # A long smooth winding is valid regardless of its global chord/arc ratio.
    win = 24
    if n_pts > win:
        for i in range(n_pts - win):
            arc_w = cum_s[i + win] - cum_s[i]
            if arc_w > 1.0:
                wx = pts[i + win, 0] - pts[i, 0]
                wy = pts[i + win, 1] - pts[i, 1]
                wz = pts[i + win, 2] - pts[i, 2]
                ch_w = np.sqrt(wx * wx + wy * wy + wz * wz)
                if ch_w < 0.78 * arc_w:
                    return i + win // 2
    return -1


@njit(cache=True)
def valid_curve_spans(pts, n_pts, seed_index):
    """Return unchanged contiguous valid pieces, in original path order.

    An invalid bend splits at its vertex; both sides retain that vertex.
    No points are interpolated, joined or moved. Pieces are independently
    checked against ALL existing geometry guards. Length eligibility is still
    applied by the caller, individually. A bad tip never discards a valid stem.
    """
    pending_start = np.empty(n_pts, dtype=np.int64)
    pending_end = np.empty(n_pts, dtype=np.int64)
    starts = np.empty(n_pts, dtype=np.int64)
    ends = np.empty(n_pts, dtype=np.int64)
    pending_start[0] = 0
    pending_end[0] = n_pts
    pending = 1
    count = 0
    while pending:
        pending -= 1
        a = pending_start[pending]
        b = pending_end[pending]
        n = b - a
        if n < 2:
            continue
        piece = pts[a:b]
        cut = first_curve_split(piece, n)
        # Preserve the original cross-seed self-loop guard too.
        local_seed = seed_index - a
        if cut < 0 and 0 < local_seed < n - 1:
            for j in range(local_seed):
                for k in range(local_seed + 1, n):
                    if k - j < 6:
                        continue
                    delta = piece[j] - piece[k]
                    if np.dot(delta, delta) < 9.0:
                        cut = (j + k) // 2
                        break
                if cut >= 0:
                    break
        # Global winding around the umbilicus is not a local loop.
        if cut < 0:
            starts[count] = a
            ends[count] = b
            count += 1
            continue
        assert 0 < cut < n - 1, "A rejected nontrivial curve must have an interior split"
        split = a + cut
        # Right first in the stack, so the left piece is processed first.
        pending_start[pending] = split
        pending_end[pending] = b
        pending += 1
        pending_start[pending] = a
        pending_end[pending] = split + 1
        pending += 1
    return starts[:count], ends[:count]


@njit(cache=True)
def trace_seed_range(
    cent,
    pres,
    nx,
    ny,
    seeds_z,
    seeds_y,
    seeds_x,
    cov_cnt,
    cov_nx,
    cov_ny,
    out_pts,
    out_tans,
    out_offsets,
    pt_cursor,
    n_curves,
    begin,
    end,
    min_raw_len=7.5,
    step_size=0.75,
    max_low_steps=3,
    core_r=2.05,
    n_pass1=-1,
):
    if n_pass1 < 0:
        n_pass1 = len(seeds_z)
    max_pts_total = len(out_pts)
    max_curves = len(out_offsets) - 1
    pts_f = np.empty((32768, 3), dtype=np.float32)
    tans_f = np.empty((32768, 3), dtype=np.float32)
    pts_b = np.empty((32768, 3), dtype=np.float32)
    tans_b = np.empty((32768, 3), dtype=np.float32)
    pts_c = np.empty((65537, 3), dtype=np.float32)
    tans_c = np.empty((65537, 3), dtype=np.float32)
    for i in range(begin, end):
        sz = seeds_z[i]
        sy = seeds_y[i]
        sx = seeds_x[i]
        if i >= n_pass1 and cov_cnt[sz, sy, sx] > 0:
            continue
        ux, uy, uz = decode_dir(nx[sz, sy, sx], ny[sz, sy, sx])
        if is_covered_same_dir(cov_cnt, cov_nx, cov_ny, sx, sy, sz, ux, uy, uz):
            continue
        cx, cy, cz = recenter_normal_plane(
            cent, pres, nx, ny, np.float32(sx), np.float32(sy), np.float32(sz), ux, uy, uz, 1.5, 4
        )
        if is_covered_same_dir(cov_cnt, cov_nx, cov_ny, cx, cy, cz, ux, uy, uz):
            continue
        if trilin_scalar_u8(pres, cx, cy, cz) < 76.5:
            continue
        ux, uy, uz = trilin_dir_aligned(pres, nx, ny, cx, cy, cz, ux, uy, uz)
        nf = trace_half_buf(
            cent,
            pres,
            nx,
            ny,
            cov_cnt,
            cov_nx,
            cov_ny,
            cx,
            cy,
            cz,
            ux,
            uy,
            uz,
            pts_f,
            tans_f,
            step_size,
            max_low_steps,
        )
        if nf == len(pts_f):
            raise RuntimeError(
                "Full-scroll half-path buffer reached; refusing an artificial endpoint"
            )
        bux, buy, buz = (-ux, -uy, -uz)
        if nf >= 3:
            k_dep = min(3, nf - 1)
            dfx = pts_f[k_dep, 0] - cx
            dfy = pts_f[k_dep, 1] - cy
            dfz = pts_f[k_dep, 2] - cz
            dfn = np.sqrt(dfx * dfx + dfy * dfy + dfz * dfz)
            if dfn > 1e-05:
                dfx /= dfn
                dfy /= dfn
                dfz /= dfn
                bux = -(0.55 * dfx + 0.45 * ux)
                buy = -(0.55 * dfy + 0.45 * uy)
                buz = -(0.55 * dfz + 0.45 * uz)
                bn = np.sqrt(bux * bux + buy * buy + buz * buz)
                bux /= bn
                buy /= bn
                buz /= bn
        nb = trace_half_buf(
            cent,
            pres,
            nx,
            ny,
            cov_cnt,
            cov_nx,
            cov_ny,
            cx,
            cy,
            cz,
            bux,
            buy,
            buz,
            pts_b,
            tans_b,
            step_size,
            max_low_steps,
        )
        if nb == len(pts_b):
            raise RuntimeError(
                "Full-scroll half-path buffer reached; refusing an artificial endpoint"
            )
        ntot = nb + 1 + nf
        if ntot < 8:
            continue
        idx = 0
        for j in range(nb - 1, -1, -1):
            pts_c[idx, 0] = pts_b[j, 0]
            pts_c[idx, 1] = pts_b[j, 1]
            pts_c[idx, 2] = pts_b[j, 2]
            tans_c[idx, 0] = -tans_b[j, 0]
            tans_c[idx, 1] = -tans_b[j, 1]
            tans_c[idx, 2] = -tans_b[j, 2]
            idx += 1
        pts_c[idx, 0] = cx
        pts_c[idx, 1] = cy
        pts_c[idx, 2] = cz
        tans_c[idx, 0] = ux
        tans_c[idx, 1] = uy
        tans_c[idx, 2] = uz
        idx += 1
        for j in range(nf):
            pts_c[idx, 0] = pts_f[j, 0]
            pts_c[idx, 1] = pts_f[j, 1]
            pts_c[idx, 2] = pts_f[j, 2]
            tans_c[idx, 0] = tans_f[j, 0]
            tans_c[idx, 1] = tans_f[j, 1]
            tans_c[idx, 2] = tans_f[j, 2]
            idx += 1
        starts, ends = valid_curve_spans(pts_c, ntot, nb)
        req_len = min_raw_len if i < n_pass1 else max(min_raw_len, np.float32(16.0))
        for part in range(len(starts)):
            a = starts[part]
            b = ends[part]
            n = b - a
            if n < 8:
                continue
            arc_len = np.float32(0.0)
            for j in range(a, b - 1):
                dx = pts_c[j + 1, 0] - pts_c[j, 0]
                dy = pts_c[j + 1, 1] - pts_c[j, 1]
                dz = pts_c[j + 1, 2] - pts_c[j, 2]
                arc_len += np.sqrt(dx * dx + dy * dy + dz * dz)
            if arc_len < req_len:
                continue
            if pt_cursor + n >= max_pts_total or n_curves >= max_curves:
                raise RuntimeError("Tracing capacity reached: increase the batch output capacity")
            mark_curve_coverage(
                cov_cnt, cov_nx, cov_ny, cent, pres, nx, ny, pts_c[a:b], tans_c[a:b], n, core_r
            )
            for j in range(n):
                out_pts[pt_cursor + j] = pts_c[a + j]
                out_tans[pt_cursor + j] = tans_c[a + j]
            pt_cursor += n
            n_curves += 1
            out_offsets[n_curves] = pt_cursor
    return pt_cursor, n_curves
