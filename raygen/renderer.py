"""The two renderers that make a training pair.

Both passes see identical geometry, materials, camera and lighting. Only the
light transport differs, so the model has to learn lighting rather than geometry.

| effect      | ``render_game`` (fast)                       | ``render_rt`` (expensive)            |
|-------------|----------------------------------------------|--------------------------------------|
| shadows     | one 64^2 shadow map, 2x2 PCF, hard + aliased  | sampled sun disk -> true penumbrae   |
| occlusion   | SSAO, screen space, haloes                    | multi-bounce GI, real AO + bleeding  |
| reflections | SSR: only what is already on screen           | real rays, whole scene, roughness    |
| water       | sky-only reflection + depth tint              | reflection *and* refraction, Fresnel |
| glass       | alpha blend against the sky                   | true refraction, dispersion, frost   |
| subsurface  | wrap-lighting approximation                   | wrap + real transmitted GI           |
| smoke       | analytic absorption, no self-shadow           | marched, sun-shadowed per step       |
| caustics    | none                                          | focused through glass, f = nR/2(n-1) |

The model input is a 23-channel G-buffer -- everything a deferred engine already
has in memory. Seven of those channels exist because of a measurement: a model
given only surface data improved terrain 4.1x but made *glass worse than not
editing it* (0.97x), because refraction depends on geometry behind the surface
and reflection on geometry off screen. So the G-buffer carries a depth-peeled
second layer (``behind_rgb``/``behind_depth``) and an environment probe
(``env_refl``). The depth peel worked: transmission went 2.10x -> 2.87x. The
sky-only probe did not -- 92% of reflection rays hit scene geometry, not sky,
so it is nearly information-free and metal stayed at ~1.2x. Making that probe
capture scene content is the outstanding fix.
"""

import math

import torch
import torch.nn.functional as F

from .scenes import (FAR, MAT_EMISSIVE, MAT_GLASS, MAT_METAL, MAT_SSS, SHADOW_MAP_RES,
                     _basis, _dot, _hw, _norm, fbm2, make_scene_batch, sky, terrain_h,
                     terrain_march, terrain_material, terrain_normal)


def camera_basis(sc):
    fwd = _norm(sc["look"] - sc["cam"])
    wu = torch.tensor([0.0, 1.0, 0.0], device=fwd.device).expand_as(fwd)
    right = _norm(torch.cross(fwd, wu, dim=-1))
    return fwd, right, torch.cross(right, fwd, dim=-1)

def camera_rays(sc, res):
    h, w = _hw(res)
    device = sc["cam"].device
    B = sc["cam"].shape[0]
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32),
                            indexing="ij")
    px = ((xs + 0.5) / w * 2 - 1) * (w / h)
    py = 1 - (ys + 0.5) / h * 2
    fwd, right, up = camera_basis(sc)
    tan = torch.tan(sc["fov"] * 0.5)[:, None, None]
    d = (fwd[:, None] + right[:, None] * (px.reshape(1, -1, 1) * tan)
         + up[:, None] * (py.reshape(1, -1, 1) * tan))
    return sc["cam"][:, None].expand(B, h * w, 3).contiguous(), _norm(d)

def _hit_spheres(ro, rd, sc, exclude=None):
    c = sc["sph_c"]
    ox = ro[..., 0:1] - c[:, None, :, 0]
    oy = ro[..., 1:2] - c[:, None, :, 1]
    oz = ro[..., 2:3] - c[:, None, :, 2]
    dx, dy, dz = rd[..., 0:1], rd[..., 1:2], rd[..., 2:3]
    b = ox * dx + oy * dy + oz * dz
    disc = b * b - (ox * ox + oy * oy + oz * oz - sc["sph_r"][:, None, :] ** 2)
    sq = torch.sqrt(disc.clamp(min=0))
    t0, t1 = -b - sq, -b + sq
    t = torch.where(t0 > 1e-3, t0, t1)
    t = torch.where((disc > 0) & (t > 1e-3), t, torch.full_like(t, FAR))
    if exclude is not None:
        # A reflection probe must never capture the object it sits inside: from
        # an object's centre every ray hits its own inner surface first, so
        # without this the probe is a picture of the host and nothing else.
        idx = torch.arange(t.shape[-1], device=t.device).view(1, 1, -1)
        t = torch.where(idx == int(exclude), torch.full_like(t, FAR), t)
    return t

def _hit_boxes(ro, rd, c, hh, yaw, want_normal=False):
    cs, sn = torch.cos(yaw)[:, None, :], torch.sin(yaw)[:, None, :]
    rx = ro[..., 0:1] - c[:, None, :, 0]
    ry = ro[..., 1:2] - c[:, None, :, 1]
    rz = ro[..., 2:3] - c[:, None, :, 2]
    ox, oz = cs * rx + sn * rz, -sn * rx + cs * rz
    dxi, dzi = rd[..., 0:1], rd[..., 2:3]
    dx, dz = cs * dxi + sn * dzi, -sn * dxi + cs * dzi
    dy = rd[..., 1:2].expand_as(dx)

    def slab(o, d, half):
        d = torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
        a, b = (-half - o) / d, (half - o) / d
        return torch.minimum(a, b), torch.maximum(a, b), d

    lo_x, hi_x, dx = slab(ox, dx, hh[:, None, :, 0])
    lo_y, hi_y, dy = slab(ry, dy, hh[:, None, :, 1])
    lo_z, hi_z, dz = slab(oz, dz, hh[:, None, :, 2])
    tmin = torch.maximum(torch.maximum(lo_x, lo_y), lo_z)
    tmax = torch.minimum(torch.minimum(hi_x, hi_y), hi_z)
    t = torch.where(tmin > 1e-3, tmin, tmax)
    t = torch.where((tmax > tmin.clamp(min=1e-3)) & (t > 1e-3), t, torch.full_like(t, FAR))
    if not want_normal:
        return t, None
    tc = t.clamp(max=1e4)
    ax = (ox + dx * tc) / hh[:, None, :, 0]
    ay = (ry + dy * tc) / hh[:, None, :, 1]
    az = (oz + dz * tc) / hh[:, None, :, 2]
    big = torch.maximum(torch.maximum(ax.abs(), ay.abs()), az.abs()) - 1e-5
    nx = torch.sign(ax) * (ax.abs() >= big).float()
    ny = torch.sign(ay) * (ay.abs() >= big).float()
    nz = torch.sign(az) * (az.abs() >= big).float()
    inv = torch.rsqrt((nx * nx + ny * ny + nz * nz).clamp(min=1e-8))
    nx, ny, nz = nx * inv, ny * inv, nz * inv
    return t, torch.stack([cs * nx - sn * nz, ny, sn * nx + cs * nz], -1)

def _trunk_boxes(sc):
    """Plant trunks reuse the box code: thin, axis-aligned, on the ground."""
    c = sc["trunk_c"]
    hh = torch.stack([torch.full_like(sc["trunk_h"], 0.055),
                      sc["trunk_h"], torch.full_like(sc["trunk_h"], 0.055)], -1)
    return c, hh, torch.zeros_like(sc["trunk_h"])

def _hit_cones(ro, rd, sc, want_normal=False):
    """Canopy cones: apex up, opening downward over height h."""
    a = sc["cone_apex"]
    k = (sc["cone_r"] / sc["cone_h"].clamp(min=1e-3)) ** 2         # [B,P_cone]
    ox = ro[..., 0:1] - a[:, None, :, 0]
    oy = ro[..., 1:2] - a[:, None, :, 1]
    oz = ro[..., 2:3] - a[:, None, :, 2]
    dx, dy, dz = rd[..., 0:1], rd[..., 1:2], rd[..., 2:3]
    kk = k[:, None, :]
    A = dx * dx + dz * dz - kk * dy * dy
    B = 2 * (ox * dx + oz * dz - kk * oy * dy)
    C = ox * ox + oz * oz - kk * oy * oy
    A = torch.where(A.abs() < 1e-7, torch.full_like(A, 1e-7), A)
    disc = B * B - 4 * A * C
    sq = torch.sqrt(disc.clamp(min=0))
    t0 = (-B - sq) / (2 * A)
    t1 = (-B + sq) / (2 * A)
    lo = torch.minimum(t0, t1)
    hi = torch.maximum(t0, t1)

    hgt = sc["cone_h"][:, None, :]

    def valid(t):
        y = oy + dy * t
        return (t > 1e-3) & (y <= 0) & (y >= -hgt)

    t = torch.where(valid(lo), lo, torch.where(valid(hi), hi, torch.full_like(lo, FAR)))
    t = torch.where(disc > 0, t, torch.full_like(t, FAR))
    if not want_normal:
        return t, None
    tc = t.clamp(max=1e4)
    x, y, z = ox + dx * tc, oy + dy * tc, oz + dz * tc
    return t, _norm(torch.stack([x, -kk * y, z], -1))

def _water_normal(p, sc):
    """Procedural wave normal -- small, high frequency, animated by phase."""
    f = sc["wave_freq"][:, None]
    a = sc["wave_amp"][:, None]
    x, z = p[..., 0], p[..., 2]
    dx = a * (torch.cos(x * f + z * 0.7) * f + 0.6 * torch.cos(x * f * 2.3 - z) * f * 2.3)
    dz = a * (torch.cos(z * f * 1.13 - x * 0.5) * f * 1.13 + 0.5 * torch.cos(z * f * 1.9) * f * 1.9)
    return _norm(torch.stack([-dx, torch.ones_like(dx), -dz], -1))

KIND_MISS, KIND_TERRAIN, KIND_SPHERE, KIND_BOX, KIND_CONE, KIND_TRUNK, KIND_WATER = range(7)

def intersect(ro, rd, sc, terrain_steps=32, cheap=False, exclude_sph=None):
    B, P, _ = ro.shape
    ts, ki = _hit_spheres(ro, rd, sc, exclude=exclude_sph).min(-1)
    tb_all, bn_all = _hit_boxes(ro, rd, sc["box_c"], sc["box_h"], sc["box_yaw"], True)
    tb, bi = tb_all.min(-1)
    tc_all, cn_all = _hit_cones(ro, rd, sc, True)
    tc, ci = tc_all.min(-1)
    trc, trh, tryaw = _trunk_boxes(sc)
    tt_all, tn_all = _hit_boxes(ro, rd, trc, trh, tryaw, True)
    tt, ti = tt_all.min(-1)

    tter = terrain_march(ro, rd, sc, steps=terrain_steps, cheap=cheap)

    wy = sc["water_y"][:, None]
    dy = rd[..., 1]
    dy_safe = torch.where(dy.abs() < 1e-6, torch.full_like(dy, -1e-6), dy)
    tw = (wy - ro[..., 1]) / dy_safe
    tw = torch.where((tw > 1e-3) & (sc["has_water"][:, None] > 0), tw, torch.full_like(tw, FAR))

    t = torch.minimum(torch.minimum(torch.minimum(ts, tb), torch.minimum(tc, tt)),
                      torch.minimum(tter, tw))
    t = t.clamp(max=1e4)

    is_s = (ts <= t) & (ts < FAR)
    is_b = (tb <= t) & (tb < FAR) & ~is_s
    is_c = (tc <= t) & (tc < FAR) & ~is_s & ~is_b
    is_tr = (tt <= t) & (tt < FAR) & ~is_s & ~is_b & ~is_c
    is_w = (tw <= t) & (tw < FAR) & ~is_s & ~is_b & ~is_c & ~is_tr
    is_ter = (tter <= t) & (tter < FAR) & ~is_s & ~is_b & ~is_c & ~is_tr & ~is_w
    hit_any = is_s | is_b | is_c | is_tr | is_w | is_ter

    p = ro + rd * t[..., None]

    def g3(src, idx):
        return torch.gather(src[:, None].expand(B, P, src.shape[1], 3), 2,
                            idx[..., None, None].expand(B, P, 1, 3)).squeeze(2)

    def g1(src, idx):
        return torch.gather(src[:, None].expand(B, P, src.shape[1]), 2,
                            idx[..., None]).squeeze(2)

    s_c = g3(sc["sph_c"], ki)
    s_alb, s_ro, s_mat = g3(sc["sph_alb"], ki), g1(sc["sph_rough"], ki), g1(sc["sph_mat"], ki)
    b_alb, b_ro, b_mat = g3(sc["box_alb"], bi), g1(sc["box_rough"], bi), g1(sc["box_mat"], bi)
    c_alb, c_ro = g3(sc["cone_alb"], ci), g1(sc["cone_rough"], ci)
    tr_alb = g3(sc["trunk_alb"], ti)
    b_n = torch.gather(bn_all, 2, bi[..., None, None].expand(B, P, 1, 3)).squeeze(2)
    c_n = torch.gather(cn_all, 2, ci[..., None, None].expand(B, P, 1, 3)).squeeze(2)
    tr_n = torch.gather(tn_all, 2, ti[..., None, None].expand(B, P, 1, 3)).squeeze(2)

    ter_n_raw = terrain_normal(p[..., 0], p[..., 2], sc)
    ter_alb, ter_ro, ter_sss, ter_n = terrain_material(p, ter_n_raw, sc, cheap)
    w_n = _water_normal(p, sc)

    zeros = torch.zeros_like(t)
    n = torch.where(is_s[..., None], _norm(p - s_c),
        torch.where(is_b[..., None], b_n,
        torch.where(is_c[..., None], c_n,
        torch.where(is_tr[..., None], tr_n,
        torch.where(is_w[..., None], w_n, ter_n)))))
    alb = torch.where(is_s[..., None], s_alb,
          torch.where(is_b[..., None], b_alb,
          torch.where(is_c[..., None], c_alb,
          torch.where(is_tr[..., None], tr_alb,
          torch.where(is_w[..., None], sc["water_col"][:, None], ter_alb)))))
    rough = torch.where(is_s, s_ro,
            torch.where(is_b, b_ro,
            torch.where(is_c, c_ro,
            torch.where(is_tr, torch.full_like(t, 0.8),
            torch.where(is_w, torch.full_like(t, 0.04), ter_ro)))))
    metal = ((is_s & (s_mat == MAT_METAL)) | (is_b & (b_mat == MAT_METAL))).float()
    trans = ((is_s & (s_mat == MAT_GLASS)) | (is_b & (b_mat == MAT_GLASS))).float()
    trans = torch.maximum(trans, is_w.float())
    sss = torch.where(is_s & (s_mat == MAT_SSS), torch.full_like(t, 0.85),
          torch.where(is_c, torch.full_like(t, 0.55),
          torch.where(is_ter, ter_sss, zeros)))

    kind = (is_ter.float() * KIND_TERRAIN + is_s.float() * KIND_SPHERE
            + is_b.float() * KIND_BOX + is_c.float() * KIND_CONE
            + is_tr.float() * KIND_TRUNK + is_w.float() * KIND_WATER)
    t = torch.where(hit_any, t, torch.full_like(t, FAR))
    return dict(t=t, kind=kind, hit=hit_any, n=n, alb=alb.clamp(0, 1), rough=rough,
                metal=metal, trans=trans, sss=sss, p=p)

def occluded(ro, rd, maxt, sc, terrain_steps=10):
    """Any-hit query. Glass only partially occludes, which is why it returns a
    transmittance in [0, 1] rather than a hard 0/1."""
    ts = _hit_spheres(ro, rd, sc)
    s_glass = (sc["sph_mat"] == MAT_GLASS)[:, None, :].expand_as(ts)
    solid_s = torch.where(s_glass, torch.full_like(ts, FAR), ts).min(-1).values
    glass_s = torch.where(s_glass, ts, torch.full_like(ts, FAR)).min(-1).values

    tb = _hit_boxes(ro, rd, sc["box_c"], sc["box_h"], sc["box_yaw"])[0]
    b_glass = (sc["box_mat"] == MAT_GLASS)[:, None, :].expand_as(tb)
    solid_b = torch.where(b_glass, torch.full_like(tb, FAR), tb).min(-1).values
    glass_b = torch.where(b_glass, tb, torch.full_like(tb, FAR)).min(-1).values

    tc = _hit_cones(ro, rd, sc)[0].min(-1).values
    trc, trh, tryaw = _trunk_boxes(sc)
    tt = _hit_boxes(ro, rd, trc, trh, tryaw)[0].min(-1).values
    tter = terrain_march(ro, rd, sc, steps=terrain_steps, cheap=True)

    blocked = ((solid_s < maxt) | (solid_b < maxt) | (tt < maxt) | (tter < maxt)).float()
    # canopies are dense but leaky; glass tints rather than blocks
    leafy = (tc < maxt).float() * 0.82
    glassy = ((glass_s < maxt) | (glass_b < maxt)).float() * 0.55
    vis = (1 - blocked) * (1 - leafy) * (1 - glassy)
    return 1 - vis

def _brdf(n, v, l, alb, rough, metal, sss):
    ndl = _dot(n, l)
    # wrap lighting: light bleeds around the terminator on scattering materials
    wrapped = ((ndl + sss * 0.6) / (1 + sss * 0.6)).clamp(min=0)
    diff = torch.where(sss > 0, wrapped, ndl.clamp(min=0))
    hv = _norm(l + v)
    a = (rough * rough).clamp(min=1e-3)
    shin = (2.0 / (a * a) - 2.0).clamp(1.0, 4096.0)
    spec = (_dot(n, hv).clamp(min=0) ** shin) * (shin + 8) / 25.13
    f0 = 0.04 + (alb - 0.04) * metal[..., None]
    kd = alb * (1 - metal[..., None])
    return kd * diff[..., None] + f0 * (spec * ndl.clamp(min=0))[..., None]

def _fresnel(cosi, ior):
    r0 = ((1 - ior) / (1 + ior)) ** 2
    return r0 + (1 - r0) * (1 - cosi.clamp(0, 1)) ** 5

def _refract(d, n, eta):
    """Snell's law; falls back to reflection on total internal reflection.

    `eta` is the index ratio, broadcastable to [B, P].
    """
    cosi = (-_dot(d, n)).clamp(-1, 1)
    k = 1 - eta * eta * (1 - cosi * cosi)
    ok = (k > 0).float()[..., None]
    r = eta[..., None] * d + (eta * cosi - torch.sqrt(k.clamp(min=0)))[..., None] * n
    return _norm(r) * ok + _norm(d - 2 * _dot(d, n)[..., None] * n) * (1 - ok)

def _tonemap(col, sc, res):
    col = col * sc["exposure"][:, None, None]
    col = col / (1.0 + col)
    h, w = _hw(res)
    return col.clamp(0, 1).pow(1 / 2.2).reshape(-1, h, w, 3).permute(0, 3, 1, 2).contiguous()

def _composite(col, hit, depth, rd, sc, res):
    fog = 1 - torch.exp(-sc["fog"][:, None, None] * depth[..., None].clamp(min=0))
    col = col * (1 - fog) + sc["sky_h"][:, None] * fog
    col = torch.where(hit[..., None], col, sky(rd, sc))
    return _tonemap(col, sc, res)

def _smoke_span(ro, rd, sc):
    """Entry/exit of the smoke sphere along the ray."""
    oc = ro - sc["smoke_c"][:, None]
    b = _dot(oc, rd)
    c = _dot(oc, oc) - (sc["smoke_r"][:, None] ** 2)
    disc = b * b - c
    sq = torch.sqrt(disc.clamp(min=0))
    t0 = (-b - sq).clamp(min=0)
    t1 = (-b + sq).clamp(min=0)
    live = (disc > 0) & (t1 > t0) & (sc["has_smoke"][:, None] > 0)
    return t0, t1, live

def _smoke_density(p, sc):
    d = (p - sc["smoke_c"][:, None]).norm(dim=-1) / sc["smoke_r"][:, None].clamp(min=1e-3)
    shell = (1 - d.clamp(0, 1)) ** 1.6
    turb = 0.6 + 0.6 * fbm2(p[..., 0] * 1.5 + p[..., 1] * 0.8,
                            p[..., 2] * 1.5 - p[..., 1] * 0.6, octaves=3, seed=5.5)
    return (shell * turb * sc["smoke_dens"][:, None]).clamp(min=0)

def smoke_cheap(ro, rd, tmax, sc):
    """Game path: analytic absorption, uniform density, no self-shadowing."""
    t0, t1, live = _smoke_span(ro, rd, sc)
    seg = (torch.minimum(t1, tmax) - t0).clamp(min=0) * live.float()
    mid = ro + rd * ((t0 + torch.minimum(t1, tmax)) * 0.5)[..., None]
    dens = _smoke_density(mid, sc) * live.float()
    tr = torch.exp(-seg * dens * 0.75)
    lit = sc["smoke_col"][:, None] * (sc["sky_h"][:, None] * 0.6 + sc["sun_col"][:, None] * 0.05)
    return tr[..., None], lit * (1 - tr)[..., None]

def smoke_marched(ro, rd, tmax, sc, steps=8, shadow_evals=3, gen=None):
    """Ray traced path: marched with real sun shadowing inside the volume.

    The sun-visibility query is the expensive part (a full any-hit trace), so it
    is evaluated at `shadow_evals` points along the span and held between them
    rather than at every step. The volume still self-shadows and still casts the
    god-ray gradient the fast path cannot produce; it just samples the shadow
    term more coarsely than the density.
    """
    t0, t1, live = _smoke_span(ro, rd, sc)
    t1 = torch.minimum(t1, tmax)
    seg = (t1 - t0).clamp(min=0)
    dt = seg / steps
    trans = torch.ones_like(seg)
    acc = torch.zeros_like(ro)
    sun = sc["sun_dir"][:, None].expand_as(ro)
    every = max(1, steps // max(1, shadow_evals))
    vis = None
    for i in range(steps):
        tc = t0 + dt * (i + 0.5)
        p = ro + rd * tc[..., None]
        dens = _smoke_density(p, sc) * live.float()
        a = (dens * dt).clamp(0, 4)
        if i % every == 0:
            vis = 1 - occluded(p, sun, torch.full_like(tc, 1e4), sc, terrain_steps=6)
            _, s1, slive = _smoke_span(p, sun, sc)
            vis = vis * torch.exp(-(s1.clamp(min=0)) * dens * 0.5 * slive.float())
        lit = (sc["smoke_col"][:, None] * (sc["sun_col"][:, None] * vis[..., None] * 0.28
                                           + sc["sky_h"][:, None] * 0.45))
        w = (1 - torch.exp(-a))[..., None] * trans[..., None]
        acc = acc + lit * w
        trans = trans * torch.exp(-a)
    return trans[..., None], acc

def shadow_map(sc, res=SHADOW_MAP_RES, extent=11.0):
    B = sc["cam"].shape[0]
    w = sc["sun_dir"]
    u, v = _basis(w)
    g = ((torch.arange(res, device=w.device, dtype=torch.float32) + 0.5) / res * 2 - 1) * extent
    a, b = torch.meshgrid(g, g, indexing="ij")
    centre = torch.tensor([0.0, 0.0, 0.0], device=w.device)
    o = (centre + u[:, None] * a.reshape(1, -1, 1) + v[:, None] * b.reshape(1, -1, 1)
         + w[:, None] * (2 * extent))
    d = (-w)[:, None].expand_as(o).contiguous()
    ts = _hit_spheres(o, d, sc).min(-1).values
    tb = _hit_boxes(o, d, sc["box_c"], sc["box_h"], sc["box_yaw"])[0].min(-1).values
    tc = _hit_cones(o, d, sc)[0].min(-1).values
    trc, trh, tryaw = _trunk_boxes(sc)
    tt = _hit_boxes(o, d, trc, trh, tryaw)[0].min(-1).values
    tter = terrain_march(o, d, sc, steps=24)
    z = torch.minimum(torch.minimum(torch.minimum(ts, tb), torch.minimum(tc, tt)), tter)
    return z.reshape(B, res, res), extent

def sample_shadow_map(sm, extent, sc, p, n, res=SHADOW_MAP_RES):
    B = p.shape[0]
    w = sc["sun_dir"]
    u, v = _basis(w)
    a = _dot(p, u[:, None]) / extent
    b = _dot(p, v[:, None]) / extent
    d = 2 * extent - _dot(p, w[:, None])
    ndl = _dot(n, w[:, None]).clamp(min=0)
    bias = 0.04 + 0.5 * (1 - ndl)
    fa = (a * 0.5 + 0.5) * res - 0.5
    fb = (b * 0.5 + 0.5) * res - 0.5
    flat = sm.reshape(B, -1)
    vis = torch.zeros_like(d)
    for da in (0, 1):
        for db in (0, 1):
            ia = (fa.floor() + da).clamp(0, res - 1).long()
            ib = (fb.floor() + db).clamp(0, res - 1).long()
            vis = vis + (d - bias <= torch.gather(flat, 1, ia * res + ib)).float()
    inside = ((a.abs() < 1) & (b.abs() < 1)).float()
    return vis * 0.25 * inside + (1 - inside)

def ssao(gb, res, taps=6, radius=2.4, gen=None):
    import math as _m
    h, w = _hw(res)
    B = gb["depth"].shape[0]
    dep = gb["depth"].reshape(B, h, w)
    occ = torch.zeros_like(dep)
    ang = torch.rand(taps, generator=gen) * 2 * _m.pi
    for k in range(taps):
        rad = 2 + int(6 * ((k + 1) / taps))
        dy = int(round(float(torch.sin(ang[k])) * rad))
        dx = int(round(float(torch.cos(ang[k])) * rad))
        diff = dep - torch.roll(dep, shifts=(dy, dx), dims=(1, 2))
        occ = occ + ((diff > 0.02) & (diff < radius)).float() * (1 - diff.clamp(0, radius) / radius)
    return (1 - occ / taps * 1.7).clamp(0.05, 1.0).reshape(B, -1)

def ssr(gb, sc, res, base, steps=10, stride=0.4):
    h, w = _hw(res)
    B = base.shape[0]
    fwd, right, up = camera_basis(sc)
    tan = torch.tan(sc["fov"] * 0.5)[:, None]
    aspect = w / h
    n = gb["n"]
    r = _norm(gb["rd"] - 2 * _dot(gb["rd"], n)[..., None] * n)
    col_flat = base.permute(0, 2, 3, 1).reshape(B, h * w, 3)
    dep_flat = gb["depth"]
    hit_col = torch.zeros_like(col_flat)
    found = torch.zeros_like(dep_flat)
    pos = gb["p"] + n * 0.02
    for s in range(1, steps + 1):
        q = pos + r * (stride * s * (1 + 0.14 * s))
        rel = q - sc["cam"][:, None]
        z = _dot(rel, fwd[:, None])
        sx = _dot(rel, right[:, None]) / (z * tan * aspect + 1e-6)
        sy = _dot(rel, up[:, None]) / (z * tan + 1e-6)
        ix = ((sx * 0.5 + 0.5) * w).long()
        iy = ((-sy * 0.5 + 0.5) * h).long()
        on = (z > 0.05) & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        idx = iy.clamp(0, h - 1) * w + ix.clamp(0, w - 1)
        zbuf = torch.gather(dep_flat, 1, idx)
        take = ((z > zbuf + 0.03) & (z < zbuf + 1.4) & on & (found < 0.5)).float()
        hit_col = hit_col + take[..., None] * torch.gather(col_flat, 1,
                                                           idx[..., None].expand(-1, -1, 3))
        found = found + take
    return hit_col, found

def render_game(sc, res=64, gb=None, gen=None, ao_taps=6, ssr_steps=10):
    gb = gb if gb is not None else gbuffer(sc, res)
    n, alb, rough, metal = gb["n"], gb["alb"], gb["rough"], gb["metal"]
    v, p = -gb["rd"], gb["p"]
    l = sc["sun_dir"][:, None].expand_as(n)

    sm, extent = shadow_map(sc)
    vis = sample_shadow_map(sm, extent, sc, p, n)
    direct = _brdf(n, v, l, alb, rough, metal, gb["sss"]) * sc["sun_col"][:, None] * vis[..., None]
    ao = ssao(gb, res, taps=ao_taps, gen=gen)[..., None]
    col = direct + alb * (1 - metal[..., None]) * sc["sky_h"][:, None] * 0.55 * ao

    base = _composite(col, gb["hit"], gb["depth"], gb["rd"], sc, res)
    refl, found = ssr(gb, sc, res, base, steps=ssr_steps)
    r = _norm(gb["rd"] - 2 * _dot(gb["rd"], n)[..., None] * n)
    env = refl + sky(r, sc) * (1 - found[..., None]) * 0.45
    k = (metal * (1 - rough * 0.7) * 0.9).clamp(0, 1)[..., None]
    col = col * (1 - k) + env * alb.clamp(min=0.25) * k

    # water: sky-only reflection plus a depth tint. No refraction of the scene.
    fres = _fresnel(_dot(v, n).clamp(min=0), sc["ior"][:, None])[..., None]
    is_water = (gb["kind"] == KIND_WATER)[..., None].float()
    water = sky(r, sc) * fres + sc["water_col"][:, None] * (1 - fres)
    water = water + refl * found[..., None] * fres * 0.8
    col = col * (1 - is_water) + water * is_water

    # glass: alpha blend against the sky, so what is *behind* it is simply wrong
    is_glass = (gb["trans"] * (1 - is_water[..., 0]))[..., None]
    glassy = sky(gb["rd"], sc) * 0.75 + col * 0.25 + sky(r, sc) * fres * 0.5
    col = col * (1 - is_glass) + glassy * is_glass

    col_img = _composite(col, gb["hit"], gb["depth"], gb["rd"], sc, res)
    tr, add = smoke_cheap(gb["ro"], gb["rd"], gb["t"], sc)
    h, w = _hw(res)
    B = col_img.shape[0]
    tr_i = tr.reshape(B, h, w, 1).permute(0, 3, 1, 2)
    add_i = _tonemap(add, sc, res)
    return (col_img * tr_i + add_i * (1 - tr_i)).clamp(0, 1)


GBUF_CHANNELS = 23   # 16 + behind_rgb(3) + behind_depth(1) + env_refl(3)

def _emissive(sc, mat_flag):
    """Radiance emitted by a surface (zero for everything but emissive spheres)."""
    return sc["emissive"][:, None] * mat_flag[..., None]

def gbuffer(sc, res, terrain_steps=32):
    """Primary hit plus the second layer behind any transparent pixel."""
    ro, rd = camera_rays(sc, res)
    h = intersect(ro, rd, sc, terrain_steps=terrain_steps)
    fwd = camera_basis(sc)[0]
    h["ro"], h["rd"] = ro, rd
    h["depth"] = _dot(h["p"] - sc["cam"][:, None], fwd[:, None]).clamp(min=0)
    h["hitf"] = h["hit"][..., None].float()

    # emissive flag needs the sphere material, which intersect() does not carry
    K = sc["sph_c"].shape[1]
    B, P, _ = ro.shape
    ts, ki = _hit_spheres(ro, rd, sc).min(-1)
    s_mat = torch.gather(sc["sph_mat"][:, None].expand(B, P, K), 2, ki[..., None]).squeeze(2)
    h["emis"] = ((h["kind"] == KIND_SPHERE) & (s_mat == MAT_EMISSIVE)).float()
    s_frost = torch.gather(sc["sph_frost"][:, None].expand(B, P, K), 2, ki[..., None]).squeeze(2)
    h["frost"] = torch.where(h["kind"] == KIND_SPHERE, s_frost, torch.zeros_like(s_frost))

    # ---- depth peel: step past the transparent surface and hit again --------
    peel_o = h["p"] + rd * 1e-2
    h2 = intersect(peel_o, rd, sc, terrain_steps=20, cheap=True)
    lit2 = _brdf(h2["n"], -rd, sc["sun_dir"][:, None].expand_as(h2["n"]),
                 h2["alb"], h2["rough"], h2["metal"], h2["sss"])
    vis2 = 1 - occluded(h2["p"] + h2["n"] * 3e-3, sc["sun_dir"][:, None].expand_as(h2["n"]),
                        torch.full_like(h2["t"], 1e4), sc, terrain_steps=8)
    behind = (lit2 * sc["sun_col"][:, None] * vis2[..., None]
              + h2["alb"] * sc["sky_h"][:, None] * 0.5)
    behind = torch.where(h2["hit"][..., None], behind, sky(rd, sc))
    h["behind_rgb"] = (behind * sc["exposure"][:, None, None])
    h["behind_rgb"] = (h["behind_rgb"] / (1 + h["behind_rgb"])).clamp(0, 1).pow(1 / 2.2)
    h["behind_depth"] = torch.where(h2["hit"], h2["t"], torch.full_like(h2["t"], 25.0))

    # ---- environment probe along the reflection direction -------------------
    refl_dir = _norm(rd - 2 * _dot(rd, h["n"])[..., None] * h["n"])
    probe = env_probe(sc)
    h["env_refl"] = sample_env(probe, refl_dir)      # the SCENE, not just sky
    h["env_probe"] = probe
    return h

def caustic_gain(sc, p):
    """Sunlight focused by glass spheres onto the point `p`.

    A sphere of radius R and index n focuses parallel light at f = nR/(2(n-1))
    beyond its centre. Points near that focus get a large intensity gain; the
    gain falls off with distance from the focal point and with how far the point
    sits from the sphere's optical axis.
    """
    glass = (sc["sph_mat"] == MAT_GLASS).float()          # [B,K]
    if float(glass.sum()) == 0:
        return torch.zeros_like(p[..., 0])
    sun = sc["sun_dir"][:, None, :]                        # [B,1,3]
    c = sc["sph_c"]                                        # [B,K,3]
    R = sc["sph_r"]
    n = sc["ior"][:, None]
    f = (n * R / (2 * (n - 1)).clamp(min=1e-3))            # [B,K]

    rel = p[:, :, None, :] - c[:, None, :, :]              # [B,P,K,3]
    along = (rel * sun[:, None]).sum(-1)                   # distance along the sun axis
    perp = (rel - along[..., None] * sun[:, None]).norm(dim=-1)

    near_focus = torch.exp(-((along - f[:, None, :]) / (0.55 * R[:, None, :])) ** 2)
    on_axis = torch.exp(-(perp / (0.9 * R[:, None, :])) ** 2)
    gain = (near_focus * on_axis * glass[:, None, :] * (along > 0).float()).sum(-1)
    return gain * sc["caustic"][:, None]

def _sun_dir_sampled(sc, shape, gen):
    w = sc["sun_dir"][:, None]
    u, v = _basis(sc["sun_dir"])
    a = torch.randn((shape[0], 1, 1), generator=gen) * sc["sun_angle"][:, None, None]
    b = torch.randn((shape[0], 1, 1), generator=gen) * sc["sun_angle"][:, None, None]
    return _norm(w + u[:, None] * a + v[:, None] * b).expand(shape)

def _sun_lit(sc, p, n, v, alb, rough, metal, sss, gen, terrain_steps=10, caustics=True):
    l = _sun_dir_sampled(sc, n.shape, gen)
    vis = 1.0 - occluded(p + n * 3e-3, l, torch.full_like(p[..., 0], 1e4), sc, terrain_steps)
    if caustics:
        vis = vis + caustic_gain(sc, p) * 0.5      # focused light adds on top
    return _brdf(n, v, l, alb, rough, metal, sss) * sc["sun_col"][:, None] * vis[..., None]

def _cosine_dir(n, gen):
    d = _norm(torch.randn((n.shape[0], 1, 3), generator=gen))
    return _norm(d * 0.85 + n)

def render_rt(sc, res=64, gb=None, gen=None, sun_samples=3, gi_samples=1, gi_bounces=2,
              smoke_steps=8, terrain_steps=32, dispersion=True):
    gb = gb if gb is not None else gbuffer(sc, res, terrain_steps)
    n, alb, rough, metal, sss = gb["n"], gb["alb"], gb["rough"], gb["metal"], gb["sss"]
    p, v = gb["p"], -gb["rd"]

    direct = torch.zeros_like(alb)
    for _ in range(max(1, sun_samples)):
        direct = direct + _sun_lit(sc, p, n, v, alb, rough, metal, sss, gen)
    col = direct / max(1, sun_samples)
    col = col + _emissive(sc, gb["emis"])          # emitters glow

    ind = torch.zeros_like(alb)
    for _ in range(max(1, gi_samples)):
        thr = alb * (1 - metal[..., None])
        op, on = p, n
        for _b in range(max(1, gi_bounces)):
            bd = _cosine_dir(on, gen)
            bo = op + on * 3e-3
            bh = intersect(bo, bd, sc, terrain_steps=14, cheap=True)
            miss = (~bh["hit"])[..., None].float()
            lit = _sun_lit(sc, bh["p"], bh["n"], -bd, bh["alb"], bh["rough"],
                           bh["metal"], bh["sss"], gen).clamp(max=6.0)
            ind = ind + thr * (lit * (1 - miss) + sky(bd, sc) * miss)
            thr = thr * bh["alb"] * (1 - bh["metal"][..., None]) * (1 - miss)
            op, on = bh["p"], bh["n"]
    col = col + ind / max(1, gi_samples)

    jit = _norm(torch.randn((n.shape[0], 1, 3), generator=gen))
    r = _norm(gb["rd"] - 2 * _dot(gb["rd"], n)[..., None] * n + jit * rough[..., None] * 0.3)
    ro2 = p + n * 3e-3
    rh = intersect(ro2, r, sc, terrain_steps=16, cheap=True)
    rmiss = (~rh["hit"])[..., None].float()
    rlit = (_sun_lit(sc, rh["p"], rh["n"], -r, rh["alb"], rh["rough"], rh["metal"],
                     rh["sss"], gen) + rh["alb"] * sc["sky_h"][:, None] * 0.3)
    refl = rlit * (1 - rmiss) + sky(r, sc) * rmiss
    k = (metal * (1 - rough * 0.7) * 0.9).clamp(0, 1)[..., None]
    col = col * (1 - k) + refl * alb.clamp(min=0.25) * k

    # ---- refraction, with dispersion and frosted transmission ---------------
    fres = _fresnel(_dot(v, n).clamp(min=0), sc["ior"][:, None])[..., None]
    is_water = (gb["kind"] == KIND_WATER)[..., None].float()
    frost = gb.get("frost", torch.zeros_like(rough))
    fjit = _norm(torch.randn((n.shape[0], 1, 3), generator=gen))

    spread = sc["dispersion"][:, None] if dispersion else torch.zeros_like(sc["ior"][:, None])
    chans = []
    for ci, off in enumerate((-1.0, 0.0, 1.0)):            # R, G, B
        eta = (1.0 / (sc["ior"][:, None] + off * spread)).expand_as(gb["t"])
        rdir = _refract(gb["rd"], n, eta)
        rdir = _norm(rdir + fjit * frost[..., None])
        th = intersect(p - n * 3e-3, rdir, sc, terrain_steps=16, cheap=True)
        tmiss = (~th["hit"])[..., None].float()
        tlit = (_sun_lit(sc, th["p"], th["n"], -rdir, th["alb"], th["rough"],
                         th["metal"], th["sss"], gen)
                + th["alb"] * sc["sky_h"][:, None] * 0.35)
        beh = tlit * (1 - tmiss) + sky(rdir, sc) * tmiss
        # Beer's law under water
        depth_in = torch.where(th["hit"], th["t"], torch.zeros_like(th["t"]))
        absorb = torch.exp(-depth_in * sc["water_absorb"][:, None] * is_water[..., 0] * 0.35)
        chans.append((beh * absorb[..., None])[..., ci:ci + 1])
        if not dispersion:
            beh_full = beh * absorb[..., None]
            chans = [beh_full[..., 0:1], beh_full[..., 1:2], beh_full[..., 2:3]]
            break
    behind = torch.cat(chans, -1)

    trans = gb["trans"][..., None]
    tinted = behind * torch.where(is_water > 0, sc["water_col"][:, None] * 2.2 + 0.25,
                                  alb.clamp(min=0.3))
    col = col * (1 - trans) + (refl * fres + tinted * (1 - fres)) * trans

    col_img = _composite(col, gb["hit"], gb["depth"], gb["rd"], sc, res)
    tr, add = smoke_marched(gb["ro"], gb["rd"], gb["t"], sc, steps=smoke_steps, gen=gen)
    h, w = _hw(res)
    B = col_img.shape[0]
    tr_i = tr.reshape(B, h, w, 1).permute(0, 3, 1, 2)
    return (col_img * tr_i + _tonemap(add, sc, res) * (1 - tr_i)).clamp(0, 1)

ENV_H, ENV_W = 48, 96


def env_probe(sc, height=2.2, steps=20, origin=None, exclude_sph=None):
    """Render the scene into a lat-long map from a probe point.

    This is the difference between handing the model *the frame* and handing it
    *the environment*. Every other channel is camera-space: it can only describe
    what is visible from the current viewpoint. A reflection shows geometry that
    is somewhere else entirely -- measured, 92% of reflection rays hit scene
    geometry and only 8% see sky -- so a screen-space buffer cannot answer the
    question no matter how large the network is.

    One probe costs 48x96 = 4,608 rays per scene, about the same as the shadow
    map, and it is shared by every pixel. Like a game's reflection probe it has
    parallax error for nearby objects, being captured from one point rather than
    from each shading point, but it carries the off-screen content that was
    simply absent before.
    """
    B = sc["cam"].shape[0]
    dev = sc["cam"].device
    v = (torch.arange(ENV_H, device=dev, dtype=torch.float32) + 0.5) / ENV_H
    u = (torch.arange(ENV_W, device=dev, dtype=torch.float32) + 0.5) / ENV_W
    theta = v[:, None] * math.pi
    phi = u[None, :] * 2 * math.pi
    dirs = torch.stack([torch.sin(theta) * torch.cos(phi),
                        torch.cos(theta).expand(ENV_H, ENV_W),
                        torch.sin(theta) * torch.sin(phi)], -1).reshape(1, -1, 3)
    rd = dirs.expand(B, ENV_H * ENV_W, 3).contiguous()

    if origin is None:
        zeros = torch.zeros(B, device=dev)
        origin = torch.stack([zeros, terrain_h(zeros, zeros, sc) + height, zeros], -1)
    ro = origin[:, None].expand_as(rd).contiguous()

    h = intersect(ro, rd, sc, terrain_steps=steps, cheap=True, exclude_sph=exclude_sph)
    sun = sc["sun_dir"][:, None].expand_as(h["n"])
    vis = 1 - occluded(h["p"] + h["n"] * 3e-3, sun, torch.full_like(h["t"], 1e4), sc,
                       terrain_steps=8)
    lit = (_brdf(h["n"], -rd, sun, h["alb"], h["rough"], h["metal"], h["sss"])
           * sc["sun_col"][:, None] * vis[..., None]
           + h["alb"] * sc["sky_h"][:, None] * 0.5)
    col = torch.where(h["hit"][..., None], lit, sky(rd, sc))
    col = col * sc["exposure"][:, None, None]
    col = (col / (1 + col)).clamp(0, 1).pow(1 / 2.2)
    return col.reshape(B, ENV_H, ENV_W, 3).permute(0, 3, 1, 2).contiguous()


def sample_env(probe, d):
    """Look up the probe along directions `d` [B,P,3] -> [B,P,3] colour."""
    B, P, _ = d.shape
    theta = torch.acos(d[..., 1].clamp(-1, 1)) / math.pi
    phi = (torch.atan2(d[..., 2], d[..., 0]) / (2 * math.pi)) % 1.0
    grid = torch.stack([phi * 2 - 1, theta * 2 - 1], -1).reshape(B, P, 1, 2)
    out = F.grid_sample(probe, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out.squeeze(-1).permute(0, 2, 1)


def motion_vectors(gb_now, sc_prev, res):
    """Where each visible surface point was in the previous frame.

    Exact, not estimated: the G-buffer already holds the world position of every
    pixel, so projecting that point through the *previous* camera gives the true
    screen-space motion. Engines produce this buffer for TAA and upscaling, so
    asking for it is not asking for anything a game does not already have.

    Returns (flow [B,2,H,W] in normalised screen units, valid [B,1,H,W]).
    """
    h, w = _hw(res)
    B = gb_now["p"].shape[0]
    fwd, right, up = camera_basis(sc_prev)
    tan = torch.tan(sc_prev["fov"] * 0.5)[:, None]
    aspect = w / h
    rel = gb_now["p"] - sc_prev["cam"][:, None]
    z = _dot(rel, fwd[:, None])
    sx = _dot(rel, right[:, None]) / (z * tan * aspect + 1e-6)
    sy = _dot(rel, up[:, None]) / (z * tan + 1e-6)
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32), indexing="ij")
    cur_x = ((xs + 0.5) / w * 2 - 1).reshape(1, -1).expand(B, -1)
    cur_y = (1 - (ys + 0.5) / h * 2).reshape(1, -1).expand(B, -1)
    flow = torch.stack([sx - cur_x, sy - cur_y], 1)
    valid = ((z > 0.05) & (sx.abs() < 1) & (sy.abs() < 1) & gb_now["hit"]).float()[:, None]
    return flow.reshape(B, 2, h, w), valid.reshape(B, 1, h, w)


def warp_previous(prev_rgb, flow):
    """Reproject the previous frame into the current one using the flow."""
    B, _, h, w = prev_rgb.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32), indexing="ij")
    gx = ((xs + 0.5) / w * 2 - 1)[None].expand(B, -1, -1) + flow[:, 0]
    gy = (1 - (ys + 0.5) / h * 2)[None].expand(B, -1, -1) + flow[:, 1]
    grid = torch.stack([gx, -gy], -1)
    return F.grid_sample(prev_rgb, grid, mode="bilinear", padding_mode="border",
                         align_corners=False)


def model_input(game_rgb, gb, sc, res):
    h, w = _hw(res)
    B = game_rgb.shape[0]

    def img(x, c):
        return x.reshape(B, h, w, c).permute(0, 3, 1, 2)

    ndl = _dot(gb["n"], sc["sun_dir"][:, None]).clamp(min=0)
    tr, _ = smoke_cheap(gb["ro"], gb["rd"], gb["t"], sc)
    return torch.cat([
        game_rgb,
        img(gb["alb"], 3),
        img(gb["n"] * 0.5 + 0.5, 3),
        img((gb["depth"] / 25.0).clamp(0, 1)[..., None], 1),
        img(gb["rough"][..., None], 1),
        img(gb["metal"][..., None], 1),
        img(ndl[..., None], 1),
        img(gb["trans"][..., None], 1),
        img(gb["sss"][..., None], 1),
        img(1 - tr, 1),
        img(gb["behind_rgb"], 3),                                   # new in v3
        img((gb["behind_depth"] / 25.0).clamp(0, 1)[..., None], 1),  # new in v3
        img(gb["env_refl"], 3),
    ], 1).contiguous()


TEMPORAL_CHANNELS = 5      # warped previous prediction (3) + validity (1) + mode flag (1)


def add_temporal(x, prev_rgb=None, flow=None, valid=None):
    """Attach the animation-mode channels.

    In image mode these are zeros and the flag is 0, so one network handles both
    modes and the flag tells it which it is looking at. In animation mode it
    receives the previous frame reprojected into this one plus a validity mask
    marking disocclusions -- the places reprojection cannot help and the model
    has to synthesise from scratch.
    """
    B, _, h, w = x.shape
    if prev_rgb is None:
        pad = torch.zeros(B, TEMPORAL_CHANNELS, h, w, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], 1)
    warped = warp_previous(prev_rgb, flow) * valid
    flag = torch.ones(B, 1, h, w, dtype=x.dtype, device=x.device)
    return torch.cat([x, warped, valid, flag], 1)

def render_pair(indices=None, batch=8, res=64, sc=None, seed=None, sun_samples=3,
                gi_samples=1, smoke_steps=8, terrain_steps=32, ao_taps=6,
                ssr_steps=10, dispersion=True, **scene_kw):
    """Render one pair. Pass `indices` for guaranteed-unique scenes."""
    gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    with torch.no_grad():
        if sc is None:
            sc = make_scene_batch(indices, **scene_kw)
        gb = gbuffer(sc, res, terrain_steps)
        game = render_game(sc, res, gb=gb, gen=gen, ao_taps=ao_taps, ssr_steps=ssr_steps)
        rt = render_rt(sc, res, gb=gb, gen=gen, sun_samples=sun_samples,
                       gi_samples=gi_samples, smoke_steps=smoke_steps,
                       terrain_steps=terrain_steps, dispersion=dispersion)
        x = model_input(game, gb, sc, res)
    return x, rt

def render_eval_pair(res=(128, 192), sc=None, **kw):
    """High-quality render of the held-out scenes; pass `sc` for a different set."""
    from .scenes import eval_scenes
    sc = eval_scenes() if sc is None else sc
    kw.setdefault("sun_samples", 8)
    kw.setdefault("gi_samples", 3)
    kw.setdefault("smoke_steps", 16)
    kw.setdefault("terrain_steps", 48)
    return render_pair(sc=sc, res=res, seed=13, **kw)


# ---------------------------------------------------------------------------
# animation mode: a scene that moves, and the frame pair it produces
# ---------------------------------------------------------------------------

def advance_scene(sc, gen=None, cam_move=0.35, obj_move=0.25):
    """Take one small step of 'gameplay': the camera moves and objects shift.

    This is what makes animation mode meaningful -- the second frame must differ
    for a reason the model can see in the scene description, not by an arbitrary
    re-randomisation.
    """
    def r(*shape):
        return (torch.rand(*shape, generator=gen) - 0.5) * 2

    nxt = dict(sc)
    B = sc["cam"].shape[0]
    nxt["cam"] = sc["cam"] + r(B, 3) * cam_move
    nxt["look"] = sc["look"] + r(B, 3) * cam_move * 0.6
    K = sc["sph_c"].shape[1]
    step = r(B, K, 3) * obj_move
    step[..., 1] = 0.0                                  # objects slide, not fly
    moved = sc["sph_c"] + step
    moved[..., 1] = terrain_h(moved[..., 0], moved[..., 2], sc) + sc["sph_r"] * 0.92
    nxt["sph_c"] = moved
    nxt["box_yaw"] = sc["box_yaw"] + r(B, sc["box_yaw"].shape[1]) * 0.3
    return nxt


def render_sequence(indices=None, batch=8, res=64, sc=None, gen=None, **kw):
    """Render frame A, advance the scene, render frame B, and the motion between.

    Returns a dict with both frames' model inputs and targets plus the flow, so
    the trainer can build an image-mode sample from A and an animation-mode
    sample from B conditioned on A.
    """
    from .scenes import make_scene_batch
    if sc is None:
        sc = make_scene_batch(indices, **{k: v for k, v in kw.items()
                                          if k in ("n_spheres", "n_boxes", "n_plants", "hard")})
    rkw = {k: v for k, v in kw.items() if k not in ("n_spheres", "n_boxes", "n_plants", "hard")}
    with torch.no_grad():
        a_x, a_y, a_gb, a_sc = _one_frame(sc, res, gen, rkw)
        sc2 = advance_scene(sc, gen=gen)
        b_x, b_y, b_gb, b_sc = _one_frame(sc2, res, gen, rkw)
        flow, valid = motion_vectors(b_gb, sc, res)
    return dict(a_x=a_x, a_y=a_y, b_x=b_x, b_y=b_y, flow=flow, valid=valid,
                sc_a=sc, sc_b=sc2)


def _one_frame(sc, res, gen, rkw):
    gb = gbuffer(sc, res, rkw.get("terrain_steps", 32))
    game = render_game(sc, res, gb=gb, gen=gen,
                       ao_taps=rkw.get("ao_taps", 6), ssr_steps=rkw.get("ssr_steps", 10))
    rt = render_rt(sc, res, gb=gb, gen=gen,
                   sun_samples=rkw.get("sun_samples", 3),
                   gi_samples=rkw.get("gi_samples", 1),
                   smoke_steps=rkw.get("smoke_steps", 8),
                   terrain_steps=rkw.get("terrain_steps", 32),
                   dispersion=rkw.get("dispersion", True))
    return model_input(game, gb, sc, res), rt, gb, sc
