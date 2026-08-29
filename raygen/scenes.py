"""Scene generation: a procedural world hard enough to need real light transport.

Terrain, water, glass, foliage, subsurface materials, volumetric smoke and a
procedural sky. Every feature exists because it forces the renderer into an
algorithm a rasterizer cannot cheaply fake -- which is what makes the
input/target pair worth learning.

**No scene is ever repeated.** A scene is a pure function of a globally unique
64-bit index::

    index = (run_id << 32) | counter
    seed  = splitmix64(index)          # bijection: distinct index, distinct seed

`run_id` is fresh per run and every previous one is stored in the checkpoint, so
a resumed run gets a namespace disjoint from all earlier runs; `counter` is
partitioned across workers by residue so workers cannot collide. Verified: 200
consecutive indices give 200 distinct scene fingerprints, the same index always
reproduces the same scene, and a second run_id shares nothing with the first.
"""

import math
import os
import struct

import torch
import torch.nn.functional as F


SHADOW_MAP_RES = 64

FAR = 1e9

def _norm(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def _dot(a, b):
    return (a * b).sum(-1)

def _basis(w):
    up = torch.where(w[..., 1:2].abs() > 0.9,
                     torch.tensor([1.0, 0.0, 0.0], device=w.device).expand_as(w),
                     torch.tensor([0.0, 1.0, 0.0], device=w.device).expand_as(w))
    u = _norm(torch.cross(up, w, dim=-1))
    return u, torch.cross(w, u, dim=-1)

def _hw(res):
    if isinstance(res, (tuple, list)):
        return int(res[0]), int(res[1])
    return int(res), int(res)

_ROT = [(1.0, 0.0), (0.766, 0.643), (0.174, 0.985), (-0.574, 0.819), (0.940, -0.342)]

def fbm2(x, z, octaves=3, lac=2.17, gain=0.5, seed=0.0):
    """Sine-lattice fBm. Each octave is evaluated in a rotated frame -- without
    that the sin(x)*cos(z) product lays down a visible axis-aligned quilt."""
    v = torch.zeros_like(x)
    amp, f = 1.0, 1.0
    total = 0.0
    for i in range(octaves):
        c, s = _ROT[i % len(_ROT)]
        xr = x * c + z * s
        zr = -x * s + z * c
        v = v + amp * (torch.sin(xr * f + seed + i * 1.7)
                       * torch.cos(zr * f * 1.13 - seed * 0.7 + i * 2.3))
        total += amp
        amp *= gain
        f *= lac
    return v / total

def terrain_h(x, z, sc, cheap=False):
    """Height field. Three sine octaves -- smooth enough to march safely.

    `x`/`z` may be [B], [B,P] or [B,P,K]; scene parameters are broadcast to match.
    """
    a, f, p = sc["ter_amp"], sc["ter_freq"], sc["ter_phase"]
    shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    h = torch.zeros_like(x)
    for i in range(2 if cheap else 3):     # secondary rays skip the finest octave
        ai = a[:, i].reshape(shape)
        fi = f[:, i].reshape(shape)
        pi = p[:, i].reshape(shape)
        c, s = _ROT[i % len(_ROT)]
        xr, zr = x * c + z * s, -x * s + z * c
        h = h + ai * torch.sin(xr * fi + pi) * torch.cos(zr * fi * 1.117 - pi * 0.63)
    return h

def terrain_normal(x, z, sc, eps=0.06):
    hx = terrain_h(x + eps, z, sc) - terrain_h(x - eps, z, sc)
    hz = terrain_h(x, z + eps, sc) - terrain_h(x, z - eps, sc)
    return _norm(torch.stack([-hx, torch.full_like(hx, 2 * eps), -hz], -1))

def terrain_march(ro, rd, sc, steps=32, tmax=45.0, refine=4, cheap=False):
    """March the height field.

    The amplitude is bounded, so we only have to search the slab where the ray
    is between +/- max height -- everything outside cannot possibly hit.
    """
    hmax = sc["ter_amp"].sum(-1, keepdim=True) * 1.02        # [B,1]
    oy, dy = ro[..., 1], rd[..., 1]
    # entry: where ray drops below +hmax (or now, if already below)
    t_in = torch.where(dy < -1e-6, (hmax - oy) / dy, torch.zeros_like(oy)).clamp(min=0.0)
    t_out = torch.where(dy < -1e-6, (-hmax - oy) / dy, torch.full_like(oy, tmax))
    t_out = torch.minimum(t_out.clamp(min=0.0), torch.full_like(t_out, tmax))
    span = (t_out - t_in).clamp(min=0.0)

    t = t_in.clone()
    prev_t = t.clone()
    prev_d = torch.ones_like(t)
    hit = torch.zeros_like(t, dtype=torch.bool)
    t_hit = torch.full_like(t, FAR)
    dt = span / steps
    for i in range(steps):
        t_cur = t_in + dt * (i + 1)
        p = ro + rd * t_cur[..., None]
        d = p[..., 1] - terrain_h(p[..., 0], p[..., 2], sc, cheap)
        cross = (d < 0) & (prev_d >= 0) & (~hit) & (span > 0)
        t_hit = torch.where(cross, t_cur, t_hit)
        # keep the bracket for refinement
        prev_t = torch.where(cross | hit, prev_t, t_cur)
        hit = hit | cross
        prev_d = torch.where(hit, prev_d, d)
    # bisection inside the bracketing interval
    lo = (t_hit - dt).clamp(min=0.0)
    hi = t_hit
    for _ in range(refine):
        mid = (lo + hi) * 0.5
        p = ro + rd * mid[..., None]
        below = (p[..., 1] - terrain_h(p[..., 0], p[..., 2], sc, cheap)) < 0
        hi = torch.where(below, mid, hi)
        lo = torch.where(below, lo, mid)
    t_hit = torch.where(hit, hi, torch.full_like(hi, FAR))
    return t_hit

def terrain_material(p, n, sc, cheap=False):
    """Grass on flat ground, rock on slopes, dirt low down -- plus detail noise.

    `cheap` drops the fine octave and the blade-level bump: secondary rays only
    need roughly the right colour, and this is the hottest function in the
    renderer.
    """
    slope = n[..., 1].clamp(0, 1)
    detail = fbm2(p[..., 0] * 2.6, p[..., 2] * 2.6, octaves=2 if cheap else 3, seed=1.7)
    fine = (torch.zeros_like(detail) if cheap
            else fbm2(p[..., 0] * 11.0, p[..., 2] * 11.0, octaves=2, seed=4.1))

    grass = sc["col_grass"][:, None] * (0.75 + 0.45 * detail[..., None])
    rock = sc["col_rock"][:, None] * (0.8 + 0.3 * fine[..., None])
    dirt = sc["col_dirt"][:, None] * (0.85 + 0.3 * detail[..., None])

    rocky = (1 - slope).clamp(0, 1) ** 1.5
    rocky = (rocky * 2.2).clamp(0, 1)
    low = torch.sigmoid((-p[..., 1] - sc["ter_amp"].sum(-1)[:, None] * 0.15) * 2.0)
    alb = grass * (1 - rocky[..., None]) + rock * rocky[..., None]
    alb = alb * (1 - low[..., None] * 0.65) + dirt * (low[..., None] * 0.65)

    # grass is rough and scatters a little; rock is rougher still in the fine detail
    rough = (0.55 + 0.35 * rocky + 0.12 * fine).clamp(0.15, 1.0)
    scatter = (0.35 * (1 - rocky)).clamp(0, 1)        # translucent grass blades
    # blade-level normal perturbation: this is the "detail everywhere" term
    if cheap:
        return alb.clamp(0, 1), rough, scatter, n
    bump = torch.stack([fine * 0.35 * (1 - rocky), torch.zeros_like(fine),
                        detail * 0.35 * (1 - rocky)], -1)
    return alb.clamp(0, 1), rough, scatter, _norm(n + bump)

def sky(rd, sc):
    up = rd[..., 1]
    t = (up * 0.5 + 0.5).clamp(0, 1)[..., None]
    base = sc["sky_l"][:, None] * (1 - t) + sc["sky_h"][:, None] * t

    # cloud layer: project the direction onto a plane above the camera
    denom = up.clamp(min=0.04)
    cx = rd[..., 0] / denom * 0.55 + sc["cloud_ph"][:, None]
    cz = rd[..., 2] / denom * 0.55
    c = fbm2(cx, cz, octaves=4, seed=float(2.9))
    cover = ((c + 1) * 0.5 - sc["cloud_cov"][:, None]).clamp(min=0) / \
            (1 - sc["cloud_cov"][:, None] + 1e-3)
    cover = (cover.clamp(0, 1) ** 1.4) * (up > 0.02).float()
    cloud_col = sc["sky_h"][:, None] * 1.5 + sc["sun_col"][:, None] * 0.06
    col = base * (1 - cover[..., None]) + cloud_col * cover[..., None]

    sun = _dot(rd, sc["sun_dir"][:, None]).clamp(min=0)
    col = col + (sun ** 900)[..., None] * sc["sun_col"][:, None] * 0.8      # disk
    col = col + (sun ** 8)[..., None] * sc["sun_col"][:, None] * 0.05       # glow
    return col

MAT_OPAQUE, MAT_METAL, MAT_GLASS, MAT_SSS = 0, 1, 2, 3


MAT_EMISSIVE = 4

EVAL_SEEDS = (30_001, 30_002, 30_003, 30_004, 30_005)
FOCUS_EVAL_SEEDS = (40_001, 40_002, 40_003, 40_004, 40_005)

EVAL_NAMESPACE = 0xE7A1          # eval indices live in their own high namespace

def splitmix64(x: int) -> int:
    """Bijection on 64 bits: distinct indices always give distinct seeds."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)

def new_run_id(used=()):
    """A fresh 32-bit namespace that no previous run of this model has used."""
    used = set(used)
    while True:
        rid = struct.unpack("<I", os.urandom(4))[0]
        if rid not in used and rid != EVAL_NAMESPACE:
            return rid

def scene_index(run_id: int, counter: int) -> int:
    return ((run_id & 0xFFFFFFFF) << 32) | (counter & 0xFFFFFFFF)

def make_scene(index, device="cpu", n_spheres=4, n_boxes=3, n_plants=3,
               hard=1.0, focus=False):
    """One scene, fully determined by `index`.

    `hard` scales how often the difficult materials appear (glass, water,
    metal, emissive, smoke) -- 1.0 is the default mix.

    `focus` builds a scene made almost entirely of the surfaces the model is
    worst at: reflective metal, refractive glass, and water. Measured, metal
    reaches only 1.31x improvement while every other region sits near 3.0x, and
    those pixels are 2.2% of a normal frame -- far too few for the model to
    spend capacity on them. A focused scene puts them everywhere, so each step
    carries many times more gradient about the hard case.
    """
    g = torch.Generator(device=device).manual_seed(splitmix64(int(index)) & 0x7FFFFFFFFFFFFFFF)

    def r(*s):
        return torch.rand(*s, generator=g, device=device)

    b = 1
    ter_amp = torch.stack([0.5 + r(b) * 1.2, 0.25 + r(b) * 0.55, 0.08 + r(b) * 0.25], -1)
    ter_freq = torch.stack([0.15 + r(b) * 0.18, 0.4 + r(b) * 0.34, 1.0 + r(b) * 0.9], -1)
    sc = dict(ter_amp=ter_amp, ter_freq=ter_freq, ter_phase=r(b, 3) * 6.28)

    ang = r(b) * 2 * math.pi
    dist = (3.2 + r(b) * 1.8) if focus else (6.5 + r(b) * 4.0)   # focus: get close
    cam = torch.stack([dist * torch.cos(ang), torch.zeros(b, device=device),
                       dist * torch.sin(ang)], -1)
    cam[:, 1] = terrain_h(cam[:, 0], cam[:, 2], sc) + 1.0 + r(b) * 2.4
    look = torch.stack([(r(b) - 0.5) * 3.0, torch.zeros(b, device=device),
                        (r(b) - 0.5) * 3.0], -1)
    look[:, 1] = terrain_h(look[:, 0], look[:, 2], sc) + 0.25 + r(b) * 1.1

    def on_ground(n, spread, lift):
        x = (r(b, n) - 0.5) * spread
        z = (r(b, n) - 0.5) * spread
        return torch.stack([x, terrain_h(x, z, sc) + lift, z], -1)

    # ---- spheres: heavier on the materials v2 could not handle ----
    K = n_spheres
    sr = (0.7 + r(b, K) * 1.1) if focus else (0.3 + r(b, K) * 0.8)
    sph_c = on_ground(K, 3.4 if focus else 7.0, sr * 0.92)
    u = r(b, K)
    if focus:
        p_metal, p_glass, p_sss, p_emis = 0.50, 0.90, 0.95, 1.0
    else:
        p_metal = 0.28 * hard
        p_glass = p_metal + 0.30 * hard
        p_sss = p_glass + 0.16
        p_emis = p_sss + 0.06
    s_mat = torch.where(u < p_metal, torch.full_like(u, MAT_METAL),
             torch.where(u < p_glass, torch.full_like(u, MAT_GLASS),
              torch.where(u < p_sss, torch.full_like(u, MAT_SSS),
               torch.where(u < p_emis, torch.full_like(u, MAT_EMISSIVE),
                           torch.full_like(u, MAT_OPAQUE)))))
    s_alb = 0.15 + r(b, K, 3) * 0.8
    s_rough = torch.where(s_mat == MAT_METAL, 0.02 + r(b, K) * 0.3,
                          0.22 + r(b, K) * 0.66)
    # frosted glass: some of it scatters on transmission
    s_frost = torch.where(s_mat == MAT_GLASS, (r(b, K) < 0.4).float() * (0.05 + r(b, K) * 0.25),
                          torch.zeros_like(sr))

    M = n_boxes
    bs = 0.55 if focus else 0.0
    b_h = torch.stack([0.22 + bs + r(b, M) * 0.62, 0.22 + bs + r(b, M) * 1.05,
                       0.22 + bs + r(b, M) * 0.62], -1)
    b_c = on_ground(M, 4.0 if focus else 7.5, b_h[..., 1] * 0.95)
    ub = r(b, M)
    bm_t, bg_t = (0.55, 0.9) if focus else (0.22 * hard, 0.44 * hard)
    b_mat = torch.where(ub < bm_t, torch.full_like(ub, MAT_METAL),
             torch.where(ub < bg_t, torch.full_like(ub, MAT_GLASS),
                         torch.full_like(ub, MAT_OPAQUE)))

    P = n_plants
    p_h = 0.7 + r(b, P) * 1.6
    trunk_h = 0.15 + r(b, P) * 0.45
    p_base = on_ground(P, 8.0, trunk_h * 2)
    p_apex = p_base.clone()
    p_apex[..., 1] = p_base[..., 1] + p_h

    sa = r(b) * 2 * math.pi
    el = 0.28 + r(b) * 1.0
    sun = torch.stack([torch.cos(sa) * torch.cos(el), torch.sin(el),
                       torch.sin(sa) * torch.cos(el)], -1)
    warm = r(b)
    sun_col = torch.stack([1.0 + warm * 0.4, 0.94 + warm * 0.12,
                           0.82 + (1 - warm) * 0.3], -1) * (2.4 + r(b, 1) * 2.4)
    sky_h = torch.stack([0.32 + r(b) * 0.22, 0.44 + r(b) * 0.28,
                         0.62 + r(b) * 0.38], -1) * (0.55 + r(b, 1) * 0.85)

    sc.update(
        cam=cam, look=look, fov=(50.0 + r(b) * 22.0) * math.pi / 180.0,
        sph_c=sph_c, sph_r=sr, sph_alb=s_alb, sph_mat=s_mat, sph_rough=s_rough,
        sph_frost=s_frost,
        box_c=b_c, box_h=b_h, box_yaw=r(b, M) * math.pi,
        box_alb=0.15 + r(b, M, 3) * 0.8, box_mat=b_mat,
        box_rough=torch.where(b_mat == MAT_METAL, 0.04 + r(b, M) * 0.3,
                              0.3 + r(b, M) * 0.6),
        cone_apex=p_apex, cone_h=p_h, cone_r=0.3 + r(b, P) * 0.6,
        cone_alb=torch.stack([0.06 + r(b, P) * 0.15, 0.28 + r(b, P) * 0.45,
                              0.05 + r(b, P) * 0.14], -1),
        cone_rough=0.55 + r(b, P) * 0.4,
        trunk_c=p_base, trunk_h=trunk_h,
        trunk_alb=torch.stack([0.16 + r(b, P) * 0.16, 0.1 + r(b, P) * 0.1,
                               0.05 + r(b, P) * 0.07], -1),
        # water is far more common in v3, and now absorbs along the path
        has_water=torch.ones(b) if focus else (r(b) < 0.75 * hard).float(),
        # focused scenes put the waterline high so it fills much of the frame
        water_y=-ter_amp.sum(-1) * ((0.0 + r(b) * 0.10) if focus else (0.02 + r(b) * 0.34)),
        water_col=torch.stack([0.02 + r(b) * 0.05, 0.1 + r(b) * 0.16,
                               0.14 + r(b) * 0.24], -1),
        water_absorb=0.25 + r(b) * 0.9,
        wave_amp=0.012 + r(b) * 0.035, wave_freq=1.6 + r(b) * 2.8,
        has_smoke=(r(b) < 0.55 * hard).float(),
        smoke_c=on_ground(1, 5.0, 0.8 + r(b, 1) * 1.5)[:, 0],
        smoke_r=1.2 + r(b) * 2.2, smoke_dens=0.35 + r(b) * 1.0,
        smoke_col=0.55 + r(b, 3) * 0.4,
        sun_dir=_norm(sun), sun_col=sun_col,
        sun_angle=(0.4 + r(b) * 4.2) * math.pi / 180.0,
        sky_h=sky_h, sky_l=sky_h * (0.3 + r(b, 1) * 0.4),
        cloud_cov=0.25 + r(b) * 0.5, cloud_ph=r(b) * 20.0,
        col_grass=torch.stack([0.06 + r(b) * 0.12, 0.18 + r(b) * 0.28,
                               0.04 + r(b) * 0.1], -1),
        col_rock=torch.stack([0.2 + r(b) * 0.24, 0.19 + r(b) * 0.22,
                              0.18 + r(b) * 0.2], -1),
        col_dirt=torch.stack([0.22 + r(b) * 0.2, 0.15 + r(b) * 0.14,
                              0.09 + r(b) * 0.1], -1),
        ior=1.33 + r(b) * 0.24,
        dispersion=0.012 + r(b) * 0.045,       # R/G/B index spread
        emissive=(1.5 + r(b, 1) * 4.0) * (0.6 + r(b, 3) * 0.7),
        caustic=0.6 + r(b) * 1.6,
        fog=0.004 + r(b) * 0.016,
        exposure=0.85 + r(b) * 0.7,
        index=torch.tensor([float(index % 2**31)]),
    )
    return sc

def make_scene_batch(indices, device="cpu", **kw):
    """Batch of scenes, one per index. Distinct indices -> distinct scenes."""
    scenes = [make_scene(int(i), device=device, **kw) for i in indices]
    return {k: torch.cat([s[k] for s in scenes], 0) for k in scenes[0]}

def eval_scenes(device="cpu", **kw):
    """Five held-out scenes, in a namespace the training stream cannot reach."""
    idx = [scene_index(EVAL_NAMESPACE, s) for s in EVAL_SEEDS]
    return make_scene_batch(idx, device=device, **kw)


def eval_scenes_focus(device="cpu", n_spheres=7, n_boxes=5, **kw):
    """Five held-out scenes dominated by metal, glass and water.

    Kept separate from the general set so a focused run can be checked for both
    things that matter: did the hard surfaces improve, and did everything else
    survive.
    """
    idx = [scene_index(EVAL_NAMESPACE, s) for s in FOCUS_EVAL_SEEDS]
    return make_scene_batch(idx, device=device, focus=True,
                            n_spheres=n_spheres, n_boxes=n_boxes, **kw)


# ---------------------------------------------------------------------------
# scene tokens: the environment itself, not a picture of it
# ---------------------------------------------------------------------------

TOKEN_DIM = 26
MAX_TOKENS = None      # derived from the scene


def scene_tokens(sc):
    """Describe every object in the scene as a token: [B, N, TOKEN_DIM].

    Every other input is camera-space and therefore blind to anything off
    screen -- which is why reflections never worked: 92% of reflection rays hit
    geometry the camera cannot see. These tokens carry each object's position,
    size, orientation, colour and material *whether or not it is visible*, plus
    the sun and the terrain, so the network can reason about the environment
    rather than about a picture of it.

    Positions are given both in world space and relative to the camera, because
    what a reflection shows depends on where the object sits relative to the
    viewer, and a network should not have to rediscover that subtraction.
    """
    B = sc["cam"].shape[0]
    dev = sc["cam"].device
    cam = sc["cam"]
    fwd = _norm(sc["look"] - cam)
    toks = []

    def add(kind, pos, size, yaw, col, rough, metal, trans, sss, emis):
        rel = pos - cam[:, None, :]
        dist = rel.norm(dim=-1, keepdim=True)
        onehot = torch.zeros(B, pos.shape[1], 5, device=dev)
        onehot[..., kind] = 1.0
        toks.append(torch.cat([
            onehot,                                     # 5  what it is
            pos * 0.15,                                 # 3  where it is
            rel * 0.15,                                 # 3  where it is from here
            dist * 0.08,                                # 1  how far
            (rel / (dist + 1e-6) * fwd[:, None, :]).sum(-1, keepdim=True),   # 1 in front?
            size * 0.6,                                 # 3  how big
            torch.sin(yaw), torch.cos(yaw),             # 2  orientation
            col,                                        # 3  colour
            rough, metal, trans, sss, emis,             # 5  material
        ], -1))

    K = sc["sph_c"].shape[1]
    one = lambda v: v[..., None]
    add(0, sc["sph_c"], sc["sph_r"][..., None].expand(B, K, 3), one(torch.zeros_like(sc["sph_r"])),
        sc["sph_alb"], one(sc["sph_rough"]), one((sc["sph_mat"] == MAT_METAL).float()),
        one((sc["sph_mat"] == MAT_GLASS).float()), one((sc["sph_mat"] == MAT_SSS).float()),
        one((sc["sph_mat"] == MAT_EMISSIVE).float()))

    M = sc["box_c"].shape[1]
    add(1, sc["box_c"], sc["box_h"], one(sc["box_yaw"]), sc["box_alb"],
        one(sc["box_rough"]), one((sc["box_mat"] == MAT_METAL).float()),
        one((sc["box_mat"] == MAT_GLASS).float()), one(torch.zeros_like(sc["box_yaw"])),
        one(torch.zeros_like(sc["box_yaw"])))

    P = sc["cone_apex"].shape[1]
    csize = torch.stack([sc["cone_r"], sc["cone_h"], sc["cone_r"]], -1)
    add(2, sc["cone_apex"], csize, one(torch.zeros_like(sc["cone_h"])), sc["cone_alb"],
        one(sc["cone_rough"]), one(torch.zeros_like(sc["cone_h"])),
        one(torch.zeros_like(sc["cone_h"])), one(torch.full_like(sc["cone_h"], 0.55)),
        one(torch.zeros_like(sc["cone_h"])))

    # one global token: sun, sky, terrain shape, water level, fog
    z = torch.zeros(B, 1, device=dev)
    g = torch.cat([
        torch.zeros(B, 1, 5, device=dev),
        sc["sun_dir"][:, None] * 0.5,
        (sc["sun_col"] / 6.0)[:, None],
        sc["sky_h"][:, None],
        sc["ter_amp"][:, None] * 0.5,
        sc["ter_freq"][:, None],
        torch.stack([sc["has_water"], sc["water_y"] * 0.3, sc["has_smoke"],
                     sc["fog"] * 30, sc["exposure"] * 0.5, sc["ior"] - 1.0,
                     sc["cloud_cov"]], -1)[:, None],
    ], -1)
    g = F.pad(g, (0, TOKEN_DIM - g.shape[-1]))
    g[..., 4] = 1.0                                     # mark it as the global token
    toks.append(g)
    return torch.cat(toks, 1)
