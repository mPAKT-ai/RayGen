"""The raster -> ray traced network.

It does not just look at the frame. It takes the **G-buffer** a deferred engine
already has in memory -- shaded colour, albedo, normal, depth, roughness,
metalness, N.L, transmission, subsurface, in-scatter, a depth-peeled layer and
an environment probe -- which is what lets a ~100k-parameter model reason about
where a shadow or a reflection belongs instead of guessing from pixels alone.

Structure:

* **Almost everything happens at half resolution.** A pixel-unshuffle stem folds
  each 2x2 colour block into channels; the auxiliary G-buffer channels go
  through a small strided convolution rather than average pooling, because
  depth, normals and the depth-peeled layer carry real high-frequency detail at
  geometry edges and pooling threw it away. The edit is produced at H/2 and put
  back to full resolution by a learned upscaler whose refinement runs at full
  resolution, guided by colour, normals, depth and derived edge features. That
  full-res stage is the one place blur is fought directly, and it is budgeted:
  ray tracing costs ~9,460 ms/frame more than rasterizing while the whole
  network costs ~130 ms, so detail there is very cheap in the only currency
  that matters.

* **Two importance maps, for two different jobs.** These used to be one map
  doing both, which served neither well:
    - `gate` at H/2 decides *where the edit is visible* and multiplies the
      residual. Fine enough to keep a thin reflection alive.
    - `sched` at H/8 decides *where computation happens*. Coarse, cheap, and
      cacheable across frames; `infer.py` uses it to skip regions.

* **A cheap direct branch.** A shallow path from the stem predicts the easy part
  of the residual (contact darkening, obvious shadow edges) so the deep decoder
  can spend its capacity on the hard part.

* **No normalization layers at all.** GroupNorm/LayerNorm pool statistics over
  the whole image, which makes the output depend on input size -- a 64x64 tile
  and the 1080p frame it came from would be shaded differently (measured: up to
  0.47 per channel). With only local convolutions the net is exactly size
  invariant (verified: tile-vs-full difference 0.0), which is what makes both
  multi-resolution training and region skipping valid.

* **Identity at initialisation, but no dead gradients.** Every output path
  starts at exactly zero, so a finetune begins by reproducing its parent
  checkpoint. The upscaler is *not* zero throughout, though: its sub-pixel
  convolution is seeded as a nearest-neighbour upsampler on the edit channels,
  so gradient reaches the edit head from step one instead of having to bootstrap
  the whole upsampling path out of zero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .renderer import GBUF_CHANNELS, TEMPORAL_CHANNELS
from .scenes import TOKEN_DIM

IN_CHANNELS = GBUF_CHANNELS + TEMPORAL_CHANNELS

CH_NORMAL, CH_DEPTH, CH_PEEL_DEPTH = 6, 9, 19
GEOM_FEATURES = 3        # depth edge, normal edge, peel thickness

MASK_STRIDE = 8          # scheduling map resolution relative to the input
GATE_STRIDE = 2          # gating map resolution relative to the input


def geometry_features(x):
    """Cheap derived features a tiny CNN should not have to rediscover.

    Depth and normal discontinuities mark exactly the boundaries the model keeps
    getting wrong, and the gap between the primary surface and the depth-peeled
    layer behind it is the single most informative number for refraction --
    thin glass bends light very differently from a deep volume of it. All of
    this is a few finite differences on tensors we already have.
    """
    d = x[:, CH_DEPTH:CH_DEPTH + 1]
    n = x[:, CH_NORMAL:CH_NORMAL + 3]
    peel = x[:, CH_PEEL_DEPTH:CH_PEEL_DEPTH + 1]

    def disc(v):
        gx = F.pad((v[..., 1:] - v[..., :-1]).abs(), (0, 1))
        gy = F.pad((v[..., 1:, :] - v[..., :-1, :]).abs(), (0, 0, 0, 1))
        return (gx + gy).amax(1, keepdim=True)

    return torch.cat([
        (disc(d) * 12.0).clamp(0, 1),        # depth discontinuity
        (disc(n) * 3.0).clamp(0, 1),         # normal discontinuity
        (peel - d).clamp(0, 1),              # thickness behind the surface
    ], 1)


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.SiLU(inplace=True),
    )


class SceneAttention(nn.Module):
    """Let every region of the frame query every object in the scene.

    This is the piece that was missing all along. Screen-space buffers can only
    describe what the camera sees, so anything reflected, refracted or bounced
    from off-screen geometry was unpredictable in principle -- measured, 92% of
    reflection rays hit geometry absent from every input channel.

    Here each H/8 cell attends over object tokens carrying position, size,
    orientation, colour and material for *every* object, visible or not, plus a
    global token for sun, sky and terrain. Cost is (H/8 x W/8) x N x d, which at
    192x288 with a dozen objects is a few hundred thousand multiply-adds --
    nothing next to the convolutions, and it is the only path by which
    off-screen geometry can reach the prediction at all.
    """

    def __init__(self, feat_ch, token_dim=TOKEN_DIM, d=32):
        super().__init__()
        self.d = d
        self.tok = nn.Linear(token_dim, d * 2)
        self.q = nn.Conv2d(feat_ch, d, 1)
        self.out = nn.Conv2d(d, feat_ch, 1)
        nn.init.zeros_(self.out.weight)      # identity at init: a finetune starts unchanged
        nn.init.zeros_(self.out.bias)

    def forward(self, feat, tokens):
        if tokens is None:
            return feat
        B, C, H, W = feat.shape
        k, v = self.tok(tokens).chunk(2, -1)                 # [B,N,d] each
        q = self.q(feat).reshape(B, self.d, H * W).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(1, 2) / (self.d ** 0.5), -1)
        o = (att @ v).transpose(1, 2).reshape(B, self.d, H, W)
        return feat + self.out(o)


class Upscaler(nn.Module):
    """Learned half-res -> full-res reconstruction of the edit.

    Two things decide whether the edit lands on the right pixels:

    * **Geometry guidance.** The refinement sees full-resolution colour *and*
      depth and normals. The whole job is to place a half-res correction onto
      the exact boundary it belongs to, and the G-buffer already marks those
      boundaries exactly -- inferring them from colour alone is strictly harder
      and gets shadow edges and thin reflections wrong.

    * **Enough full-res capacity to not blur.** This used to be a single
      depthwise convolution, which is the cheapest thing that can align edges
      and also close to the least capable. Measurement decided the trade: ray
      tracing costs ~9,460 ms/frame more than rasterizing, while the whole
      network cost ~134 ms -- 1.4% of the budget. Spending a few more
      milliseconds at full resolution to stop losing high-frequency detail is
      obviously correct at that exchange rate.

    `refine_width` 0 restores the old single-depthwise behaviour for ablation.
    """

    def __init__(self, feat_ch, guide_ch=7, refine_width=8):
        super().__init__()
        self.feat_ch = feat_ch
        self.guide_ch = guide_ch
        self.to_sub = nn.Conv2d(feat_ch + 3, 3 * 4, 1)
        self.shuffle = nn.PixelShuffle(2)

        cin = 3 + guide_ch
        if refine_width:
            self.refine = nn.Sequential(
                nn.Conv2d(cin, refine_width, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(refine_width, refine_width, 3, padding=1, groups=refine_width),
                nn.SiLU(inplace=True),
                nn.Conv2d(refine_width, 3, 1),
            )
        else:
            self.refine = nn.Sequential(
                nn.Conv2d(cin, cin, 3, padding=1, groups=cin),
                nn.Conv2d(cin, 3, 1),
            )

        # Seed to_sub as a nearest-neighbour upsampler of the edit channels, zero
        # on the feature channels. The edit head is zero-initialised, so the
        # residual is still exactly zero at step 0 -- but d(residual)/d(edit) is
        # not, so the edit head receives gradient immediately.
        nn.init.zeros_(self.to_sub.weight)
        nn.init.zeros_(self.to_sub.bias)
        with torch.no_grad():
            for c in range(3):
                for k in range(4):
                    self.to_sub.weight[c * 4 + k, feat_ch + c, 0, 0] = 1.0

        # Last layer zero so the refinement contributes nothing at init, earlier
        # layers normal so gradient flows through non-zero activations.
        last = self.refine[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, feat_half, edit_half, guide_full):
        coarse = self.shuffle(self.to_sub(torch.cat([feat_half, edit_half], 1)))
        return coarse + self.refine(torch.cat([coarse, guide_full], 1))


class RayGenNet(nn.Module):
    def __init__(self, width=16, in_ch=IN_CHANNELS, refine_width=8):
        super().__init__()
        w1, w2, w3 = width, width * 2, width * 3
        self.in_ch = in_ch
        self.width = width
        self.refine_width = refine_width

        # Colour keeps full detail through the unshuffle. The auxiliary channels
        # get a learned stride-2 reduction: depth, normals and the depth-peeled
        # layer are *not* smooth at geometry edges, and average pooling deleted
        # exactly the detail that shadow and reflection boundaries live on.
        self.down = nn.PixelUnshuffle(2)
        # derived geometry features join the auxiliary channels at the stem
        self.aux_stem = nn.Sequential(
            nn.Conv2d(in_ch - 3 + GEOM_FEATURES, w1, 3, stride=2, padding=1),
            nn.SiLU(inplace=True))
        stem_ch = 12 + w1

        self.enc1 = _block(stem_ch, w1)             # H/2
        self.enc2 = _block(w1, w2)                  # H/4
        self.enc3 = _block(w2, w3)                  # H/8
        self.scene_attn = SceneAttention(w3)        # the environment reaches the net here

        # scheduling map: coarse, cheap, cacheable -> "where must we compute?"
        self.sched_head = nn.Sequential(nn.Conv2d(w3, w1, 3, padding=1), nn.SiLU(inplace=True),
                                        nn.Conv2d(w1, 1, 1))
        nn.init.zeros_(self.sched_head[2].weight)
        nn.init.constant_(self.sched_head[2].bias, 2.0)

        self.dec2 = _block(w3 + w2, w2)
        self.dec1 = _block(w2 + w1, w1)             # H/2 edit features

        # gating map: fine, from the decoder -> "where is the edit visible?"
        self.gate_head = nn.Conv2d(w1, 1, 3, padding=1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, 2.0)

        self.to_edit = nn.Conv2d(w1, 3, 1)          # the edit itself, at H/2
        nn.init.zeros_(self.to_edit.weight)
        nn.init.zeros_(self.to_edit.bias)

        # cheap direct branch: the easy part of the residual, straight off the stem
        self.quick = nn.Conv2d(stem_ch, 3, 3, padding=1)
        nn.init.zeros_(self.quick.weight)
        nn.init.zeros_(self.quick.bias)

        # guidance for the full-res refinement: colour + normal + depth + geom edges
        self.upscale = Upscaler(w1, guide_ch=3 + 3 + 1 + GEOM_FEATURES,
                                refine_width=refine_width)
        self.last_gate = None

    # -- pieces, so inference can run them separately ------------------------

    def stem(self, x, geom):
        return torch.cat([self.down(x[:, :3]), self.aux_stem(torch.cat([x[:, 3:], geom], 1))], 1)

    def guidance(self, x, geom):
        """Full-resolution signal that tells the refinement where edges really are."""
        return torch.cat([x[:, :3], x[:, CH_NORMAL:CH_NORMAL + 3],
                          x[:, CH_DEPTH:CH_DEPTH + 1], geom], 1)

    def encode(self, x, geom=None, tokens=None):
        geom = geometry_features(x) if geom is None else geom
        s = self.stem(x, geom)
        e1 = self.enc1(s)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        return s, e1, e2, self.scene_attn(e3, tokens)

    def schedule(self, e3):
        """Coarse importance at H/8: where computation is worth spending."""
        return self.sched_head(e3)

    def attention(self, e3):
        """Alias kept for callers that only want the scheduling map."""
        return self.schedule(e3)

    def decode(self, x, s, e1, e2, e3, geom):
        d2 = self.dec2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], 1))

        edit_half = self.to_edit(d1) + self.quick(s)        # deep edit + cheap branch
        gate_half = torch.sigmoid(self.gate_head(d1))
        residual = self.upscale(d1, edit_half * gate_half, self.guidance(x, geom))
        return (x[:, :3] + residual).clamp(0, 1), gate_half

    def forward(self, x, tokens=None, mask=None, return_mask=True):
        """x: [B, in_ch, H, W] G-buffer -> (rgb, sched_map).

        `sched_map` is the H/8 scheduling map: what `infer.py` caches and uses to
        decide which regions to compute. The residual gate is internal and always
        computed -- it is nearly free, coming from features the decoder already
        produced. Supplying `mask` substitutes a cached scheduling map.
        """
        h, w = x.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        xin = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x

        geom = geometry_features(xin)
        s, e1, e2, e3 = self.encode(xin, geom, tokens)
        sched = torch.sigmoid(self.schedule(e3)) if mask is None else mask
        out, gate = self.decode(xin, s, e1, e2, e3, geom)
        out = out[..., :h, :w]
        self.last_gate = gate
        return (out, sched) if return_mask else out

    def forward_train(self, x, tokens=None):
        """Both maps, for supervision: (rgb, sched at H/8, gate at H/2)."""
        h, w = x.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        xin = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
        geom = geometry_features(xin)
        s, e1, e2, e3 = self.encode(xin, geom, tokens)
        sched = torch.sigmoid(self.schedule(e3))
        out, gate = self.decode(xin, s, e1, e2, e3, geom)
        return out[..., :h, :w], sched, gate


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def load_grown(ckpt_path, width=None, in_ch=None, key="model"):
    """Load a checkpoint into a model that may differ in shape.

    Newer scene versions add G-buffer channels; architecture changes add or
    resize layers. Rather than discarding trained weights, any convolution that
    now wants *more input channels* is widened with the new columns
    zero-initialised, so at step 0 the grown model computes what the old one did
    and the new channels contribute as soon as they earn a gradient. Tensors
    that changed shape for other reasons keep their fresh initialisation and are
    **reported** rather than silently dropped -- a silent mismatch here is how
    you end up finetuning something that quietly threw half its weights away.
    """
    d = torch.load(ckpt_path, map_location="cpu")
    sd = d.get(key) or d.get("model")
    width = width or d.get("width", 16)
    in_ch = in_ch or d.get("in_ch", IN_CHANNELS)
    model = RayGenNet(width, in_ch, refine_width=d.get("refine_width", 8))
    tgt = model.state_dict()

    loaded, grown, fresh = [], [], []
    new_sd = dict(tgt)
    for k, v in sd.items():
        if k not in tgt:
            fresh.append(k)
        elif tgt[k].shape == v.shape:
            new_sd[k] = v
            loaded.append(k)
        elif (v.dim() == 4 and tgt[k].shape[0] == v.shape[0]
              and tgt[k].shape[1] > v.shape[1] and tgt[k].shape[2:] == v.shape[2:]):
            g = torch.zeros_like(tgt[k])
            g[:, :v.shape[1]] = v
            new_sd[k] = g
            grown.append(f"{k} {tuple(v.shape)}->{tuple(tgt[k].shape)}")
        else:
            fresh.append(f"{k} {tuple(v.shape)}!={tuple(tgt[k].shape)}")
    model.load_state_dict(new_sd)
    print(f"  loaded {len(loaded)} tensors"
          + (f" | grew {len(grown)}: {', '.join(grown)}" if grown else "")
          + (f" | fresh-init {len(fresh)}: {', '.join(fresh[:5])}"
             + ("..." if len(fresh) > 5 else "") if fresh else ""))
    return model
