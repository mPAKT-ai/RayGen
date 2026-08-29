# RayGen

A ~115k-parameter network that takes a frame rendered the way a game renders it
— shadow maps, SSAO, screen-space reflections — and predicts what the same frame
would look like ray traced: soft shadows, real ambient occlusion, bounce light,
refraction through glass and water, and volumetric smoke.

* **No diffusion.** One feed-forward pass.
* **No GPU required.** Everything is tuned for CPU.
* **No dataset.** Both sides of every training pair are rendered on the fly.
* **Any resolution.** Fully convolutional and exactly size-invariant.
* **Two modes.** Render a single image, or predict how the next frame changes as
  the scene moves.

```bash
pip install torch numpy pillow
python example.py
```

## How it works

`raygen/renderer.py` renders every scene **twice from the same camera, geometry,
materials and lighting**. Only the light transport differs, so the network has
to learn lighting rather than geometry.

| | input: `render_game` | target: `render_rt` |
|---|---|---|
| shadows | 64² shadow map, 2×2 PCF — hard, aliased, with acne | sampled sun disk → true penumbrae |
| occlusion | SSAO, screen-space, haloes | multi-bounce GI: real AO and colour bleeding |
| reflections | SSR — only what is already on screen | real rays, whole scene, roughness-sampled |
| water | sky-only reflection + depth tint | reflection **and** refraction, Fresnel blended |
| glass | alpha blend against the sky | true refraction, dispersion, frosted transmission |
| smoke | analytic absorption, no self-shadow | marched, sun-shadowed, god rays |
| caustics | none | focused through glass, `f = nR/2(n−1)` |

Scenes are ray-marched terrain with slope- and altitude-based texturing, water,
glass, metal, subsurface materials, cone foliage, emissive objects, volumetric
smoke and a procedural fBm sky.

## What the model sees

Not just the frame. It receives **28 channels**:

* the **G-buffer** any deferred renderer already has — shaded colour, albedo,
  normal, depth, roughness, metalness, N·L, transmission, subsurface, in-scatter
* a **depth-peeled layer** — the colour and depth of what lies *behind* each
  transparent pixel, which is what makes refraction learnable
* an **environment probe** sampled along the reflection direction
* **animation channels** — the previous frame reprojected by motion vectors, a
  disocclusion mask, and a mode flag

…plus the **scene itself**, as one token per object carrying position, size,
orientation, colour and material *whether or not the camera can see it*, with a
global token for sun, sky, terrain, water and fog. A cross-attention layer lets
every region of the frame query every object. Screen-space buffers cannot
describe what a reflection shows, because that geometry is usually off-camera —
these tokens are the only path by which it reaches the prediction.

## Architecture

```
G-buffer + scene tokens
      │
   stem: pixel-unshuffle colour, learned stride-2 reduction of the rest
      │
   encoder  H/2 → H/4 → H/8 ──── cross-attention over scene tokens
      │                     └──── scheduling map (H/8): where to compute
   decoder  → edit at H/2  ┌───── gating map (H/2): where the edit is visible
      │                    │
   learned upscaler ───────┘  → full-resolution residual
```

Notable choices, each of which was measured rather than assumed:

* **No normalization layers.** GroupNorm pools statistics over the whole image,
  which makes output depend on input size — a 64×64 tile and the 1080p frame it
  came from were shaded differently by up to 0.47 per channel. Without it the
  network is *exactly* size-invariant: tile-vs-full difference is 0.0.
* **Two importance maps, not one.** `sched` (H/8) decides where computation
  happens and is cacheable across frames; `gate` (H/2) decides where the edit is
  visible. One map doing both jobs served neither.
* **Identity at initialisation.** Every output path starts at exactly zero, so a
  finetune begins by reproducing its parent — but the upscaler is seeded as a
  nearest-neighbour upsampler on the edit channels, so gradients reach the edit
  head from step one instead of bootstrapping out of zero.
* **Half-resolution edits.** No convolution runs at full resolution except one
  guided refinement, which sees colour, normals, depth and derived edge features
  so the correction lands on real boundaries.

## Every scene is unique

A scene is a pure function of a 64-bit index:

```
index = (run_id << 32) | counter
seed  = splitmix64(index)          # bijection: distinct index, distinct seed
```

`run_id` is fresh per run and every previous one is stored in the checkpoint, so
a resumed run draws from a namespace disjoint from all earlier runs; `counter` is
partitioned across workers by residue. Verified: 200 consecutive indices give 200
distinct scene fingerprints, the same index always reproduces the same scene, and
a second run shares nothing with the first.

## Results

On five held-out scenes the model never trained on, measured against a converged
ray traced reference:

| region | share of pixels | unedited | model | improvement |
|---|---|---|---|---|
| overall | 100% | 0.0469 | 0.0154 | **3.05×** |
| transmissive (glass, water) | 9.6% | 0.1375 | 0.0457 | 3.01× |
| flat lit | 20.3% | 0.0646 | 0.0208 | 3.11× |
| geometry edges | 67.1% | 0.0253 | 0.0090 | 2.80× |
| shadowed | 5.6% | 0.0893 | 0.0332 | 2.69× |
| reflective (metal) | 2.2% | 0.0842 | 0.0644 | 1.31× |

### Known limitation: reflections

Reflections are the weakest region by a wide margin, and this is a structural
limit rather than a tuning problem. Shading a mirror pixel requires following a
ray into the scene and finding what it hits — a search over geometry. A
convolutional network performs local filtering, and cross-attention is a coarse
lookup; neither computes ray-scene intersection. Everything else the model does
well — shadows, GI, contact darkening, refraction through a known depth-peeled
layer — is a filtering problem over data already aligned to the pixel.

The evidence: training on scenes where metal covers **52%** of pixels instead of
2.2% — roughly 24× more reflective gradient per step, for 20,000 steps — improved
metal by only 17% (1.18× → 1.38×). Reflections are not data-limited.

The practical fix is a hybrid: trace **one** reflection ray per pixel and feed
the result in as a channel. That is a small fraction of full ray tracing (no GI,
no multi-sample soft shadows, no refraction, no volumetrics) and it supplies the
one computation the architecture cannot express.

## Files

| file | |
|---|---|
| `raygen/scenes.py` | scene generation, terrain, sky, unique indexing, scene tokens |
| `raygen/renderer.py` | both render passes, G-buffer, environment probe, motion vectors |
| `raygen/model.py` | the network: attention, generator and upscaler as one module |
| `raygen/data.py` | infinite non-repeating training stream |
| `example.py` | end-to-end usage: data, training, image mode, animation mode |
