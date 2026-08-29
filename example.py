"""RayGen by example: generate data, train, and render in both modes.

Run it directly for a short end-to-end demonstration::

    python example.py

Everything here is CPU-friendly and small enough to finish in a couple of
minutes. It is meant to be read as much as run -- each section is the shortest
correct version of something you would do for real.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

from raygen.data import make_loader
from raygen.model import IN_CHANNELS, RayGenNet, count_params
from raygen.renderer import (add_temporal, render_eval_pair, render_pair,
                             render_sequence)
from raygen.scenes import eval_scenes, make_scene_batch, new_run_id, scene_index, scene_tokens


# ---------------------------------------------------------------------------
# 1. Data: a scene is a pure function of a unique index
# ---------------------------------------------------------------------------

def example_data():
    """Render one training pair.

    `x` is the 23-channel G-buffer a deferred engine already has (shaded colour,
    albedo, normal, depth, roughness, metalness, N.L, transmission, subsurface,
    in-scatter, a depth-peeled layer behind transparent pixels, and an
    environment probe). `y` is the ray traced frame we want to predict.

    Scenes are addressed by a 64-bit index, so the same index always reproduces
    the same scene and two different indices never collide -- which is what lets
    a training run guarantee it never repeats an example, even across restarts.
    """
    run_id = new_run_id()
    indices = [scene_index(run_id, i) for i in range(4)]

    x, y = render_pair(indices=indices, res=(64, 96))
    print(f"  G-buffer {tuple(x.shape)}   ray traced target {tuple(y.shape)}")

    # The scene description itself: one token per object, whether or not the
    # camera can see it, plus a global token for sun/sky/terrain/water/fog.
    tokens = scene_tokens(make_scene_batch(indices))
    print(f"  scene tokens {tuple(tokens.shape)}  (batch, objects, features)")
    return x, y, tokens


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------

def example_training(steps=20):
    """A minimal training loop.

    The real trainer adds an EMA, a resolution curriculum, material-weighted and
    Laplacian losses, and crash-safe checkpointing -- but this is the core of it,
    and it is a plain supervised regression.
    """
    model = RayGenNet(width=16)
    print(f"  model: {count_params(model):,} parameters, {IN_CHANNELS} input channels")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader, _ = make_loader(new_run_id(), batch=4, workers=0, anim_frac=0.35)
    it = iter(loader)

    for step in range(1, steps + 1):
        x, y, tokens = next(it)
        pred, sched, gate = model.forward_train(x, tokens)

        # image term, plus a gradient term because plain L1 is happy to answer
        # with a blurred conditional mean
        loss = F.l1_loss(pred, y) + 0.5 * (
            F.l1_loss(pred[..., 1:] - pred[..., :-1], y[..., 1:] - y[..., :-1])
            + F.l1_loss(pred[..., 1:, :] - pred[..., :-1, :], y[..., 1:, :] - y[..., :-1, :]))

        # the scheduling map learns where skipping would cost the most: if a
        # region is skipped its output is the input, so that error is exactly
        # |target - input|
        want = (y - x[:, :3]).abs().mean(1, keepdim=True)
        want = F.interpolate((F.max_pool2d(want, 8, 8, ceil_mode=True) / 0.12).clamp(0, 1),
                             size=sched.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss + 0.15 * F.binary_cross_entropy(sched.clamp(1e-4, 1 - 1e-4), want)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0 or step == 1:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    torch.save({"model": model.state_dict(), "width": 16, "in_ch": IN_CHANNELS},
               "example_model.pt")
    print("  saved example_model.pt")
    return model


# ---------------------------------------------------------------------------
# 3. Inference: image mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def example_image_mode(model):
    """One frame in, one frame out.

    `add_temporal(x)` with no previous frame fills the animation channels with
    zeros and sets the mode flag to 0, so the same weights serve both modes.
    """
    sc = eval_scenes()
    x, target = render_eval_pair(res=(96, 144))
    x = add_temporal(x)
    tokens = scene_tokens(sc)

    pred, sched = model(x, tokens)
    before = (x[:, :3] - target).abs().mean()
    after = (pred - target).abs().mean()
    print(f"  L1 to ray traced: {before:.4f} unedited -> {after:.4f} predicted")
    print(f"  scheduling map {tuple(sched.shape)} (H/8) -- cache it to skip regions")
    return pred


# ---------------------------------------------------------------------------
# 4. Inference: animation mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def example_animation_mode(model):
    """Render one frame normally, then predict how the next one changes.

    This is the shape a game integration takes: you already have a G-buffer and
    motion vectors for the new frame, plus the previous frame you rendered. The
    model receives the previous frame reprojected into the current one, a
    validity mask marking disocclusions, and a flag saying it is in animation
    mode.
    """
    run_id = new_run_id()
    seq = render_sequence(indices=[scene_index(run_id, i) for i in range(3)], res=(64, 96))

    # frame A: ordinary image mode
    a_pred, _ = model(add_temporal(seq["a_x"]), scene_tokens(seq["sc_a"]))

    # frame B: conditioned on what we produced for frame A
    b_in = add_temporal(seq["b_x"], a_pred, seq["flow"], seq["valid"])
    b_pred, _ = model(b_in, scene_tokens(seq["sc_b"]))

    err = (b_pred - seq["b_y"]).abs().mean()
    reuse = (a_pred - seq["b_y"]).abs().mean()
    print(f"  frame B: predicted {err:.4f} vs {reuse:.4f} if you simply reused frame A")
    print(f"  reprojection valid on {float(seq['valid'].mean())*100:.0f}% of pixels "
          f"(the rest is disocclusion the model must fill in)")
    return b_pred


# ---------------------------------------------------------------------------
# 5. Loading a checkpoint
# ---------------------------------------------------------------------------

def example_load(path="example_model.pt"):
    d = torch.load(path, map_location="cpu")
    model = RayGenNet(d.get("width", 16), d.get("in_ch", IN_CHANNELS))
    model.load_state_dict(d["ema"] if "ema" in d else d["model"])
    model.eval()
    print(f"  loaded {path}: width {d.get('width', 16)}, {count_params(model):,} params")
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    print("\n[1] generating a training pair")
    example_data()

    print("\n[2] training for a few steps")
    model = example_training(steps=20)

    print("\n[3] inference, image mode")
    example_image_mode(model)

    print("\n[4] inference, animation mode")
    example_animation_mode(model)

    print("\n[5] reloading the checkpoint")
    example_load()
    print("\ndone.")
