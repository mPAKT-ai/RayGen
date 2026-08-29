"""Training stream: infinite, on the fly, and every scene provably new.

Nothing is written to disk: worker processes render fresh pairs while the main
process trains.

An earlier version seeded each worker once and drew from that stream forever,
which hides a real flaw: on `--resume` the workers reseed from the same base and
regenerate **the same scenes**, so any power cut would silently start feeding
the model repeats. Here every scene is a pure function of a globally unique 64-bit index:

    index = (run_id << 32) | counter

* `run_id` is fresh per run, and the checkpoint carries every run_id used
  before, so a resumed run gets a namespace disjoint from all earlier ones.
* `counter` is partitioned across workers by residue -- worker `w` of `W` emits
  `w, w+W, w+2W, ...` -- so two workers cannot collide inside a run.
* index -> seed goes through splitmix64, a bijection on 64 bits.

Verified empirically: 200 consecutive indices produce 200 distinct scene
fingerprints, the same index always reproduces the same scene, and scenes from
a second run_id never intersect the first.
"""

import os

import torch
from torch.utils.data import IterableDataset, DataLoader

from .renderer import render_pair, render_sequence, add_temporal
from .scenes import scene_tokens
from .scenes import scene_index

# Cost is per pixel and the batch is scaled by area, so these stay comparable.
SHAPE_STAGES = [
    [(56, 56), (56, 72), (64, 64)],
    [(64, 64), (64, 96), (72, 96)],
    [(80, 80), (88, 128), (96, 144)],
]


class SceneStreamV3(IterableDataset):
    def __init__(self, run_id, batch=8, start_counter=0, workers=1, anim_frac=0.35, focus_frac=0.0,
                 sun_samples=3, gi_samples=1, smoke_steps=8, terrain_steps=32,
                 ao_taps=6, ssr_steps=10, dispersion=True, hard=1.0,
                 stage_holder=None):
        super().__init__()
        self.run_id = run_id
        self.batch = batch
        self.start = start_counter
        self.workers = max(1, workers)
        self.hard = hard
        self.anim_frac = anim_frac
        self.focus_frac = focus_frac
        self.render_kw = dict(sun_samples=sun_samples, gi_samples=gi_samples,
                              smoke_steps=smoke_steps, terrain_steps=terrain_steps,
                              ao_taps=ao_taps, ssr_steps=ssr_steps,
                              dispersion=dispersion)
        self.stage_holder = stage_holder

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        torch.set_num_threads(1)
        # this worker owns counters start + wid, start + wid + W, ...
        counter = self.start + wid
        rng = torch.Generator().manual_seed(self.run_id * 131 + wid)
        while True:
            stage = int(self.stage_holder.item()) if self.stage_holder is not None else 1
            shapes = SHAPE_STAGES[min(stage, len(SHAPE_STAGES) - 1)]
            h, w = shapes[int(torch.randint(len(shapes), (1,), generator=rng))]
            b = max(2, int(self.batch * (64 * 64) / (h * w)))
            idx = [scene_index(self.run_id, counter + i * self.workers) for i in range(b)]
            counter += b * self.workers

            # A focused batch is built almost entirely from the surfaces the
            # model is worst at (metal 52% of pixels vs 2% normally). Keeping a
            # share of ordinary scenes is deliberate: training only on the hard
            # case is how a model forgets the other 90% of the frame.
            focused = float(torch.rand(1, generator=rng)) < self.focus_frac
            skw = {}
            if focused:
                skw = dict(skw, focus=True, n_spheres=7, n_boxes=5)

            # Animation mode costs two frames, so it is a fraction of batches
            # rather than all of them. The model sees both jobs and the mode flag
            # tells it which one it is doing.
            if float(torch.rand(1, generator=rng)) < self.anim_frac:
                d = render_sequence(indices=idx, res=(h, w), hard=self.hard,
                                    **skw, **self.render_kw)
                x = add_temporal(d["b_x"], d["a_y"], d["flow"], d["valid"])
                yield x, d["b_y"], scene_tokens(d["sc_b"])
            else:
                gx, gy = render_pair(indices=idx, res=(h, w), hard=self.hard,
                                     **skw, **self.render_kw)
                from .scenes import make_scene_batch
                yield (add_temporal(gx), gy,
                       scene_tokens(make_scene_batch(idx, hard=self.hard, **skw)))


def make_loader(run_id, batch=8, workers=None, prefetch=3, start_counter=0,
                stage_holder=None, anim_frac=0.35, focus_frac=0.0, **kw):
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)
    ds = SceneStreamV3(run_id, batch=batch, start_counter=start_counter,
                       workers=workers, stage_holder=stage_holder, anim_frac=anim_frac,
                       focus_frac=focus_frac, **kw)
    return DataLoader(ds, batch_size=None, num_workers=workers,
                      persistent_workers=workers > 0,
                      prefetch_factor=prefetch if workers > 0 else None), workers
