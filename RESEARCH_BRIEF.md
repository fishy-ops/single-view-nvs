# How far can one photo move? Predicting where single-image view synthesis breaks

Brief for Dr. Kumar, from an overnight build (15 Sep 2026). Interactive report:
shared on request (private link).

## Question
Apple's SHARP (arXiv 2512.10685) turns one photo into ~1.2M 3D Gaussians in under a second, and
spatial-photo UIs let people move the camera around the result. Every such system has to decide
how far to let the camera go. Today that limit is fixed. **Can we predict, per photo and per
requested camera pose, how bad the view will look, using only what's available at test time?**

## Setup
- **Data:** LLFF (8 forward-facing scenes) + Mip-NeRF 360 (7 orbit scenes), all public. 120 source
  photos (8 per scene) and 3,607 (source, target) pairs where the target is a *real photo* from
  another calibrated camera. Poses come from COLMAP. SHARP is scale-aligned per photo with one
  scalar fit to COLMAP sparse points.
- **Metrics:** LPIPS (headline), PSNR, SSIM; each split into pixels the source photo could see vs
  pixels it could not ("unseen", from rendered depth + a z-test).
- **Baselines:** pixel-splat warp at SHARP's depth, same warp at Depth Pro depth, and reusing the
  source photo. Every method gets the same Telea hole fill.
- **Hardware:** M4 Pro 48GB (SHARP inference 5.7s/photo on MPS, UNet training), RTX 2070 Super
  8GB (gsplat rendering + metrics).

## Findings (95% CIs from a 1000x scene-level bootstrap where given)
1. **SHARP's edge over plain depth warping is perceptual.** With identical hole filling: +0.49dB
   PSNR, but LPIPS better in 100% of 1,350 pairs by 0.075 [0.066, 0.084]. The gap grows with
   camera distance and is largest in unseen regions (LPIPS 0.50 vs 0.64). In seen regions SHARP
   and its own-depth warp are nearly equal on PSNR (16.62 vs 16.49): on what the photo shows, SHARP
   is roughly its depth map.
2. **SHARP's own geometry predicts how bad a view will be.** Leave-one-scene-out ridge on
   test-time features (camera move + the share of the target view SHARP's geometry says is unseen):
   R2 0.648 vs 0.284 for camera distance alone, gain **+0.36 [0.20, 0.55]**. Rank correlation among
   views at the same distance: 0.79 vs 0.40, gain +0.39 [0.16, 0.58].
3. **A per-photo safe radius.** The offset where each photo's error curve crosses median LPIPS,
   predicted vs measured over 119 photos: Spearman **0.82 [0.71, 0.90]**; typical error x1.5 vs x2.6
   for a single fixed radius. As a policy, allowing half of all requested moves lets through 21% bad
   views vs 43% for a distance limit (reduction 23 points [7, 33]).
4. **It transfers.** Train on LLFF, test on 360: camera-distance rule R2 -1.01, unseen-area model
   0.70 (reverse direction: 0.09 vs 0.52). With masks from Depth Pro's *own* depth, the same model
   predicts the Depth Pro warp (R2 0.28 to 0.55; safe-radius rho 0.79), so any renderer with
   explicit geometry can estimate its own range. Within one photo, the prediction ranks directions
   better than distance on 360 orbits (rho 0.87 vs 0.68) but not on LLFF grids (0.91 vs 0.93), where
   views differ mostly in distance.
5. **The signal is disocclusion, not frame overlap.** Splitting the unseen share into
   out-of-frame vs occluded-inside-the-frame: camera move + out-of-frame share *lowers* held-out
   R2 (0.28 to 0.23; LLFF 0.15 to -0.12), while camera move + occluded share reaches 0.63 (both
   together 0.68). A frustum-overlap heuristic doesn't predict quality; what's hidden behind
   foreground surfaces does.
6. **Pixel-level confidence maps.** A 2M-param UNet on render + opacity + depth + unseen mask
   predicts per-pixel LPIPS on held-out scenes: pixel Spearman 0.63 vs 0.41 for the unseen mask
   alone, AUROC for the worst 20% of pixels 0.83 vs 0.79. Foliage scenes stay hard (0.41).
7. **Negative result: warn with the map, don't repair with it.** LaMa inpainting at a matched
   pixel budget on 600 views: filling the unseen mask improves LPIPS by 0.025 (65% of views);
   filling the UNet's highest-error pixels improves it by 0.006 and loses to the unseen mask 80% of
   the time (it still beats random regions 88%). Adding the top 5% confidence pixels to the mask, or
   dilating it, doesn't help. Flagged pixels in seen regions are misaligned but mostly right, and
   inpainting destroys them.

## Prior art and positioning
- **US 12,051,150** (priority BR 2021-12-21) scores whether an input image suits a single-image 3D
  effect by counting non-occluded pixels in generated target views. That is our unseen-area cue as a
  binary suitability score, so the cue itself is not new.
- arXiv 2603.22786 (2026): predictive photometric uncertainty for 3DGS optimized from many views
  (per-scene, multi-view, not single-image feed-forward).
- arXiv 2606.07514 UniSHARP (2026): SHARP for any camera model; no quality prediction.
- **Defensible contributions:** (a) real-GT benchmark of single-image 3DGS vs camera move with
  seen/unseen decomposition; (b) continuous per-photo safe radius and a movement-limit policy,
  validated on held-out scenes with CIs; (c) cross-dataset and cross-renderer transfer; (d)
  pixel-level confidence for feed-forward single-image models, with the negative
  repair-guidance result; (e) evidence that occlusion, not frame overlap, carries the signal, which
  separates this from a plain non-occluded pixel count.

## Threats to validity
- LLFF's nearest neighbours are 2-5 degrees apart, so the sub-degree regime of phone parallax UIs
  isn't measured. 360 views mix rotation and translation (rotation dominates in partial
  correlations).
- 15 scenes. CIs are scene-level but wide on some quantities (safe-radius policy lower bound 7
  points).
- Per-photo scale fit from COLMAP. Horns has a 27% median depth-ratio error.
- Prediction errors are mostly per-scene offsets (flower +0.15 LPIPS, orchids -0.12), while ranking
  within a scene is strong (rho 0.86-0.97). Single-image statistics tried so far (texture,
  self-render error, depth edges) don't remove the offset (exploratory). A first test of one-anchor
  calibration, using the nearest real view as a stand-in for a phone's second lens or Live Photo
  frame, raises rank correlation from 0.81 to 0.89 and safe-radius rho from 0.84 to 0.89, but barely
  moves R2 (0.66 to 0.67). The same anchor on a camera-distance model fails (R2 -0.13), so an anchor
  helps only on top of the geometry cue.
- LPIPS as a stand-in for perceived quality: no user study yet.

## Phase 3 (16 Sep): can a learned layer beat SHARP in the unseen regions?
Built a differentiable splatter in pure PyTorch (`nvs/splat_torch.py`, verified against gsplat to
0.0017 mean colour) so training runs on the Mac, then trained a background-layer head: per source
pixel, predict the colour/depth/opacity of the surface *hidden behind* it, render that layer at the
target camera, and supervise with the real photo inside the unseen mask only.

- **It does not beat SHARP.** Fair composite eval, 205 held-out pairs: L1 in unseen regions SHARP
  **0.144** vs layer composite **0.157-0.163**; LPIPS 0.362 vs 0.412-0.414; the composite wins 12-15%
  of pairs. Where the layer paints, it is worse than SHARP's smear (0.135-0.146 vs 0.113).
- **But the geometry half works:** the layer is the *steadiest* fill measured (flicker 6.6 vs 7.4 for
  a one-shot SDXL fill, 11.2 LaMa, 19.0 per-frame SDXL; raw SHARP is 3.1 but empty). Predicting once
  in 3D removes flicker by construction.
- **Only useful configuration:** apply the layer *only* where SHARP is truly empty - L1 0.1425 vs
  0.1440, LPIPS 0.359 vs 0.362 - but that is 0.6% of pixels, because SHARP's stretched Gaussians
  cover nearly everything.
- **Data is not the bottleneck.** Same model and steps on 2,000 vs 7,012 pairs: L1 0.1640 vs 0.1626,
  win rate 11.7% in both. 3.5x data changes essentially nothing.
- **Revised compute recommendation:** HPRC time should go to adapting a large pretrained generative
  prior (video / multi-view diffusion) or a better layer parameterisation - not to scaling this
  from-scratch head, which this ablation shows would waste the allocation.

## Proposed next steps
1. **Capture:** 10-20 phone orbits with dense, directionally varied cameras around a hero view
   (COLMAP-posed) to test per-direction prediction and the sub-degree regime.
2. **User study:** 2AFC on views inside vs outside the predicted safe radius, to calibrate the LPIPS
   threshold against perception.
3. **Second learned model:** run the same protocol on another feed-forward single-image 3DGS model
   (Flash3D needs a CUDA 11.8 toolkit build, so it needs a machine where installing one is okay).
4. **Second-lens anchor:** capture iPhone wide + ultra-wide pairs (a real, free anchor view) and test
   whether they calibrate the per-photo safe radius better than the proxy used here.
5. **Training signal:** use the predicted unseen share as an auxiliary target or loss weight when
   fine-tuning a SHARP-style decoder.

## Reproduce
Code in `~/Documents/single-view-nvs/nvs/`; the running log with every number is in `PROGRESS.md`.
Mac: `conda activate sharp; python -m nvs.predict_sharp --sources 8 --scenes ...`; Depth Pro in
`.venv-depth`. Desktop (`aurora:~/dev/single-view-nvs/.venv`, torch 2.4.0 + gsplat 1.5.3 cu124):
`python -m nvs.render_eval --method sharp --scenes all [--fill] [--own-masks]`,
`python -m nvs.export_pairs`. Analysis (Mac): `python3 -m nvs.analyze`, `python3 -m
nvs.predictability [method]`, `python3 -m nvs.bootstrap`, `python -m nvs.train_confidence --out
confidence_v2`, `python -m nvs.inpaint_experiment`, `python3 -m nvs.build_report`.
