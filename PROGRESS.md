# Single-image novel view synthesis — overnight log

Research project with Dr. Kumar. Idea: take one 2D photo and move the camera, like Apple's
SHARP (arXiv 2512.10685, github.com/apple/ml-sharp) and iOS "spatial scenes".

## Research question (working)
**How far can you move the camera from a single photo before it breaks, and can we predict
that limit per image?** SHARP-style feed-forward 3DGS is strong near the source view; quality
drops with baseline and with disocclusion. Step 1 measures that curve on real multi-view data
with known poses. Step 2 checks whether the drop is predictable from the single input (depth
discontinuities, occlusion edges), which would give a per-image "safe viewing cone".

## Machines
- Mac M4 Pro, 48GB (43GB GPU wired limit): SHARP inference (conda env `sharp`, py3.13,
  torch 2.8), Depth Pro (`.venv-depth`), big-memory training.
- Desktop `aurora` RTX 2070 Super 8GB: gsplat rendering + metrics. `~/dev/single-view-nvs/.venv`
  py3.10, **torch 2.4.0 + torchvision 0.19.0 cu124 + gsplat 1.5.3+pt24cu124** (the wheel is
  cp310 only and ABI-breaks on torch 2.4.1). No CUDA toolkit on the box, so prebuilt wheels only.

## Pipeline
1. `nvs/scenes.py` - LLFF COLMAP poses at images_4 (1008x756), cached to npz (pycolmap and torch
   clash on libomp on macOS, so pycolmap only runs in a subprocess).
2. `nvs/predict_sharp.py` (Mac) - 3 source views per scene (center + 2 farther), fp16 Gaussians.
3. `nvs/depth_depthpro.py` (Mac) - Depth Pro depth. HF `predicted_depth` is **inverse** depth.
4. `nvs/make_warp.py` (desktop) - naive 3D-photo baseline: one opaque Gaussian per pixel.
   `warp_sharpdepth` uses SHARP's own depth (isolates SHARP's learned decoder); `warp_depthpro`.
5. `nvs/render_eval.py` (desktop) - one scale per source fit to COLMAP sparse depth (median),
   render all other views, then PSNR/SSIM/LPIPS full + covisible + disoccluded. Masks come from
   SHARP geometry and are reused for all methods. Also scores the identity baseline (reuse the
   source photo).

**Shared machines (user, 2026-09-15):** another project of the user's runs on the Mac and the
desktop. Never kill processes this project didn't start. Check load (`uptime`, `nvidia-smi`,
`memory_pressure`) before launching, run jobs under `nice`, cap threads (`OMP_NUM_THREADS=4`),
and leave GPU memory and CPU headroom. Use whichever machine is idle.

Sync: `rsync` code, `results/sharp`, `results/depth` and `images_4` to `aurora:dev/single-view-nvs`.

## Status
- [x] envs, LLFF (8 scenes), SHARP on 24 sources (5.7s each on MPS, 1.18M Gaussians)
- [x] eval harness validated on fern (SHARP beats identity by ~3.5dB at the nearest view, so
      the pose direction is right; figure looks right)
- [x] LLFF eval: sharp, warp_sharpdepth, warp_depthpro; source_features (LLFF)
- [~] LLFF hole-filled warps (`--fill`, Telea, CPU-heavy), running on desktop
- [x] Mip-NeRF 360: 7 scenes (images_4 + sparse, 1.2GB), SHARP + Depth Pro on 21 sources
- [~] `nvs/queue_m360.sh` on desktop (waits for the fill job): sharp eval, features, warps, fills
- [x] predictability v1 on LLFF (see findings)
- [x] `nvs/export_viewer.py`: 7x7 camera-grid sprite sheets (SHARP / warp / unseen mask),
      verified on fortress:21 (~3MB per scene)
- [~] **Confidence-map experiment:** `nvs/export_pairs.py` (desktop, queued via
      `nvs/queue_pairs.sh` after the 360 queue; ~1MB per pair, width 384) writes SHARP render +
      alpha + relative depth + unseen mask, with GT and LPIPS map as targets.
      `nvs/train_confidence.py` (Mac MPS) trains a 1.95M-param UNet to predict the per-pixel LPIPS
      map, with scene-level 3-fold CV vs a "mask prior" (least squares on unseen + 1-alpha).
      Metrics: pixel Spearman, AUROC for bad pixels (above the train 80th percentile), image Spearman.
- [ ] predictability v2 with 15 scenes; export ridge coefficients for the viewer's quality readout
- [x] **Report published v1: private link**, built from
      `report/index.html` + `report/data.json` via `python3 -m nvs.build_report`. Republish the same
      file path with the `files` map (data.json + assets/*) to update. The confidence section
      appears automatically once `results/confidence/results.json` exists (rebuild data, then
      also add the `assets/fig_*.jpg` files).
- [x] report artifact: viewer hero plus charts (LPIPS vs distance; covisible vs unseen;
      predictability). Design plan: ground #F2F4F3 / ink #171B20, SHARP blue #2a78d6, warp
      orange #eb6834, Depth Pro aqua #1baf7a, identity grey, unseen-mask red #EB4034; type
      Familjen Grotesk (display), Source Serif 4 (prose), JetBrains Mono (figures)

## Notes / findings
- Pose convention verified (`nvs/diag_pose.py`): `T = w2c_t @ inv(w2c_s)` with t/scale beats
  inv(T), R-only, t-only and scale x0.5/x2 by 3-7dB on fern and room.
- SHARP self-render (no camera move) PSNR: fern 27.0, room 32.3. The representation is lossy
  on foliage before any motion happens.
- SHARP LLFF (891 pairs): PSNR 15.4 vs identity 12.1; LPIPS 0.360 vs 0.503; beats identity on
  LPIPS in 90% of pairs. PSNR is flat across baseline bins (15.9, 15.0, 15.5, 15.4, 15.2) because
  even the nearest LLFF neighbours are far enough that misalignment saturates PSNR. LPIPS does
  trend (0.295 to 0.388), and disoccluded regions are much worse (LPIPS 0.459 vs 0.327 covisible).
  Use LPIPS plus the covis/disocc split as the headline metrics.
- Naive pixel-Gaussian warps (no hole fill), LLFF: warp_sharpdepth PSNR 11.86 / LPIPS 0.460,
  warp_depthpro 11.82 / 0.466. **Covisible PSNR is nearly equal** (SHARP 15.80, warp_sharpdepth
  15.69, warp_depthpro 15.23), so on what the source photo sees SHARP is basically its depth map.
  The learned decoder earns its keep in disoccluded regions (LPIPS 0.459 vs 0.68). SHARP wins
  on LPIPS in 100% of pairs vs both. Hole-filled (Telea) variants are queued for fairness.
- **Hole-filled warps (Telea), LLFF, 891 pairs:** warp_sharpdepth_fill PSNR 15.06 / LPIPS 0.437,
  warp_depthpro_fill 14.97 / 0.442, SHARP 15.40 / 0.360. With holes filled, **SHARP's PSNR lead is
  only 0.35dB, but it still wins LPIPS in 100% of pairs (mean 0.076).** The LPIPS gap grows with
  distance (0.058 in the nearest bin to 0.088 in the farthest) and holds in covisible regions
  (0.327 vs 0.382), where pixel warps stretch and alias at depth edges. SHARP's advantage is
  perceptual, not pixel accuracy.
- **All 15 scenes (8 LLFF + 7 Mip-NeRF 360, 1350 pairs):** SHARP PSNR 15.44 / LPIPS 0.388;
  warp_sharpdepth_fill 15.47 / 0.458; warp_depthpro_fill 15.41 / 0.461; identity 12.10 / 0.536.
  PSNR is a tie (-0.03dB), but **SHARP wins LPIPS in 98% of pairs (mean 0.070)**. On 360, SHARP
  loses PSNR in 5/7 scenes (m360_room 16.9 vs 18.3) while winning LPIPS in all 7. Cause, from
  the figures: SHARP leaves out-of-frame regions black (alpha 0) and Telea paints smooth colour
  there, which PSNR rewards, and that area grows with baseline. `sharp --fill` is queued
  (`nvs/queue_sharpfill.sh`) for an apples-to-apples PSNR.
- **Apples-to-apples (both Telea-filled), 15 scenes:** sharp_fill PSNR 15.95 / LPIPS 0.383 vs
  warp_sharpdepth_fill 15.47 / 0.458. **SHARP +0.49dB PSNR, lower LPIPS in 100% of 1350 pairs
  (mean 0.075).** That confirms the black out-of-frame hypothesis: filling moved SHARP's own
  disoccluded PSNR from 14.19 to 15.01, while covisible PSNR is unchanged (16.62). The report
  uses the filled variants for every method.
- **Predictability v2 (15 scenes, LOSO):** camera move only R2 0.318, within-bin rho 0.403;
  **+ SHARP-predicted disocclusion: R2 0.661, within-bin rho 0.765**, which holds on each dataset
  alone (LLFF 0.43/0.64, 360 0.75/0.87). Source image stats still hurt out-of-scene (0.553 / 0.670),
  so drop them. Headline: SHARP's own geometry at the requested pose roughly doubles explained
  variance in how bad the view will look, with no GT.
- **Movement-limit policies (15 scenes, LOSO predictions; bad = LPIPS > median 0.389):** allowing
  50% of pairs, a fixed distance limit yields 43.7% bad views vs 21.0% for a predicted-quality
  limit (camera move + predicted disocclusion); at 20% it's 25.2% vs 6.3%. **But within a single
  photo the gain vanishes** (25/50/75% coverage: fixed 14.8/31.4/42.5%, adaptive 16.3/30.8/41.7%).
  The signal identifies *which photos* tolerate movement (a per-image safe radius), not *which
  directions* are safe within a photo. Caveat: LLFF/360 targets from one source differ mostly in
  distance, not direction, so the within-photo test is weak.
- **Within-photo ranking splits by capture type (01:35):** mean per-photo Spearman of LOSO
  predicted vs actual LPIPS: LLFF predicted 0.879 vs distance 0.922 (24 photos); **360 predicted
  0.825 vs distance 0.685** (21 photos). Where targets differ in direction (360 orbits), the
  prediction ranks a photo's own views better than distance. On forward-facing LLFF grids
  (mostly distance), it doesn't. The report caveat is updated.
- **Confidence UNet, folds 0-1:** fold 0 (fern, horns, bicycle, garden, stump, room) pixel rho
  0.681 vs mask prior 0.469, AUROC 0.854 vs 0.802. Fold 1 (flower, leaves, bonsai, kitchen, trex)
  0.398 vs 0.325, AUROC 0.683 vs 0.677. It gains most where errors are geometric; foliage stays hard.
- **Confidence UNet final (3-fold scene CV, 1350 pairs):** pixel Spearman **0.611 vs mask prior
  0.389**; AUROC (worst 20% pixels) 0.823 vs 0.789; pixel MAE 0.151 vs 0.172; image-level rho
  0.823 vs 0.808. Fold 2 (fortress, counter, 360 room, orchids) scored 0.755 vs 0.375. The
  constant baseline's rank metrics are meaningless (ties), so the report shows n/a. Figures are
  regenerated by `nvs/conf_figs.py` (shared per-row scale, median-distance pair). Report v2 is
  published with this section.
- **02:00 generalisation test:** Flash3D was dropped (needs nvcc/CUDA 11.8 compilation and the
  desktop has no toolkit; installing one is a system change). Instead, `render_eval --own-masks`
  scores a method with covisibility from its *own* geometry, and `python3 -m nvs.predictability
  <method>` fits the predictor for that method. Already, using SHARP's masks to predict
  **warp_depthpro_fill** LPIPS gives R2 0.668 and within-bin rho 0.762, the same as for SHARP
  itself, which suggests the signal is scene occlusion structure rather than something
  SHARP-specific. TODO after re-scoring: `bash nvs/queue_own_masks.sh` on the desktop (it was
  stopped so it wouldn't run in parallel with the re-scoring).
- **02:15, 120 source photos (8 per scene, 3607 pairs):** predictor R2 0.648 (camera only 0.284);
  within-bin rho 0.791 (0.404). Policy at 50%: fixed limit 43.4% bad vs predicted 20.6%.
  **Safe radius: Spearman 0.823 over 119 photos; median |log err| 0.437 (x1.5) vs constant 0.965
  (x2.6)**; LPIPS@0.15 rho 0.863. Within-photo rho: LLFF predicted 0.912 vs distance 0.933 (64
  photos); 360 predicted **0.870 vs 0.676** (56 photos).
- **Generalisation (Depth Pro warp, own-depth masks, 45 photos):** R2 0.278 to 0.549, within-bin rho
  0.384 to 0.731, safe radius Spearman 0.786 (log err 0.390 vs 0.651). The idea transfers to any
  explicit-geometry renderer. It's weaker than the same warp predicted with SHARP's masks (R2
  0.668), so SHARP's geometry estimates true disocclusion better than Depth Pro's.
- **Related work / novelty check (02:25):** **US 12051150** (BR priority 2021-12-21) classifies a
  candidate image for single-image 3D effects by an image score from the number of non-occluded
  pixels in generated target views. That is our disocclusion feature used as a binary suitability
  score, so the core cue is prior art. Also relevant: arXiv 2603.22786 (predictive photometric
  uncertainty for per-scene multi-view 3DGS, not single-image) and arXiv 2606.07514 UniSHARP (SHARP
  for any camera model; no quality prediction). Defensible novelty: real-GT validation on
  held-out scenes, continuous per-photo safe radius, per-direction ranking (360), pixel-level
  confidence for feed-forward single-image 3DGS, cross-renderer transfer, perceptual-vs-pixel
  analysis. Tell Dr. Kumar before framing a paper.
- **03:00 viewer overlay:** `export_viewer.py` now also saves `conf_inputs.npy` (alpha, depth_rel,
  unseen per cell; ~49MB per scene, Mac only, deleted from the desktop). `nvs/conf_viewer.py <run>
  <keys>` runs the held-out fold UNet on each cell and writes a `conf.png` sprite (alpha =
  predicted LPIPS on a 0.35-0.8 scale). Report v5 has a "Predicted error" overlay toggle. It
  currently uses the v1 models; rerun with `confidence_v2` once that training finishes.
- **03:35 cross-dataset transfer (`transfer()` in predictability.py, 120 photos):** train LLFF,
  test 360: camera-only R2 **-1.010** (rho 0.750) vs + disocclusion R2 **0.696** (rho 0.872). Train
  360, test LLFF: 0.092 (0.585) vs 0.518 (0.789). Distance rules don't transfer across capture
  styles; the geometry cue does. Reported in Finding 3 (it goes live on the next publish).
- **Motion type (partial Spearman with LPIPS):** rotation controlling for translation: LLFF 0.624,
  360 0.755. Translation controlling for rotation: -0.372 / -0.095 (trans and rot are correlated
  0.77 on LLFF, so read that cautiously). Disocclusion controlling for both: 0.764 / 0.815, so the
  geometry cue carries information beyond motion magnitude. Rotation is also confounded with
  out-of-frame area.
- **Confidence v2 fold 0 (3607 pairs, 25 epochs):** UNet pixel rho 0.693 (v1 0.681), AUROC 0.866
  (v1 0.854); mask prior 0.502 / 0.807.
- **04:15 confidence v2 final (3607 pairs, 120 photos, 3-fold scene CV, 25 epochs):** UNet pixel rho
  **0.625** (mask prior 0.414), AUROC 0.830 (0.786), MAE 0.158 (0.177), image rho 0.838 (0.829). v1 on
  1350 pairs was 0.611 / 0.823. The folds: fern/horns/bicycle/garden/stump/room 0.693; foliage fold
  (flower/leaves/bonsai/kitchen/trex) 0.414; fortress/counter/360 room/orchids 0.768. Report v6 uses
  the v2 models for the figures and viewer overlay (`conf_figs.py confidence_v2 ...`,
  `conf_viewer.py confidence_v2 ...`).
- **04:25 actionability experiment (`nvs/inpaint_experiment.py`, .venv-depth + simple-lama-inpainting,
  big-lama.pt in ~/.cache/torch/hub):** same pixel budget (the unseen share, clipped 2-50%); inpaint
  with LaMa at (a) the unseen mask, (b) the top pixels of the held-out UNet predicted error, (c) a
  random control (another pair's predicted blobs); compare with raw. LPIPS/PSNR vs GT on ≤40 pairs per
  scene. Output in results/inpaint/.
  **Result (600 pairs, mean area ~30%), negative for confidence as a repair guide:** inpaint unseen:
  dLPIPS +0.025 (better in 65%), dPSNR +0.72; inpaint conf-top: +0.006 (48.5%), +0.37; random:
  -0.032 (2.7%), -0.33. Conf beats random 88% but loses to unseen 80%. Interpretation: flagged
  covisible regions still hold mostly-right (misaligned) content that inpainting destroys; unseen
  regions are black or smeared, so filling them only helps. **Confidence maps are for warning users
  where not to trust a view; repair should target the geometric unseen mask.**
  Follow-up (results/inpaint_v2, same 600 pairs): unseen dilated 6px (area 52%): dLPIPS **-0.034**
  (better in 25%, beats plain unseen 0.7%); unseen + top-5% conf (area 32%): +0.0237 vs unseen
  +0.0246, beats unseen in only 14%. The exact geometric mask is the right repair target.
- **04:20 scene-level bootstrap (`nvs/bootstrap.py`, 1000 resamples of 15 scenes, LOSO preds fixed):**
  R2 camera-only 0.284 [-0.011, 0.481]; + disocclusion 0.648 [0.439, 0.780]; **paired R2 gain 0.364
  [0.204, 0.551]**. Within-bin rho 0.404 [0.150, 0.675] vs 0.791 [0.625, 0.883]; gain 0.387 [0.156,
  0.577]. Safe-radius rho 0.823 [0.705, 0.901]. Bad@50% fixed 0.434 [0.286, 0.550] vs predicted 0.206
  [0.052, 0.401]; reduction 0.228 [0.066, 0.326]. SHARP vs pixel warp (filled) LPIPS gain 0.075 [0.066,
  0.084], win 100% [100, 100]. Every headline gain's 95% CI excludes 0.
- **04:25 `RESEARCH_BRIEF.md` written** (question, setup, 6 findings with CIs, prior art, threats,
  next steps, reproduce). Report v9 adds bootstrap CIs to Finding 3.
- **04:30 mechanism ablation queued:** `nvs/mask_parts.py` (desktop) splits the unseen share into
  out-of-frame vs occluded-but-in-frame vs hole. Then: does predictability come from frame borders or
  true disocclusion?
  **Result (3607 pairs, LOSO R2 / rho):** mean shares out-of-frame 0.132, occluded 0.116, hole 0.062.
  camera only 0.284/0.495; + out-of-frame **0.231**/0.440 (hurts); + occluded **0.631**/0.803; + both
  0.678/0.822; + combined unseen 0.648/0.795. LLFF: out-of-frame -0.120 vs occluded 0.436 (the best).
  360: out-of-frame 0.557, occluded 0.717, both 0.760. **The signal is true disocclusion behind
  foreground surfaces, not frame-border overlap.** Frustum overlap alone doesn't predict quality
  out-of-scene. Robust to the z-test tolerance: occluded-only R2 0.634 / 0.631 / 0.625 and both 0.681 /
  0.678 / 0.665 at tol 0.02 / 0.05 / 0.15 (`python -m nvs.mask_parts <tol>`).
- **04:30 exploratory residual analysis (base = move + out-of-frame + occluded + hole, R2 0.678):**
  adding single test-time source stats doesn't help (self_lpips 0.646, texture 0.644,
  depth_edge_frac 0.610, depth_log_range 0.627, opacity_lowfrac 0.670). align_err helps (0.697) but
  uses COLMAP GT, so it isn't test-time. Per-scene residuals are **offsets** (flower +0.152,
  orchids -0.120, fortress -0.107, 360 room -0.078) while within-scene rho is 0.86-0.97. The
  predictor ranks well; what remains is per-scene calibration that single-image stats don't
  capture. Exploratory (6 features tried), so don't headline it.
- **05:02 one-anchor calibration (`results/anchor_calibration.json`):** per photo, the nearest real
  view (median offset/depth 0.038; stands in for a second lens or Live Photo frame) sets an additive
  offset; evaluated on the other 3487 pairs. Geometry model: R2 0.662 to 0.672, rho 0.813 to **0.885**,
  safe-radius rho 0.842 to **0.893** (log err x1.37 to x1.33). Camera-only + anchor: R2 **-0.134**, radius
  rho 0.298 (x3.0). An anchor helps ranking and the radius, but only together with the geometry cue;
  offset calibration alone is useless.
- **05:07 model-class check (LOSO, HistGradientBoosting vs ridge):** camera-only 0.292 vs 0.284;
  move+oof+occ+hole **0.690 vs 0.678**; + 5 source stats 0.639 vs 0.590. The linear model is
  essentially sufficient, and source stats hurt out-of-scene even for trees. scikit-learn is
  installed in .venv-depth.
- **Status 05:08:** every planned milestone is done. Anything more useful needs new data (phone
  orbits, a second-lens capture, a user study; see RESEARCH_BRIEF next steps). The loop holds with
  hourly heartbeats until 08:00 and posts a final summary.
- Memory saved: `project-single-view-nvs.md`. Desktop footprint: ~14G in ~/dev/single-view-nvs (venv
  ~6G, results/sharp 3.7G, data ~1.5G, warp_depthpro 0.9G). Left in place for reproducibility; the user
  can delete `results/sharp` there (the Mac has copies). `results/warp_depthpro` exists only on the
  desktop, but `python -m nvs.make_warp --depth depthpro --scenes ...` regenerates it.
- Report v11: setup copy fix (8 sources per scene). Verified all charts and sections render, no console errors.
- Report v3 published (safe-radius keyfacts, generalisation paragraph, 3607-point scatter).
- **02:00 INCIDENT: desktop disk hit 100%** (0 bytes free). Both launchers of the re-scoring had
  run in parallel (the Mac waiter I thought was dead was still rsyncing) and crashed with
  `OSError: requested and written`. Disk use was 91G at 00:14 and 130G at 01:56. Mine: uv cache
  9.3GiB (three torch installs), pairs 1.3G, warp gaussians 1.7G, sharp 3.7G, figs 0.4G. Roughly
  15G of the growth is not from this project. Freed: deleted the desktop copies of `results/pairs`
  (all on the Mac), `results/warp_sharpdepth` (reproducible, evals saved), `results/figs`, and ran
  `uv cache clean` (the cache dir dates from 2026-08-31, so it may also have held the user's
  entries; it's a download cache, re-fetched on demand). Now 12G free; venv verified; JSONLs
  validated; results/sharp re-synced with --checksum. **Keep desktop footprint small:** don't
  store pairs or gaussians there longer than needed.
- **Predictability (LLFF, LOSO ridge, `nvs/predictability.py`):** target is SHARP LPIPS per pair.
  Camera-move-only features: R2 0.17, within-baseline-bin Spearman 0.32. Adding SHARP-predicted
  disocclusion (1-covis from SHARP geometry at the requested pose, so no GT needed): R2 **0.43**,
  within-bin rho **0.64**. Source image stats overfit with 8 scenes (R2 negative), so re-test after
  adding the Mip-NeRF 360 scenes.
- 2026-09-15 00:26: my desktop waiter used `pgrep -f` with a pattern that matched its own
  `bash -c` command line, so it waited forever. Use `pgrep -f "^python -m nvs\."` or PID waits.
  Repeated at 01:50 on the Mac, now fixed with the bracket trick: `pgrep -f "[n]vs.predict_sharp"`.
- **Safe radius per photo (45 photos, `safe_radius()` in predictability.py):** fit LPIPS vs log
  offset per photo and find where it crosses tau. Predicted radius vs measured: Spearman 0.698;
  median |log error| 0.416 (x1.5) vs 0.803 (x2.2) for a constant radius taken from other scenes.
  Predicted LPIPS at offset 0.15 vs measured: Spearman 0.839. Rerun with 8 sources per scene
  (~120 photos) is in progress.
- Scale-fit residual (median |log ratio|): fortress 0.02, fern 0.06, flower 0.13, horns 0.29
  (horns is bad, so check whether its SHARP depth is off on the glass/railings).

## Loop schedule
User (2026-09-15 00:36): keep the /loop running **until 08:00 CDT on 2026-09-15** unless told
otherwise. Every iteration re-arms ScheduleWakeup, and none stops early just because the milestones are
done: pick the next useful experiment instead. A `caffeinate -i -s` running until about 08:00
keeps the Mac awake on AC.

## 09:30 Phase 2: generative fill pilot (local)
User direction: fabricating plausible content in unseen regions is fine (user opted into the effect),
so GT-match metrics are the wrong yardstick there. Keep compute local for as long as possible (TAMU
HPRC later). **Storage:** the desktop's 1TB NTFS HDD (`/dev/sdb2` DATA) is mounted on demand at
`/mnt/data` (ntfs3 rw, no fstab entry; remount after reboot:
`sudo mount -t ntfs3 -o rw,noatime,uid=1000,gid=1000,umask=022 /dev/sdb2 /mnt/data`).
`~/dev/single-view-nvs/{data,results}` are symlinks into `/mnt/data/single-view-nvs/`. Windows Fast
Startup is verified off (HiberbootEnabled=0, HibernateEnabled=0, no hiberfil.sys). SSD usage by this
project is 7.3G (venv + code). Another ~5G of SSD loss came from containerd image pulls, not this project.
- `nvs/render_paths.py` (desktop): 25-frame horizontal sweeps (+-0.3 x median depth, look-at) for
  m360_kitchen:30, fortress:21, horns:33, m360_counter:117, m360_bonsai:12. Unseen share is 20-49% at the
  ends and ~4-7% at the centre.
- `nvs/fill_pilot.py` (Mac, .venv-depth + diffusers 0.40 + SDXL-inpainting-0.1 fp16, 37s/frame at
  1024px on MPS): LaMa per frame; SDXL per frame (fixed seed); "lift": SDXL fills of the two end
  frames, max-depth-dilated background depth, unprojected to points. `nvs/render_lifted.py` (desktop)
  re-renders the path with those points as Gaussians. Kitchen check: SDXL continues the woven mat
  texture behind the truck and fabricates a plausible dark post in the out-of-frame region; there is
  a stray green tile at the bottom edge.
- **10:10 first lift results (kitchen montage):** raw lifted re-render (`lifted.npz`) fabricates
  consistent out-of-frame structure (table edge and bench at the sweep ends), but opaque fill points
  leak through SHARP's soft truck edges as grey/white speckles in the middle frames. The layered
  composite (`layered.npz`: fill layer shown only inside each frame's dilated unseen mask) reduces
  but doesn't remove it. Cause: the keyframe fill masks include pixels on the truck silhouette, and
  those get background depth, so they become floaters behind the truck that show in other views'
  unseen areas. There's also a grey zigzag band along the bottom edge in the end frames.
  **Next:** save keyframe fills in fill_pilot (so no SDXL re-runs); lift only pixels >=3px from
  foreground depth edges; use 4 keyframes (0, 6, 18, 24), each frame taking fill from the nearest
  keyframe; then `nvs/fill_eval.py` stability on raw/lama/sdxl/lifted/layered.
- **11:40 stability (`nvs/fill_eval.py`, RAFT-large flow on the raw frames, backward-warp t+1 to t,
  fwd-bwd check; mean abs colour change 0-255 inside frame-t unseen pixels; 5 scenes x 24 frame pairs):**
  raw 3.1 (black and smear stay put, so it is "stably wrong" and not a real winner) | LaMa 11.2 |
  **SDXL per-frame 19.0** | **lifted 7.4** | layered 8.5. Lifting the generative fill into 3D cuts
  SDXL's flicker by 61% and beats LaMa, even with the v1 floaters. Seen-pixel sanity: raw 1.6-3.3;
  lifted leaks into seen pixels on horns (9.5) and bonsai (4.2); layered brings horns to 5.7.
  Per scene, unseen: kitchen 3.2/10.2/17.2/6.5/7.4, fortress 2.7/8.1/20.2/4.0/4.6, counter
  3.2/12.5/18.7/7.0/8.0, horns 3.7/14.5/20.6/12.4/14.3, bonsai 2.7/10.5/18.4/7.0/7.9.
- **12:10 lift v2 (4 keyframes 0/6/18/24, saved fills in keyframes_v2.npz, silhouette-adjacent pixels
  dropped (only 1-8% of mask px), nearest-keyframe layer per frame): WORSE.** Unseen change: lifted_v2
  8.7, layered_v2 10.2 vs v1 lifted 7.4, layered 8.5. Seen change: raw 2.24, lama 2.49, sdxl 3.47,
  lifted 4.11, layered 3.25, lifted_v2 4.44, layered_v2 3.18. The nearest-keyframe switches add
  temporal jumps, and the silhouette filter barely touches the ghosting. The visual check still shows
  grey/white patches behind the truck cab in v1 and v2: keyframes are inpainted at *novel* views
  whose context includes SHARP's semi-transparent edge smear, so SDXL continues the grey, and
  each keyframe's unseen region has a different shape.
  **Next, more principled (3D-photo style):** split the *source* view at depth discontinuities into
  foreground/background, inpaint the background behind foreground edges once in the source view
  (context = background pixels only, foreground masked out), place it at background depth as its own
  Gaussian layer, and composite behind SHARP. One fill, one geometry, so it should be stable by
  construction. If multiple fills are needed, blend by camera distance instead of switching.
- TODO: consistency metric (RAFT flow warping error inside unseen masks between consecutive frames),
  no-reference quality (pyiqa), and a viewer demo with SHARP / LaMa / SDXL / lifted side by side.

## 08:00 wrap-up
Loop ended on schedule. No jobs running on either machine; desktop at 12G free. Deliverables: report
v12 (private link), `RESEARCH_BRIEF.md`, this log, and code in `nvs/`.
- **Fill comparison viewer published: private link**
  (`fillviewer/index.html` + `data.json` + `assets/<scene>/<method>.jpg|mask.png`, 5x5 frame sprites
  at 448px, 23MB). It has a scene picker, single or side-by-side views, drag/slider/play over the
  25-frame sweep, an unseen outline, per-method flicker bars, and the across-scene table. To add a
  method: write `results_local/fill/<scene>/<method>.npz`, add it to `methods` in the sprite build
  snippet, rerun it, and republish with the new asset files.

## 2026-09-16 overnight: background-layer training (Phase 3)
- `nvs/splat_torch.py`: differentiable Gaussian splatting in pure PyTorch so training can run on the
  Mac (gsplat is CUDA-only). Verified against gsplat (`nvs/check_splat.py`, m360_kitchen:30, 256px):
  mean |colour diff| 0.0017, alpha 0.0001, depth 0.02%; 4.2s vs 0.35s. `window=` renders a crop
  (verified identical to the full render within 0.003) - training uses 160px crops centred on unseen
  pixels after a full-frame graph blew the Mac's memory at step 3400.
- `nvs/export_train_pairs.py` (desktop): 3607 pairs with poses, 3.3G on the 1TB drive.
- `nvs/train_bglayer.py` (Mac): BGNet (2M-param UNet) predicts per source pixel a hidden-surface
  colour, a depth behind the front surface, and an opacity; points are rendered at the target camera
  and compared with the real photo inside the unseen mask only. Fold 0 (test: fern, horns, bicycle,
  garden, stump, room), 6000 steps, ~0.6s/step with crops.
- **Negative result (`nvs/eval_bglayer.py`, fair composite, 204 held-out pairs):** L1 in unseen
  regions SHARP **0.135** vs composite **0.155** (layer alone 0.247); the composite beats SHARP in
  only 10% of pairs; the layer covers 60% of unseen pixels. Training loss plateaued ~0.20 after
  ~3000 steps. Cause: 2184 training pairs + L1 loss -> blurry averages, and SHARP's smear beats blur
  on L1. The training-time eval was unfair (bare layer vs SHARP's full render, 0.254 vs 0.139) -
  use the composite eval.
- Next: (1) render the viewer paths with the layer and publish it as an 8th method; (2) scale data
  ~4x with 30 sources per scene (~13k pairs, free, local); (3) predict a *residual* over a classical
  background estimate and add a perceptual term, so the model stops preferring blur.
- **Stability of the trained layer (same RAFT protocol, 5 sweeps):** bglayer **6.57**, the steadiest
  fill of all (lifted 7.4, layered 8.46, lama 11.16, sdxl 19.03; raw 3.1 is empty-but-static). So
  the 3D-layer design delivers stability by construction; only the *content* is wrong (L1 0.155 vs
  SHARP 0.135). That splits the problem cleanly: geometry/consistency solved, appearance needs data
  and a perceptual objective - which is what the v2 run tests.
- **v2 (11909 pairs from 399 sources, 7012 train; residual over a Telea background estimate; +LPIPS
  term; 9000 steps):** fair composite eval on 205 held-out pairs: L1 unseen SHARP **0.1440** vs
  composite **0.1574** (v1 gap was +0.0196, v2 +0.0134, so ~30% narrower); composite beats SHARP in
  15.1% of pairs (v1 10.3%). Two OOM kills along the way (full-frame graph, then at step 5300);
  fixed with 128px crops, LPIPS every 4th step, empty_cache every 20, and a skip-on-OOM guard
  (6 skips in 4000 steps, 0.16%).
- **Diagnosis (`nvs/eval_bglayer.py` extras):** where the layer actually paints, it is *worse* than
  SHARP's smear (L1 0.135 vs 0.113; SHARP wins 85% of pairs), so coverage (57%) isn't the problem -
  the predicted content is. LPIPS agrees (composite 0.4145 vs SHARP 0.3619, layer wins 4%), so this
  isn't the "L1 rewards blur" trap.
- **Only useful configuration: holes-only compositing** (use the layer *only* where SHARP's alpha is
  empty): L1 **0.1425 vs 0.1440**, LPIPS **0.3587 vs 0.3619** - slightly better than SHARP, but it
  touches just 0.56% of pixels, because SHARP's stretched Gaussians cover nearly everything.
- **Conclusion so far:** SHARP's learned stretch is a strong baseline in disoccluded regions; a small
  from-scratch layer on ~7k pairs cannot beat it. Headroom exists only where SHARP is truly empty.
  Beating it needs either a pretrained generative prior (SDXL/video-diffusion scale, as CAT3D/GEN3C
  do) or far more data. Running now: a clean **data-scaling ablation** (same model and steps, 2k vs
  7k pairs) to measure whether more data actually moves the number - the evidence that decides
  whether HPRC-scale training is worth it.
- **Data-scaling ablation (identical model, 4000 steps, same held-out fold; `nvs/run_scaling.sh`):**
  2,000 pairs -> L1 composite **0.1640**, used-region 0.1465, LPIPS 0.4126, wins 11.7%;
  7,012 pairs -> L1 composite **0.1626**, used-region 0.1458, LPIPS 0.4136, wins **11.7%**.
  **3.5x the data buys -0.0014 L1, +0.0010 LPIPS and zero change in win rate.** Data is not the
  bottleneck at this scale - the from-scratch appearance model is. Practical consequence: do NOT
  spend HPRC time scaling this architecture; spend it adapting a large pretrained generative prior
  (video/multi-view diffusion, as CAT3D/GEN3C do), or on a fundamentally better layer parameterisation.
- **Measured (250 pairs):** SHARP covers **75.0%** of unseen pixels with opaque stretched content;
  **25.0%** is genuinely empty (mostly beyond the source frame). Two consequences: (a) benchmarks
  that score unseen pixels are largely scoring smear, so a fair protocol should mark them unknown;
  (b) a background layer built from *source pixels* structurally cannot fill beyond-frame regions,
  which caps this design. (An earlier draft of the report said "99.4% covered" - wrong, from
  misreading hole_share as a share of unseen rather than of all pixels; corrected to 75%.)
- **Beyond-frame outpainting (exploratory, `nvs/outpaint_layer.py`):** place the source view on a
  1.5x canvas, SDXL-outpaint the border, extrapolate depth outward from the frame edge, lift the new
  pixels to 3D points (215-246k per scene), composite inside the unseen mask. Visually it does fill
  some beyond-frame area but coverage is patchy and the extrapolated depth puts content at the wrong
  distance. **Measured share of unseen that is beyond-frame: 8.3% along the viewer sweeps** (kitchen
  11.5%, fortress 13.3%, counter 12.8%, horns 3.4%, bonsai 0.5%) **vs 25% across the held-out eval
  pairs**, whose targets rotate more - so the structural limit is protocol-dependent. Not added to
  the viewer: the composite still Telea-fills everything the layer misses, so ~90% of changed unseen
  pixels come from Telea rather than from the outpainted layer, which would make the comparison
  misleading. Report wording corrected to give both numbers.

## 2026-09-16 08:00 wrap-up (Phase 2-3)
Deliverables: report v14 (private link) with the follow-up
section; fill comparison viewer v3 (private link) with 8
methods; `RESEARCH_BRIEF.md` with a Phase 3 section and the revised compute recommendation; this log.
Code: 29 scripts in `nvs/`. Checkpoints (bglayer, bglayer_v2, bglayer_small, bglayer_full), rendered
fills and docs are backed up to the desktop's 1TB drive.
Machine state: Mac 31G free (20G is this project, mostly the 11,909 training pairs, which also exist
on the 1TB drive and can be deleted locally); desktop 205G free on /mnt/data, 11G on its SSD.
Nothing running on either machine.
Open threads for the next session: (1) adapt a pretrained video/multi-view diffusion prior instead of
training a small head; (2) capture phone orbits + wide/ultra-wide pairs for per-direction prediction
and anchor calibration; (3) a fair benchmark protocol that marks SHARP's smear as unknown rather than
scoring it; (4) user study to calibrate the LPIPS threshold behind the safe radius.
- **Anatomy of the unseen region (`nvs/unseen_anatomy.py`, 15 scenes x <=40 pairs, depth-gap
  threshold 5% of scene depth):** **beyond-frame 64.9%**, **background behind an object 34.0%**,
  **foreground object's own hidden surface 1.2%**. Per scene, beyond-frame ranges 31% (orchids) to
  93.6% (360 room). Consequence for planning: object-level image-to-3D completion addresses ~1% of
  the problem under lateral/phone-style motion (it would matter for orbits around a hero object);
  outpainting past the frame plus background inpainting behind objects covers ~99%.
  **Correction:** the earlier note (and report v14) said "25% is genuinely empty, mostly beyond the
  frame" - that 25% is the share where SHARP's alpha is empty, which is NOT the same as beyond-frame
  (65%). SHARP covers much of the beyond-frame area with stretched splats. Report wording fixed.
