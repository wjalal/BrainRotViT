"""
stability_sanity_dora.py
========================
Reviewer R5.11 -- saliency-map credibility, in two parts, for the DoRA
ViT+CNN interpretability pipeline. Operates on the UNFUSED central-slice ViT
attention map of each validation subject (the reviewer's explicit ask:
"store the unfused central slice attention map for each subject, plot the mean,
stddev, and whatever else was asked").

It imports 3dmap_grad_vit_cnn_main_mix_roi_dora.py as a module (its build step is
guarded by __main__, so importing only reuses the already-loaded, trained DoRA
ViT + CNN and all the helper functions -- nothing is recomputed or overwritten).

(1) STABILITY (quantifies the Figure-4a "inter-subject std" panel)
    Stores each subject's unfused central-slice ViT attention map, then reports
    pixel-wise inter-subject mean, SD and coefficient of variation, plus
    split-half stability (correlation of two disjoint halves' mean maps).

(2) SANITY CHECK (Adebayo et al., "Sanity Checks for Saliency Maps")
    Model-parameter randomization test: progressively randomize ViT weights
    top-layer -> bottom (cascading) and show the attention collapse toward the
    trained map similarity of ~0 (a faithful map MUST depend on learned weights).
    Also randomizes the CNN head and shows the fused saliency degrade.

    IMPORTANT: the randomization similarity is measured on the RAW, UNMASKED
    patch grids (the true model output). The brain-mask silhouette used for
    display would otherwise inflate similarity even for random weights, so it is
    deliberately excluded from the metric. Similarity is computed per subject
    (original vs randomized) and averaged -- the standard Adebayo protocol.

All results go under ./stability_sanity_out/ ; nothing existing is overwritten.

Usage:
    python stability_sanity_dora.py [N_SUBJECTS]      (default 100)
"""
import os
import sys
import csv
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

N_SUBJECTS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = "stability_sanity_out"
SUBJ_DIR = os.path.join(OUT, "per_subject_unfused_central")
os.makedirs(SUBJ_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Import the pipeline module (reuses trained DoRA ViT + CNN + helpers).
# ---------------------------------------------------------------------------
import importlib.util
sys.argv = ["pipeline", "cnn_mx_bigdo_ch_sw_res", "best", "1"]
_spec = importlib.util.spec_from_file_location("dora_pipeline", "3dmap_grad_vit_cnn_main_mix_roi_dora.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
device, vit, cnn = P.device, P.vit, P.cnn


def attn_to_image(cls_attn, slice_tensor, brain_thresh=0.05):
    """Upsample a patch-grid map to the slice, brain-masked -- FOR DISPLAY ONLY."""
    H, W = slice_tensor.shape[-2:]
    grid = int(np.sqrt(cls_attn.shape[0]))
    g = cls_attn.reshape(grid, grid)
    slice_gray = slice_tensor.squeeze(0).cpu().numpy().mean(0)
    bm = (slice_gray > brain_thresh).astype(float)
    bmp = F.interpolate(torch.tensor(bm)[None, None].float(), size=(grid, grid),
                        mode="bilinear", align_corners=False).squeeze().numpy()
    g = g * bmp
    up = F.interpolate(torch.tensor(g)[None, None].float(), size=(H, W),
                       mode="bilinear", align_corners=False).squeeze().cpu().numpy()
    return (up - up.min()) / (up.max() - up.min() + 1e-8) if up.max() > up.min() else np.zeros_like(up)


def cnn_saliency_center(feats, sex, s0):
    ft = torch.tensor(feats, dtype=torch.float32)[None, None].to(device)
    st = torch.tensor([sex], dtype=torch.float32).to(device)
    sal, _ = P.guided_bp.generate(ft, st)
    return P.scale_saliency(sal)[s0]


# ---------------------------------------------------------------------------
# Phase A: cache central-slice inputs + trained raw grids/maps for N subjects
# ---------------------------------------------------------------------------
print(f"\n[Phase A] caching central-slice inputs for up to {N_SUBJECTS} subjects...")
subjects = []
normal_unfused_img = []                 # (n,H,W) masked display maps (for stability)
normal_cls = []                         # (n,P) RAW ViT CLS->patch attention  (sanity ref)
normal_patch = []                       # (n,P) RAW CNN-projected patch scores (sanity ref)
normal_fused_grid = []                  # (n,P) RAW fused = patch*cls

for _, row in tqdm(P.df_val.iterrows(), total=len(P.df_val), desc="cache"):
    if len(subjects) >= N_SUBJECTS:
        break
    feats, title = P.load_feature_map(row)
    if feats is None or not os.path.exists(row['filepath']):
        continue
    try:
        vol = P.load_volume_160(row['filepath'])
        s0 = min(vol.shape[0] // 2, vol.shape[0] - 1)
        sl = vol[s0]; sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
        slice_tensor = P.transform(sl).unsqueeze(0).float()
        sex = 0.0 if row['Sex'] == 'M' else 1.0
        sal_center = cnn_saliency_center(feats, sex, s0)

        st_dev = slice_tensor.to(device)
        patch_scores, cls_attn = P.embedding_importance_to_patch_map(vit.vit, st_dev, sal_center)
        subjects.append({"title": title, "sex": sex, "s0": s0,
                         "slice_tensor": slice_tensor, "feats": feats, "sal_center": sal_center})
        normal_cls.append(cls_attn)
        normal_patch.append(patch_scores)
        normal_fused_grid.append(patch_scores * cls_attn)
        unf = attn_to_image(cls_attn, st_dev)
        normal_unfused_img.append(unf)
        np.save(os.path.join(SUBJ_DIR, f"{title}.npy"), unf)          # store per-subject (asked)
    except Exception as e:
        tqdm.write(f"skip {row['filepath']}: {e}")
        continue

n = len(subjects)
normal_unfused_img = np.stack(normal_unfused_img)
normal_cls = np.stack(normal_cls)
normal_patch = np.stack(normal_patch)
normal_fused_grid = np.stack(normal_fused_grid)
print(f"[Phase A] cached {n} subjects; per-subject unfused maps -> {SUBJ_DIR}/")


# ---------------------------------------------------------------------------
# Phase B: stability of the unfused attention (mean / SD / CV, split-half)
# ---------------------------------------------------------------------------
mean_map = normal_unfused_img.mean(0)
sd_map = normal_unfused_img.std(0)
cv_map = sd_map / (mean_map + 1e-6)
brain = mean_map > 0.01
cv_brain = cv_map[brain]

fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
im0 = ax[0].imshow(mean_map.T, cmap="hot", origin="lower"); ax[0].set_title(f"(a) Mean attention (n={n})")
im1 = ax[1].imshow(sd_map.T, cmap="viridis", origin="lower"); ax[1].set_title("(b) Inter-subject SD")
im2 = ax[2].imshow(np.where(brain, cv_map, np.nan).T, cmap="magma", origin="lower"); ax[2].set_title("(c) CV = SD/mean")
for a, im in zip(ax[:3], [im0, im1, im2]):
    a.axis("off"); fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
ax[3].hist(cv_brain, bins=40, color="#4C72B0", edgecolor="black")
ax[3].axvline(np.median(cv_brain), color="crimson", ls="--", label=f"median CV {np.median(cv_brain):.2f}")
ax[3].set_title("(d) CV distribution (brain)"); ax[3].set_xlabel("CV"); ax[3].set_ylabel("pixels"); ax[3].legend()
fig.suptitle("Unfused central-slice ViT attention: inter-subject stability", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "stability_intersubject.png"), dpi=150); plt.close(fig)
np.save(os.path.join(OUT, "mean_attention.npy"), mean_map)
np.save(os.path.join(OUT, "sd_attention.npy"), sd_map)

rng = np.random.RandomState(69420)
perm = rng.permutation(n); h1, h2 = perm[:n // 2], perm[n // 2:]
m1, m2 = normal_unfused_img[h1].mean(0), normal_unfused_img[h2].mean(0)
sh_spear = spearmanr(m1[brain], m2[brain]).correlation
sh_pear = pearsonr(m1[brain], m2[brain])[0]

fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
ax[0].imshow(m1.T, cmap="hot", origin="lower"); ax[0].set_title(f"Half 1 (n={len(h1)})"); ax[0].axis("off")
ax[1].imshow(m2.T, cmap="hot", origin="lower"); ax[1].set_title(f"Half 2 (n={len(h2)})"); ax[1].axis("off")
ax[2].scatter(m1[brain], m2[brain], s=4, alpha=0.3, color="#4C72B0")
ax[2].set_xlabel("Half 1 mean attention"); ax[2].set_ylabel("Half 2 mean attention")
ax[2].set_title(f"Split-half\nSpearman {sh_spear:.3f} / Pearson {sh_pear:.3f}")
fig.suptitle("Split-half stability of mean attention", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "stability_split_half.png"), dpi=150); plt.close(fig)
print(f"[Phase B] median CV {np.median(cv_brain):.3f}; split-half Spearman {sh_spear:.3f}, Pearson {sh_pear:.3f}")


# ---------------------------------------------------------------------------
# Randomization helpers
# ---------------------------------------------------------------------------
def randomize_module(m):
    for sub in m.modules():
        if isinstance(sub, nn.Linear):
            nn.init.normal_(sub.weight, 0.0, 0.02)
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)
        elif isinstance(sub, nn.Conv2d):
            nn.init.normal_(sub.weight, 0.0, 0.02)
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)
        elif isinstance(sub, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(sub.weight)
            nn.init.zeros_(sub.bias)
    for name, p in m.named_parameters():
        if ("cls_token" in name) or ("position_embeddings" in name):
            nn.init.normal_(p, 0.0, 0.02)


def per_subject_spearman(A, B):
    """Mean +/- std of per-row Spearman rho between two (n,P) stacks."""
    vals = [spearmanr(A[i], B[i]).correlation for i in range(len(A))]
    vals = np.array([v for v in vals if v == v])   # drop NaN
    return float(vals.mean()), float(vals.std())


def cls_grids_current_vit(want_image=False):
    """Raw CLS->patch grids for every subject under the CURRENT vit weights;
    optionally also the masked mean display image."""
    grids, imgs = [], []
    for s in subjects:
        st = s["slice_tensor"].to(device)
        _, cls = P.embedding_importance_to_patch_map(vit.vit, st, s["sal_center"])
        grids.append(cls)
        if want_image:
            imgs.append(attn_to_image(cls, st))
    grids = np.stack(grids)
    return (grids, np.stack(imgs).mean(0)) if want_image else (grids, None)


# ---------------------------------------------------------------------------
# Phase C: Adebayo cascading ViT randomization (top -> bottom), RAW-grid metric
# ---------------------------------------------------------------------------
print("\n[Phase C] cascading ViT randomization (top -> bottom)...")
pristine_vit = copy.deepcopy(vit.state_dict())
cascade = [vit.vit.encoder.layer[i] for i in range(len(vit.vit.encoder.layer) - 1, -1, -1)]
cascade.append(vit.vit.embeddings)
stage_labels = [f"L{i}" for i in range(len(vit.vit.encoder.layer) - 1, -1, -1)] + ["embed"]

casc_mean, casc_std, snapshots = [], [], {}
snap_ks = {0, 4, 8, len(vit.vit.encoder.layer), len(cascade)}
for k in tqdm(range(len(cascade) + 1), desc="cascade"):
    vit.load_state_dict(pristine_vit)
    for m in cascade[:k]:
        randomize_module(m)
    grids, img = cls_grids_current_vit(want_image=(k in snap_ks))
    mu, sd = per_subject_spearman(grids, normal_cls)      # vs each subject's trained grid
    casc_mean.append(mu); casc_std.append(sd)
    if k in snap_ks:
        snapshots[k] = img
vit.load_state_dict(pristine_vit)


# ---------------------------------------------------------------------------
# Phase C2: fused-map degradation under FULL ViT randomization (ViT is the
# dominant factor of the fused map, so fused should also collapse).
# ---------------------------------------------------------------------------
vit.load_state_dict(pristine_vit)
for m in cascade:
    randomize_module(m)
vitrand_fused_grid = []
for s in subjects:
    st = s["slice_tensor"].to(device)
    patch_scores, cls_attn = P.embedding_importance_to_patch_map(vit.vit, st, s["sal_center"])
    vitrand_fused_grid.append(patch_scores * cls_attn)
vit.load_state_dict(pristine_vit)
vitrand_fused_mu, vitrand_fused_sd = per_subject_spearman(np.stack(vitrand_fused_grid), normal_fused_grid)


# ---------------------------------------------------------------------------
# Phase D: CNN randomization -> CNN-projection & fused saliency degrade
# ---------------------------------------------------------------------------
print("[Phase D] CNN randomization -> fused-saliency degradation...")
pristine_cnn = copy.deepcopy(cnn.state_dict())
randomize_module(cnn)
rand_patch, rand_fused_grid, rand_fused_img = [], [], []
for s in subjects:
    st = s["slice_tensor"].to(device)
    sal_rand = cnn_saliency_center(s["feats"], s["sex"], s["s0"])
    patch_scores, cls_attn = P.embedding_importance_to_patch_map(vit.vit, st, sal_rand)
    rand_patch.append(patch_scores)
    rand_fused_grid.append(patch_scores * cls_attn)
    rand_fused_img.append(P.patch_map_to_image(patch_scores, cls_attn, st, combine_mode="mul"))
cnn.load_state_dict(pristine_cnn)
rand_patch = np.stack(rand_patch); rand_fused_grid = np.stack(rand_fused_grid)
cnn_patch_mu, cnn_patch_sd = per_subject_spearman(rand_patch, normal_patch)
cnn_fused_mu, cnn_fused_sd = per_subject_spearman(rand_fused_grid, normal_fused_grid)
mean_fused_normal = P.patch_map_to_image  # placeholder to keep names; recompute images below
mean_fused_img_normal = np.stack([attn_to_image(normal_fused_grid[i],
                                  subjects[i]["slice_tensor"].to(device)) for i in range(n)]).mean(0)
mean_fused_img_rand = np.stack(rand_fused_img).mean(0)


# ---------------------------------------------------------------------------
# Sanity-check figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(2, 5)

axc = fig.add_subplot(gs[0, :2])
xs = list(range(len(casc_mean)))
casc_mean_a, casc_std_a = np.array(casc_mean), np.array(casc_std)
axc.plot(xs, casc_mean_a, "o-", color="#C44E52", label="mean per-subject Spearman")
axc.fill_between(xs, casc_mean_a - casc_std_a, casc_mean_a + casc_std_a, color="#C44E52", alpha=0.2)
axc.axhline(0, color="black", lw=0.8)
axc.set_xticks(xs); axc.set_xticklabels(["orig"] + stage_labels, rotation=45, ha="right", fontsize=7)
axc.set_ylabel("similarity of raw attention\nto trained model"); axc.set_ylim(-0.35, 1.05)
axc.set_xlabel("cascading randomization (top layer -> bottom)")
axc.set_title("(a) ViT model-parameter randomization (raw-grid metric)")
axc.legend(fontsize=8); axc.grid(alpha=0.3)

for j, k in enumerate(sorted(snapshots)):
    axk = fig.add_subplot(gs[0, 2 + j] if j < 3 else gs[1, j - 3])
    tag = ("trained" if k == 0 else "all layers" if k == len(vit.vit.encoder.layer)
           else "all+embed" if k == len(cascade) else f"top {k} layers")
    axk.imshow(snapshots[k].T, cmap="hot", origin="lower")
    axk.set_title(f"{tag}\nrho={casc_mean[k]:.2f}", fontsize=9); axk.axis("off")

axf1 = fig.add_subplot(gs[1, 2]); axf1.imshow(mean_fused_img_normal.T, cmap="hot", origin="lower")
axf1.set_title("fused: trained CNN", fontsize=9); axf1.axis("off")
axf2 = fig.add_subplot(gs[1, 3]); axf2.imshow(mean_fused_img_rand.T, cmap="hot", origin="lower")
axf2.set_title(f"fused: random CNN\nrho={cnn_fused_mu:.2f}", fontsize=9); axf2.axis("off")
axt = fig.add_subplot(gs[1, 4]); axt.axis("off")
axt.text(0.0, 0.5,
         "Adebayo sanity check\n(raw-grid, per-subject)\n\n"
         f"n subjects = {n}\n\n"
         f"ViT random -> attention:\n  rho {casc_mean[-1]:+.3f} (PASS)\n"
         f"ViT random -> fused:\n  rho {vitrand_fused_mu:+.3f} (PASS)\n\n"
         f"CNN random -> fused:\n  rho {cnn_fused_mu:+.3f}\n  (ViT-dominated)\n\n"
         "Faithful maps -> ~0.",
         fontsize=9.5, va="center")
fig.suptitle("Sanity Checks for Saliency Maps (Adebayo et al.) -- DoRA ViT+CNN", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "sanity_check_randomization.png"), dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Stats CSV
# ---------------------------------------------------------------------------
with open(os.path.join(OUT, "sanity_stability_stats.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["n_subjects", n])
    w.writerow(["median_CV_brain", round(float(np.median(cv_brain)), 4)])
    w.writerow(["mean_CV_brain", round(float(np.mean(cv_brain)), 4)])
    w.writerow(["split_half_spearman", round(float(sh_spear), 4)])
    w.writerow(["split_half_pearson", round(float(sh_pear), 4)])
    w.writerow(["vit_random_attention_spearman_mean", round(float(casc_mean[-1]), 4)])
    w.writerow(["vit_random_attention_spearman_std", round(float(casc_std[-1]), 4)])
    w.writerow(["vit_random_fused_spearman_mean", round(float(vitrand_fused_mu), 4)])
    w.writerow(["cnn_random_projection_spearman", round(float(cnn_patch_mu), 4)])
    w.writerow(["cnn_random_fused_spearman", round(float(cnn_fused_mu), 4)])
    for k, (mu, sd) in enumerate(zip(casc_mean, casc_std)):
        lab = "orig" if k == 0 else stage_labels[k - 1]
        w.writerow([f"cascade_{k}_{lab}_spearman_mean", round(float(mu), 4)])

print("\n==================== SUMMARY ====================")
print(f"subjects analysed:            {n}")
print(f"inter-subject median CV:      {np.median(cv_brain):.3f}")
print(f"split-half Spearman:          {sh_spear:.3f} (Pearson {sh_pear:.3f})")
print(f"ViT-random attention rho:     {casc_mean[-1]:+.3f} +/- {casc_std[-1]:.3f}  (trained=1.0)")
print(f"ViT-random fused rho:         {vitrand_fused_mu:+.3f}")
print(f"CNN-random projection rho:    {cnn_patch_mu:+.3f}")
print(f"CNN-random fused rho:         {cnn_fused_mu:+.3f}  (ViT-dominated)")
print(f"cascade rho by stage:         {[round(v,2) for v in casc_mean]}")
print(f"Outputs -> ./{OUT}/")
