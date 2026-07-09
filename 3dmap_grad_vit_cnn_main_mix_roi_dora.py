"""
3dmap_grad_vit_cnn_main_mix_roi_dora.py
=======================================
Corrected interpretability pipeline (DoRA edition).

What was wrong before (3dmap_grad_vit_cnn_main_mix_roi.py)
---------------------------------------------------------
The CNN guided-backprop saliency was generated from RANDOM NOISE
(`features = torch.randn(1, 1, 160, 768)`), producing a single meaningless
(160,768) saliency map that was then reused for every subject and every slice.

What this file fixes
--------------------
1. ADNI split matches vit_dora_train_feature_cnn_main_mix_roi.py: the ADNI
   metadata is de-duplicated to one (middle-age) row per SubjectID BEFORE the
   sort/head(900), so the concatenated `df` and the seed-69420 random_split are
   identical to the DoRA training run.
2. Attention maps come from the DoRA-adapted ViT: the ViT is rebuilt with the
   same LoRA/DoRA config, the DoRA checkpoint is loaded, and adapters are merged
   (merge_and_unload) so `vit.vit(...)` uses the fine-tuned weights.
3. Per subject, the ACTUAL saved feature map is loaded from that dataset's
   `*_features/train_dora_e5_s32/` folder, the CNN guided-backprop saliency is
   computed FROM IT (and saved as an image), and that subject's saliency is
   fused with that same subject's ViT patch attention -- slice by slice. Maps
   are then averaged across subjects into a 3D attention volume.

Usage
-----
    python 3dmap_grad_vit_cnn_main_mix_roi_dora.py <cnn_module> <load_saved> [max_subjects]
        cnn_module    e.g. cnn_mx_bigdo_ch_sw_res  (must expose AgePredictionCNN)
        load_saved    'best' or 'last'  (which CNN checkpoint under model_dumps/mix/)
        max_subjects  optional, default 50  (val subjects averaged per slice)
"""
import os
import sys
import pickle
import importlib
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from scipy.ndimage import zoom, label, find_objects
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
from transformers import ViTFeatureExtractor, ViTForImageClassification
from peft import LoraConfig, get_peft_model


def set_random_seed(seed=69420):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


universal_seed = 69420
set_random_seed(universal_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ViT feature-extraction config (must match vit_dora_train_feature...)
vit_train_epochs = 5
slice_count = 32
roi = 160
FEATDIR = f"train_dora_e{vit_train_epochs}_s{slice_count}"


# ---------------------------------------------------------------------------
# Average CNN saliency heatmap, plotted EXACTLY like guided_backprop.py
# (jet, rotated, colorbar + title + axis captions). Built by averaging the
# per-subject maps already saved under <sal_dir>/*.npy -- no pipeline rerun.
# ---------------------------------------------------------------------------
def plot_avg_saliency_gbp(sal_dir="maps_out_dora/cnn_saliency",
                          out_path="maps_out_dora/cnn_saliency_average_gbp.png"):
    import glob
    files = sorted(glob.glob(os.path.join(sal_dir, "*.npy")))
    if not files:
        print(f"No .npy saliency maps found in {sal_dir}")
        return
    avg = np.mean(np.stack([np.load(f) for f in files], axis=0), axis=0)  # (160, 768)

    plt.figure(figsize=(6, 12))
    plt.imshow(avg.T, cmap="jet", aspect="auto", origin="lower")
    plt.colorbar(label="Guided Backprop Gradient")
    plt.title("Guided Backpropagation Saliency Map (rotated)")
    plt.xlabel("Sagittal slice index (160)")
    plt.ylabel("Embedding dimension (768)")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved GBP-style average saliency -> {out_path}  (averaged {len(files)} maps)")


# Standalone mode: regenerate the styled average heatmap from existing maps
# without loading any model or rerunning the interpretability pipeline.
#   python 3dmap_grad_vit_cnn_main_mix_roi_dora.py plot_avg [sal_dir] [out_path]
if len(sys.argv) > 1 and sys.argv[1] == "plot_avg":
    _sal_dir = sys.argv[2] if len(sys.argv) > 2 else "maps_out_dora/cnn_saliency"
    _out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        os.path.dirname(_sal_dir.rstrip("/")), "cnn_saliency_average_gbp.png")
    plot_avg_saliency_gbp(_sal_dir, _out)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Preprocessing helpers (identical to the training/interpretability pipeline)
# ---------------------------------------------------------------------------
def resample_nifti(img_data, target_slices=160):
    zoom_factor = target_slices / img_data.shape[0]
    return zoom(img_data, (zoom_factor, 1, 1), order=3)


def calculate_bounding_box_from_volume(volume, intensity_threshold=0.1):
    volume_normalized = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    binary_mask = volume_normalized > intensity_threshold
    labeled_array, _ = label(binary_mask)
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0
    largest_component = np.argmax(component_sizes)
    slices = find_objects((labeled_array == largest_component).astype(int))[0]
    min_indices = [s.start for s in slices]
    max_indices = [s.stop - 1 for s in slices]
    return min_indices, max_indices


def crop_brain_volumes(brain_data):
    mn, mx = calculate_bounding_box_from_volume(brain_data)
    return brain_data[mn[0]:mx[0] + 1, mn[1]:mx[1] + 1, mn[2]:mx[2] + 1]


# ===========================================================================
# 1. Build df with the ADNI split fix (dedup one middle-age row per SubjectID)
# ===========================================================================
df_adni = pd.read_csv("adni_storage/adni_brainrotnet_metadata.csv")
df_adni['filepath'] = df_adni.apply(
    lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1)
# --- FIX: one row per SubjectID (the one whose Age is closest to that
#          subject's median Age), exactly as in the DoRA training file. ---
df_adni = df_adni.loc[
    df_adni.groupby('SubjectID')['Age'].apply(lambda x: (x - x.median()).abs().idxmin())
].reset_index(drop=True)
df_adni = df_adni.sort_values(by='Age', ascending=True).reset_index(drop=True).head(900)

df_ixi = pd.read_csv("ixi_storage/ixi_brainrotnet_metadata.csv")
df_ixi['filepath'] = df_ixi.apply(
    lambda row: f"ixi_storage/IXI_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz", axis=1)

df_abide = pd.read_csv("abide_storage/abide_brainrotnet_metadata.csv")
df_abide['filepath'] = df_abide.apply(
    lambda row: f"abide_storage/ABIDEII_bias_corrected/{row['ImageID'][7:]}.stripped.N4.nii.gz", axis=1)
df_abide = df_abide.sort_values(by='Age', ascending=False).reset_index(drop=True).head(750)

df_dlbs = pd.read_csv("dlbs_storage/dlbs_brainrotnet_metadata.csv")
df_dlbs['filepath'] = df_dlbs.apply(
    lambda row: f"dlbs_storage/DLBS_bias_corrected/{row['ImageID'][4:]}.stripped.N4.nii.gz", axis=1)

df_cobre = pd.read_csv("cobre_storage/cobre_brainrotnet_metadata.csv")
df_cobre['filepath'] = df_cobre.apply(
    lambda row: f"cobre_storage/COBRE_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz", axis=1)

df_fcon = pd.read_csv("fcon1000_storage/fcon1000_brainrotnet_metadata.csv")
df_fcon['filepath'] = df_fcon.apply(
    lambda row: f"fcon1000_storage/fcon1000_bias_corrected/{row['ImageID'][8:]}.stripped.N4.nii.gz", axis=1)
df_fcon = df_fcon.dropna()

df_corr = pd.read_csv("corr_storage/corr_brainrotnet_metadata.csv")
df_corr['filepath'] = df_corr.apply(
    lambda row: f"corr_storage/CORR_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz", axis=1)
df_corr = df_corr.sort_values(by='Age', ascending=True).reset_index(drop=True)

df_oas1 = pd.read_csv("oasis1_storage/oasis1_brainrotnet_metadata.csv")
df_oas1['filepath'] = df_oas1.apply(
    lambda row: f"oasis1_storage/oasis_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz", axis=1)
df_oas1 = df_oas1.sort_values(by='Age', ascending=False).reset_index(drop=True).head(300)

df_camcan = pd.read_csv("camcan_storage/camcan_brainrotnet_metadata.csv")
df_camcan['filepath'] = df_camcan.apply(
    lambda row: f"camcan_storage/CamCAN_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz", axis=1)

df_nimh = pd.read_csv("nimh_storage/nimh_mprage_brainrotnet_metadata.csv")
df_nimh['filepath'] = df_nimh.apply(
    lambda row: f"nimh_storage/nimh_bias_corrected/{row['ImageID'][5:]}_ses-01_acq-MPRAGE_rec-SCIC_T1w.stripped.N4.nii.gz",
    axis=1)

df_bold = pd.read_csv("bold_storage/bold_brainrotnet_metadata.csv")
df_bold['filepath'] = df_bold.apply(
    lambda row: f"bold_storage/bold_bias_corrected/{row['ImageID'][5:]}_T1w.stripped.N4.nii.gz", axis=1)

# Same datasets / order as vit_dora (SALD excluded).
df = pd.concat([
    df_adni[['ImageID', 'Sex', 'Age', 'filepath']],
    df_ixi[['ImageID', 'Sex', 'Age', 'filepath']],
    df_abide[['ImageID', 'Sex', 'Age', 'filepath']],
    df_dlbs[['ImageID', 'Sex', 'Age', 'filepath']],
    df_cobre[['ImageID', 'Sex', 'Age', 'filepath']],
    df_fcon[['ImageID', 'Sex', 'Age', 'filepath']],
    df_corr[['ImageID', 'Sex', 'Age', 'filepath']],
    df_oas1[['ImageID', 'Sex', 'Age', 'filepath']],
    df_camcan[['ImageID', 'Sex', 'Age', 'filepath']],
    df_nimh[['ImageID', 'Sex', 'Age', 'filepath']],
    df_bold[['ImageID', 'Sex', 'Age', 'filepath']],
], ignore_index=True)

df['Age_Group'] = df['Age'].astype(int).apply(lambda x: f"{x:03d}"[:-1] + "0") + df['Sex']
num_classes = df['Age_Group'].nunique()

# Identical split (torch random_split, generator seed 69420, 80/20).
n = len(df)
train_size = int(0.8 * n)
generator = torch.Generator().manual_seed(universal_seed)
_train_ds, _val_ds = torch.utils.data.random_split(range(n), [train_size, n - train_size], generator=generator)
val_indices = list(_val_ds.indices)
df_val = df.iloc[val_indices].reset_index(drop=True)
print(f"Total {n}  ->  train {train_size} / val {len(df_val)}  (num_classes={num_classes})")

# Transform matching ViT feature extraction (ViTFeatureExtractor mean/std).
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])


# ===========================================================================
# 2. Load CNN (trained head) + DoRA-adapted ViT
# ===========================================================================
cnn_module_name = sys.argv[1] if len(sys.argv) > 1 else "cnn_mx_bigdo_ch_sw_res"
load_saved = sys.argv[2] if len(sys.argv) > 2 else "best"
max_subjects = int(sys.argv[3]) if len(sys.argv) > 3 else 50

AgePredictionCNN = getattr(importlib.import_module(cnn_module_name), "AgePredictionCNN")
input_shape = (1, roi, 768)
cnn = AgePredictionCNN(input_shape=input_shape).to(device)
cnn_optimizer = optim.Adam(cnn.parameters(), lr=0.0005)
if load_saved != "none":
    with open(f"model_dumps/mix/{cnn_module_name}_{load_saved}_model_with_metadata.pkl", "rb") as f:
        ck = pickle.load(f)
    cnn.load_state_dict(ck["model_state"])
    cnn_optimizer.load_state_dict(ck["optimizer_state"])
    print(f"Loaded CNN '{cnn_module_name}' ({load_saved}) epoch {ck['epoch']+1}, val loss {ck['loss']:.4f}")
cnn.eval()

# --- DoRA ViT: rebuild config, load DoRA checkpoint, merge adapters ---
vit = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True,
    attn_implementation="eager",       # required for output_attentions
)
dora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
    target_modules=["query", "value"], modules_to_save=["classifier"], use_dora=True,
)
vit = get_peft_model(vit, dora_config)
_dora_ckpt = torch.load("model_dumps/vit_dora_train_checkpoint.pth", map_location="cpu")
vit.load_state_dict(_dora_ckpt["model_state_dict"])
vit = vit.merge_and_unload()            # fold DoRA into base weights
vit.to(device).eval()
print("Loaded DoRA ViT and merged adapters.")


# ===========================================================================
# CNN guided-backprop (now driven by REAL feature maps, not noise)
# ===========================================================================
class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.forward_relu_outputs = []
        self.hooks = []

        def relu_forward_hook(module, inp, out):
            self.forward_relu_outputs.append(out)

        def relu_backward_hook(module, grad_input, grad_output):
            forward_output = self.forward_relu_outputs.pop()
            forward_mask = (forward_output > 0).float()
            return (grad_output[0].clamp(min=0.0) * forward_mask,)

        for module in self.model.modules():
            if isinstance(module, (nn.ReLU, nn.SiLU)):
                self.hooks.append(module.register_forward_hook(relu_forward_hook))
                self.hooks.append(module.register_full_backward_hook(relu_backward_hook))

    def generate(self, input_tensor, sex_tensor):
        self.forward_relu_outputs.clear()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        pred = self.model(input_tensor, sex_tensor).squeeze()
        self.model.zero_grad()
        pred.backward()
        g = input_tensor.grad.detach().cpu().numpy()[0, 0]           # (roi, 768)
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        return g, float(pred.item())

    def close(self):
        for h in self.hooks:
            h.remove()


guided_bp = GuidedBackprop(cnn)


# ===========================================================================
# 3. Map each val subject to its saved DoRA feature file
# ===========================================================================
def feature_path_for(row):
    """Return the .npy feature path for a val row, using the exact per-dataset
    folder + image_title rules from vit_dora_train_feature_cnn_main_mix_roi.py."""
    fp, iid = row['filepath'], row['ImageID']
    if fp.startswith("adni_storage"):
        # image_title = f"{ImageID[4:]}_{SubjectID}"; recover it from the filename.
        title = os.path.basename(fp).replace(".stripped.N4.nii.gz", "")[1:]
        sub = "adni_storage/ADNI_features"
    elif fp.startswith("ixi_storage"):
        title, sub = iid, "ixi_storage/IXI_features"
    elif fp.startswith("abide_storage"):
        title, sub = iid[7:], "abide_storage/ABIDEII_features"
    elif fp.startswith("dlbs_storage"):
        title, sub = iid[4:], "dlbs_storage/DLBS_features"
    elif fp.startswith("cobre_storage"):
        title, sub = iid[5:], "cobre_storage/COBRE_features"
    elif fp.startswith("fcon1000_storage"):
        title, sub = iid[5:], "fcon1000_storage/fcon1000_features"
    elif fp.startswith("corr_storage"):
        title, sub = iid[5:], "corr_storage/CORR_features"
    elif fp.startswith("oasis1_storage"):
        title, sub = iid, "oasis1_storage/oasis1_features"
    elif fp.startswith("camcan_storage"):
        title, sub = iid, "camcan_storage/CamCAN_features"
    elif fp.startswith("nimh_storage"):
        title, sub = iid, "nimh_storage/nimh_features"
    elif fp.startswith("bold_storage"):
        title, sub = iid, "bold_storage/bold_features"
    else:
        return None, None
    return f"{sub}/{FEATDIR}/{title}_features.npy", title


def load_feature_map(row):
    path, title = feature_path_for(row)
    if path is None or not os.path.exists(path):
        return None, title
    feats = np.load(path)
    feats = feats[len(feats) // 2 - roi // 2: len(feats) // 2 + roi // 2]   # ROI crop
    return feats.astype(np.float32), title


# ===========================================================================
# ViT patch-attention fusion helpers (unchanged logic; G_vec is now per subject)
# ===========================================================================
def embedding_importance_to_patch_map(vit_backbone, slice_tensor, G_vec):
    with torch.no_grad():
        outputs = vit_backbone(slice_tensor, output_attentions=True)
    patch_tokens = outputs.last_hidden_state[0, 1:, :]                       # (P, D)
    G = torch.as_tensor(G_vec, dtype=patch_tokens.dtype, device=patch_tokens.device)
    patch_scores = (patch_tokens @ G).detach().cpu().numpy()                # (P,)
    if patch_scores.max() > patch_scores.min():
        patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min() + 1e-8)
    else:
        patch_scores = np.zeros_like(patch_scores)
    cls_attn = outputs.attentions[-1][0, :, 0, 1:].mean(0).detach().cpu().numpy()
    if cls_attn.max() > cls_attn.min():
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    else:
        cls_attn = np.zeros_like(cls_attn)
    return patch_scores, cls_attn


def patch_map_to_image(patch_scores, cls_attn, slice_tensor, combine_mode="mul", brain_thresh=0.05):
    H, W = slice_tensor.shape[-2:]
    grid = int(np.sqrt(patch_scores.shape[0]))
    if combine_mode == "mul":
        fused = patch_scores * cls_attn
    elif combine_mode == "add":
        fused = patch_scores + cls_attn
    else:
        fused = patch_scores
    fused_grid = fused.reshape(grid, grid)

    slice_gray = slice_tensor.squeeze(0).cpu().numpy().mean(0)
    brain_mask = (slice_gray > brain_thresh).astype(float)
    brain_mask_patch = F.interpolate(torch.tensor(brain_mask)[None, None].float(),
                                     size=(grid, grid), mode="bilinear", align_corners=False).squeeze().numpy()
    fused_grid = fused_grid * brain_mask_patch

    up = F.interpolate(torch.tensor(fused_grid)[None, None].float(),
                       size=(H, W), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
    if up.max() > up.min():
        up = (up - up.min()) / (up.max() - up.min() + 1e-8)
    else:
        up = np.zeros_like(up)
    return up


def load_volume_160(path):
    nii = nib.load(path)
    ornt = ornt_transform(io_orientation(nii.affine), axcodes2ornt(("R", "A", "S")))
    data = apply_orientation(nii.get_fdata(), ornt)
    data = crop_brain_volumes(data)
    return resample_nifti(data, target_slices=160)


# Saliency contrast transform: "log" lifts low/mid importance (good for peaky
# maps), "exp" does the opposite -- it suppresses low/mid values and sharpens
# onto only the strongest peaks, "linear" applies no contrast bending (raw
# min-max normalized saliency). Switch with the SAL_SCALE env var.
SCALE_MODE = os.environ.get("SAL_SCALE", "log")   # "log", "exp", or "linear"


def to_log_scale(x, k=99.0):
    """Log-compress a saliency map, then min-max normalize onto [0,1] for
    clarity. Guided-backprop maps are peaky (a few voxels dominate); a log scale
    lifts low/mid-importance structure. y = log1p(k * x/max) / log1p(k) spans
    ~2 decades (k=99) and stays monotonic; the final min-max stretch guarantees
    the output fills [0,1] regardless of the input's minimum."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0, None)
    m = x.max()
    if m <= 0:
        return x.astype(np.float32)
    y = np.log1p(k * (x / m)) / np.log1p(k)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)     # normalize after log
    return y.astype(np.float32)


def to_exp_scale(x, k=6.0):
    """Exponential (inverse of the log) contrast, then min-max normalize onto
    [0,1]. y = (exp(k * x/max) - 1) / (exp(k) - 1) is convex, so it suppresses
    low/mid-importance structure and emphasizes only the strongest peaks
    (larger k -> sparser, more peak-focused)."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0, None)
    m = x.max()
    if m <= 0:
        return x.astype(np.float32)
    y = np.expm1(k * (x / m)) / np.expm1(k)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)     # normalize after exp
    return y.astype(np.float32)


def to_linear_scale(x):
    """No contrast bending: just min-max normalize the raw saliency onto [0,1]."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0, None)
    y = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return y.astype(np.float32)


def scale_saliency(x):
    """Dispatch to the configured contrast transform (see SCALE_MODE)."""
    if SCALE_MODE == "exp":
        return to_exp_scale(x)
    if SCALE_MODE == "linear":
        return to_linear_scale(x)
    return to_log_scale(x)


# ===========================================================================
# Main build: per-subject real saliency -> per-slice fusion -> 3D volume
# ===========================================================================
def build_interpretation(df_val, out_dir="maps_out_dora", combine_mode="mul", max_subjects=50):
    os.makedirs(out_dir, exist_ok=True)
    sal_dir = os.path.join(out_dir, "cnn_saliency")
    os.makedirs(sal_dir, exist_ok=True)
    per_slice = [[] for _ in range(slice_count if False else 160)]   # 160 slices
    sal_log_sum = np.zeros((roi, 768), dtype=np.float64)             # accumulator for the average

    used = 0
    for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Subjects"):
        if used >= max_subjects:
            break
        feats, title = load_feature_map(row)                 # REAL feature map (160,768)
        if feats is None:
            continue
        if not os.path.exists(row['filepath']):
            continue
        try:
            # --- CNN saliency from the ACTUAL feature map (not noise) ---
            sex_val = 0.0 if row['Sex'] == 'M' else 1.0
            feat_tensor = torch.tensor(feats, dtype=torch.float32)[None, None].to(device)  # (1,1,160,768)
            sex_tensor = torch.tensor([sex_val], dtype=torch.float32).to(device)
            saliency, pred_age = guided_bp.generate(feat_tensor, sex_tensor)               # (160,768)

            # --- contrast transform (log or exp) for clarity; this is what we fuse AND save ---
            saliency_log = scale_saliency(saliency)
            sal_log_sum += saliency_log

            # save the per-subject (transformed) CNN saliency image + array
            plt.imsave(os.path.join(sal_dir, f"{title}.png"), saliency_log.T, cmap="jet", origin="lower")
            np.save(os.path.join(sal_dir, f"{title}.npy"), saliency_log)

            # --- Fuse this subject's transformed saliency with its own ViT attention, slice by slice ---
            vol = load_volume_160(row['filepath'])
            for s in range(min(160, vol.shape[0])):
                sl = vol[s]
                sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
                slice_tensor = transform(sl).unsqueeze(0).to(device).float()
                patch_scores, cls_attn = embedding_importance_to_patch_map(vit.vit, slice_tensor, saliency_log[s])
                heat = patch_map_to_image(patch_scores, cls_attn, slice_tensor, combine_mode=combine_mode)
                per_slice[s].append(heat)
            used += 1
        except Exception as e:
            tqdm.write(f"Skipping {row['filepath']}: {e}")
            continue

    print(f"Fused maps from {used} subjects.")

    # --- Average CNN saliency map across all used subjects ---
    # Mean of the per-subject (already log-scaled) maps, then min-max stretched
    # for display contrast (re-logging would double-compress).
    if used > 0:
        avg_saliency = sal_log_sum / used
        avg_saliency = (avg_saliency - avg_saliency.min()) / (avg_saliency.max() - avg_saliency.min() + 1e-8)
        np.save(os.path.join(out_dir, "cnn_saliency_average.npy"), avg_saliency)
        plt.imsave(os.path.join(out_dir, "cnn_saliency_average.png"),
                   avg_saliency.T, cmap="jet", origin="lower")
        print(f"Saved average saliency map ({used} subjects) -> {out_dir}/cnn_saliency_average.png")
        # Also emit the guided_backprop.py-styled version (colorbar/title/labels).
        plot_avg_saliency_gbp(sal_dir, os.path.join(out_dir, "cnn_saliency_average_gbp.png"))

    # Average across subjects per slice, save each slice, stack into a 3D volume.
    per_slice_images = []
    for s in range(160):
        if len(per_slice[s]) == 0:
            per_slice_images.append(np.zeros((224, 224)))
            continue
        m = np.stack(per_slice[s], axis=0).mean(axis=0)
        if m.max() > m.min():
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        per_slice_images.append(m)
        np.save(os.path.join(out_dir, f"slice_{s:03d}.npy"), m)
        plt.imsave(os.path.join(out_dir, f"slice_{s:03d}.png"), m, cmap="hot")

    attention_volume = np.stack(per_slice_images, axis=0)
    if attention_volume.max() > attention_volume.min():
        attention_volume = (attention_volume - attention_volume.min()) / (attention_volume.max() - attention_volume.min() + 1e-8)
    nib.save(nib.Nifti1Image(attention_volume, np.eye(4)),
             os.path.join(out_dir, "attention_3d_mapped_backprop_dora.nii.gz"))
    print(f"Saved {out_dir}/attention_3d_mapped_backprop_dora.nii.gz")
    return attention_volume


if __name__ == "__main__":
    # separate output dir per contrast mode so log/exp runs don't overwrite
    out_dir = "maps_out_dora" if SCALE_MODE == "log" else f"maps_out_dora_{SCALE_MODE}"
    print(f"Saliency contrast mode: {SCALE_MODE}  ->  {out_dir}")
    build_interpretation(df_val, out_dir=out_dir, combine_mode="mul", max_subjects=max_subjects)
    guided_bp.close()
