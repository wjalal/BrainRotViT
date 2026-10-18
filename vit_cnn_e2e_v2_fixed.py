import os
import sys
import math
import pickle
import importlib
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor
from peft import LoraConfig, get_peft_model
from scipy.ndimage import zoom, label, find_objects
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


def set_random_seed(seed=69420):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


universal_seed = 69420
set_random_seed(universal_seed)


FAST = os.environ.get("FAST", "0") == "1"

NUM_SLICES  = int(os.environ.get("NUM_SLICES", 160))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", 4))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))
GRAD_CKPT   = os.environ.get("GRAD_CKPT", "1") == "1"
AMP_DTYPE   = {"bf16": torch.bfloat16, "fp16": torch.float16,
               "off": None}[os.environ.get("AMP", "off")]
FAST_MATH   = os.environ.get("FAST_MATH", "1" if FAST else "0") == "1"


ZNORM_TARGET      = os.environ.get("ZNORM_TARGET", "1") == "1"   # fix #1
ACCUM_STEPS       = int(os.environ.get("ACCUM_STEPS", 2))        # fix #2
HEAD_LR           = float(os.environ.get("HEAD_LR", 1e-3))       # fix #3
VIT_LR            = float(os.environ.get("VIT_LR", 1e-5))        # fix #3
FREEZE_VIT_EPOCHS = int(os.environ.get("FREEZE_VIT_EPOCHS", 3))  # fix #4
CLIP_NORM         = float(os.environ.get("CLIP_NORM", 1.0))      # fix #5
DROPOUT           = float(os.environ.get("DROPOUT", 0.1))        # fix #6
WARMUP_STEPS      = int(os.environ.get("WARMUP_STEPS", 200))     # fix #7
COLLAPSE_WARN     = float(os.environ.get("COLLAPSE_WARN", 0.05)) # fix #8

LOSS_FN           = os.environ.get("LOSS", "mse").lower()        # mse | huber
RUN_TAG           = os.environ.get("RUN_TAG", "v2")
assert LOSS_FN in ("huber", "mse"), f"LOSS must be huber|mse, got {LOSS_FN}"

if FAST_MATH:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load metadata for every dataset (same sources/filters as the cached pipeline
# in vit_train_feature_cnn_main_mix_roi.py)
# ---------------------------------------------------------------------------
csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
df_adni = pd.read_csv(csv_path)
df_adni['filepath'] = df_adni.apply(
    lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1
)
# Keep a single scan per subject: the one closest to that subject's median age
df_adni = df_adni.loc[
    df_adni.groupby('SubjectID')['Age'].apply(lambda x: (x - x.median()).abs().idxmin())
].reset_index(drop=True)
df_adni = df_adni.sort_values(by='Age', ascending=True).reset_index(drop=True)
df_adni = df_adni.head(900)

metadata_path = "ixi_storage/ixi_brainrotnet_metadata.csv"
df_ixi = pd.read_csv(metadata_path)
df_ixi['filepath'] = df_ixi.apply(
    lambda row: f"ixi_storage/IXI_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "abide_storage/abide_brainrotnet_metadata.csv"
df_abide = pd.read_csv(metadata_path)
df_abide['filepath'] = df_abide.apply(
    lambda row: f"abide_storage/ABIDEII_bias_corrected/{row['ImageID'][7:]}.stripped.N4.nii.gz",
    axis=1
)
df_abide = df_abide.sort_values(by='Age', ascending=False).reset_index(drop=True)
df_abide = df_abide.head(750)

metadata_path = "dlbs_storage/dlbs_brainrotnet_metadata.csv"
df_dlbs = pd.read_csv(metadata_path)
df_dlbs['filepath'] = df_dlbs.apply(
    lambda row: f"dlbs_storage/DLBS_bias_corrected/{row['ImageID'][4:]}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "cobre_storage/cobre_brainrotnet_metadata.csv"
df_cobre = pd.read_csv(metadata_path)
df_cobre['filepath'] = df_cobre.apply(
    lambda row: f"cobre_storage/COBRE_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "fcon1000_storage/fcon1000_brainrotnet_metadata.csv"
df_fcon = pd.read_csv(metadata_path)
df_fcon['filepath'] = df_fcon.apply(
    lambda row: f"fcon1000_storage/fcon1000_bias_corrected/{row['ImageID'][8:]}.stripped.N4.nii.gz",
    axis=1
)
df_fcon = df_fcon.dropna()

metadata_path = "corr_storage/corr_brainrotnet_metadata.csv"
df_corr = pd.read_csv(metadata_path)
df_corr['filepath'] = df_corr.apply(
    lambda row: f"corr_storage/CORR_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz",
    axis=1
)
df_corr = df_corr.sort_values(by='Age', ascending=True).reset_index(drop=True)

metadata_path = "oasis1_storage/oasis1_brainrotnet_metadata.csv"
df_oas1 = pd.read_csv(metadata_path)
df_oas1['filepath'] = df_oas1.apply(
    lambda row: f"oasis1_storage/oasis_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)
df_oas1 = df_oas1.sort_values(by='Age', ascending=False)
df_oas1 = df_oas1.reset_index(drop=True).head(300)

metadata_path = "camcan_storage/camcan_brainrotnet_metadata.csv"
df_camcan = pd.read_csv(metadata_path)
df_camcan['filepath'] = df_camcan.apply(
    lambda row: f"camcan_storage/CamCAN_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "nimh_storage/nimh_mprage_brainrotnet_metadata.csv"
df_nimh = pd.read_csv(metadata_path)
df_nimh['filepath'] = df_nimh.apply(
    lambda row: f"nimh_storage/nimh_bias_corrected/{row['ImageID'][5:]}_ses-01_acq-MPRAGE_rec-SCIC_T1w.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "bold_storage/bold_brainrotnet_metadata.csv"
df_bold = pd.read_csv(metadata_path)
df_bold['filepath'] = df_bold.apply(
    lambda row: f"bold_storage/bold_bias_corrected/{row['ImageID'][5:]}_T1w.stripped.N4.nii.gz",
    axis=1
)

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

# Drop rows whose NIfTI file is missing on disk
missing = ~df['filepath'].apply(os.path.exists)
if missing.any():
    print(f"Skipping {missing.sum()} rows with missing files.")
df = df[~missing].reset_index(drop=True)

# Smoke-test knob: LIMIT_SUBJECTS=200 runs the whole pipeline end to end in a
# few minutes. Stratified by age so the subset still spans the range. 
_LIMIT = int(os.environ.get("LIMIT_SUBJECTS", 0))
if _LIMIT and _LIMIT < len(df):
    _keep = np.linspace(0, len(df) - 1, _LIMIT).astype(int)
    df = df.sort_values("Age").iloc[_keep].reset_index(drop=True)
    print(f"[v2] LIMIT_SUBJECTS={_LIMIT}: smoke-test subset, ages "
          f"{df.Age.min():.1f}-{df.Age.max():.1f}", flush=True)
print(df)

sex_encoded = df['Sex'].apply(lambda x: 0 if x == 'M' else 1).tolist()
age_list = df['Age'].tolist()
filepath_list = df['filepath'].tolist()

num_slices = NUM_SLICES  # every resampled slice is used and kept in the backward graph


def resample_nifti(img_data, target_slices=160):
    current_slices = img_data.shape[0]
    zoom_factor = target_slices / current_slices
    return zoom(img_data, (zoom_factor, 1, 1), order=3)


def calculate_bounding_box_from_volume(volume, intensity_threshold=0.1):
    volume_normalized = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    binary_mask = volume_normalized > intensity_threshold
    labeled_array, _ = label(binary_mask)
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # exclude background
    largest_component = np.argmax(component_sizes)
    brain_mask = labeled_array == largest_component
    slices = find_objects(brain_mask.astype(int))[0]
    min_indices = [s.start for s in slices]
    max_indices = [s.stop - 1 for s in slices]
    return min_indices, max_indices


def crop_brain_volumes(brain_data):
    min_indices, max_indices = calculate_bounding_box_from_volume(brain_data)
    return brain_data[min_indices[0]:max_indices[0] + 1,
                       min_indices[1]:max_indices[1] + 1,
                       min_indices[2]:max_indices[2] + 1]


feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])


def load_volume_slices(filepath, target_slices=num_slices):
    nii_img = nib.load(filepath)
    orig_ornt = io_orientation(nii_img.affine)
    ras_ornt = axcodes2ornt(("R", "A", "S"))
    ornt_trans = ornt_transform(orig_ornt, ras_ornt)
    data = nii_img.get_fdata()
    data = apply_orientation(data, ornt_trans)
    data = crop_brain_volumes(data)
    data = resample_nifti(data, target_slices=target_slices)

    slices = []
    for slice_idx in range(data.shape[0]):
        slice_data = data[slice_idx, :, :]
        slice_min, slice_max = np.min(slice_data), np.max(slice_data)
        denom = slice_max - slice_min
        # Uniform slices (e.g. all-zero background after resampling) have
        # max == min; leave them as zeros instead of dividing by zero -> NaN.
        if denom > 0:
            slice_data = (slice_data - slice_min) / denom
        else:
            slice_data = np.zeros_like(slice_data)
        slices.append(transform(slice_data))
    return torch.stack(slices)  # [target_slices, 3, 224, 224]


class VolumeDataset(Dataset):
    """Lazily loads and preprocesses whole volumes on every access -- no
    slices or ViT features are ever cached to disk."""

    def __init__(self, filepath_list, sex_list, age_list, target_slices=num_slices):
        self.filepath_list = filepath_list
        self.sex_list = sex_list
        self.age_list = age_list
        self.target_slices = target_slices

    def __len__(self):
        return len(self.age_list)

    def __getitem__(self, idx):
        slices = load_volume_slices(self.filepath_list[idx], self.target_slices)
        sex = torch.tensor(self.sex_list[idx], dtype=torch.float32)
        age = torch.tensor(self.age_list[idx], dtype=torch.float32)
        return slices, sex, age


dataset = VolumeDataset(filepath_list, sex_encoded, age_list, target_slices=num_slices)

batch_size = BATCH_SIZE
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(universal_seed)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
train_indices = train_dataset.indices
val_indices = val_dataset.indices

_train_ages = np.array([age_list[i] for i in train_indices], dtype=np.float64)
AGE_MU = float(_train_ages.mean())
AGE_SD = float(_train_ages.std()) if ZNORM_TARGET else 1.0
if not ZNORM_TARGET:
    AGE_MU = 0.0
print(f"[v2] target standardisation: mu={AGE_MU:.3f} sd={AGE_SD:.3f} "
      f"(computed on the {len(_train_ages)} TRAIN subjects only)", flush=True)


_loader_kw = dict(
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=FAST and NUM_WORKERS > 0,
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, **_loader_kw)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **_loader_kw)

train_outputs = {}
val_outputs = {}


module_name = sys.argv[1]
class_name = "AgePredictionCNN"
module = importlib.import_module(module_name)
AgePredictionCNN = getattr(module, class_name)
print(f"Successfully imported {class_name} from {module_name}.")

vit_backbone = ViTModel.from_pretrained("google/vit-base-patch16-224", add_pooling_layer=False)

if GRAD_CKPT:
    vit_backbone.gradient_checkpointing_enable()
    vit_backbone.enable_input_require_grads()


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",

    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
    use_dora=True,
)
vit_backbone = get_peft_model(vit_backbone, lora_config)
vit_backbone.print_trainable_parameters()


class ViTCNNEndToEnd(nn.Module):
    def __init__(self, vit, cnn):
        super().__init__()
        self.vit = vit
        self.cnn = cnn

    def forward(self, slices, sex):
        b, n, c, h, w = slices.shape
        slices = slices.view(b * n, c, h, w)
        if getattr(self, "freeze_vit", False):

            with torch.no_grad():
                embeddings = self.vit(pixel_values=slices).last_hidden_state.mean(dim=1)
            embeddings = embeddings.detach()
        else:
            embeddings = self.vit(pixel_values=slices).last_hidden_state.mean(dim=1)
        embeddings = embeddings.view(b, n, -1).unsqueeze(1)  # [b, 1, n, hidden]
        return self.cnn(embeddings, sex)


cnn_head = AgePredictionCNN((1, num_slices, vit_backbone.config.hidden_size))

# FIX #6 -- v1 stacked two p=0.3 dropouts on a head fed by batch-1 gradients.
for _m in cnn_head.modules():
    if isinstance(_m, nn.Dropout):
        _m.p = DROPOUT

model = ViTCNNEndToEnd(vit_backbone, cnn_head).to(device)


criterion = nn.SmoothL1Loss() if LOSS_FN == "huber" else nn.MSELoss()
eval_crit = nn.L1Loss()          # still MAE in YEARS, comparable to v1's curve


_vit_params = [q for q in model.vit.parameters() if q.requires_grad]
_cnn_params = list(model.cnn.parameters())
optimizer = optim.AdamW([
    {"params": _cnn_params, "lr": HEAD_LR, "name": "head"},
    {"params": _vit_params, "lr": VIT_LR,  "name": "vit"},
])
print(f"[v2] run_tag={RUN_TAG} loss={LOSS_FN} freeze_vit_epochs={FREEZE_VIT_EPOCHS}", flush=True)
print(f"[v2] param groups: head {sum(q.numel() for q in _cnn_params):,} @ {HEAD_LR} | "
      f"vit(DoRA) {sum(q.numel() for q in _vit_params):,} @ {VIT_LR}", flush=True)
best_loss = np.inf
start_epoch = 0

filename = f"{sys.argv[1]}_e2e_{RUN_TAG}"

load_saved = sys.argv[2]  # "none", "last", "best"
if load_saved != "none":
    with open(f"model_dumps/mix/{filename}_best_model_with_metadata.pkl", "rb") as f:
        checkpoint = pickle.load(f)
    best_loss = checkpoint["loss"]

    with open(f"model_dumps/mix/{filename}_{load_saved}_model_with_metadata.pkl", "rb") as f:
        checkpoint = pickle.load(f)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    torch.set_rng_state(checkpoint["t_rng_st"])
    np.random.set_state(checkpoint["n_rng_st"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_st"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_st"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"Loaded model from epoch {start_epoch} with validation loss {checkpoint['loss']:.4f}, best loss {best_loss:.4f}")

epochs = int(sys.argv[3])
csv_file = f"model_dumps/mix/{filename}.csv"

if os.path.exists(csv_file):
    epoch_data = pd.read_csv(csv_file).to_dict(orient="records")
    print(f"Loaded existing epoch data from {csv_file}.")
else:
    epoch_data = []
    print("No existing epoch data found. Starting fresh.")


def update_loss_plot(epoch_data, filename):
    df_loss = pd.DataFrame(epoch_data)
    df_loss.to_csv(f"model_dumps/mix/{filename}.csv", index=False)

    plt.figure(figsize=(8, 6))
    if "train_mae" in df_loss:
        plt.plot(df_loss['epoch'], df_loss['train_mae'], label="Train MAE (years)", marker="o")
    plt.plot(df_loss['epoch'], df_loss['val_loss'], label="Validation MAE (years)", marker="o")
    # Reference lines: what we already know is achievable and what failure is.
    plt.axhline(20.08, ls=":", c="grey", label="constant-mean baseline (20.1y)")
    plt.axhline(6.91, ls="--", c="green", label="ridge on frozen ViT (6.9y)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (years)")
    plt.title("Age prediction error vs. epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"model_dumps/mix/{filename}.png")
    plt.close()


import contextlib

def amp_ctx():

    if AMP_DTYPE is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)


_amp_name = "off" if AMP_DTYPE is None else str(AMP_DTYPE).replace("torch.", "")
print(f"[config] {'FAST (tuned)' if FAST else 'FAITHFUL (matches original)'}: "
      f"slices={num_slices} batch={batch_size} workers={NUM_WORKERS} "
      f"amp={_amp_name} grad_ckpt={GRAD_CKPT} fast_math={FAST_MATH} "
      f"persistent_workers={FAST and NUM_WORKERS > 0}", flush=True)


_steps_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)
_ck = checkpoint if load_saved != "none" else {}
_env_sched = os.environ.get("SCHED_EPOCHS")
SCHED_EPOCHS = int(_env_sched) if _env_sched else int(_ck.get("sched_epochs") or epochs)
_total_steps = max(1, SCHED_EPOCHS * _steps_per_epoch)
_global_step = int(_ck.get("sched_step", start_epoch * _steps_per_epoch))
def _lr_lambda(step):
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    prog = (step - WARMUP_STEPS) / max(1, _total_steps - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

for _g in optimizer.param_groups:
    _g["initial_lr"] = HEAD_LR if _g.get("name") == "head" else VIT_LR
scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=_global_step - 1)
print(f"[v2] lr schedule: horizon {SCHED_EPOCHS} epochs ({_total_steps} steps), "
      f"resuming at step {_global_step} -> head lr "
      f"{optimizer.param_groups[0]['lr']:.4e}, vit lr {optimizer.param_groups[1]['lr']:.4e}"
      + ("" if _env_sched or _ck.get("sched_epochs") or start_epoch == 0 else
         "  *** CAUTION (lr schedule): resumed checkpoint has no saved horizon and SCHED_EPOCHS is "
         "unset; using the epochs argument, which is only right if it matches the "
         "original launch ***"), flush=True)


@torch.no_grad()
def embedding_health(n_batches=4):

    was_training = model.training
    model.eval()
    chunks = []
    for i, (sl, _sx, _ag) in enumerate(val_loader):
        if i >= n_batches:
            break
        sl = sl.to(device, non_blocking=True)
        b, n, c, h, w = sl.shape
        with amp_ctx():
            e = model.vit(pixel_values=sl.view(b * n, c, h, w)).last_hidden_state.mean(dim=1)
        chunks.append(e.float().view(b, n, -1).mean(dim=1).cpu())
    if was_training:
        model.train()
    if not chunks:
        return float("nan")
    X = torch.cat(chunks).numpy()
    return float(X.std(0).mean() / (np.abs(X).mean() + 1e-12))


_health0 = embedding_health()
print(f"[v2] embedding health BEFORE training: {_health0:.5f} "
      f"(frozen reference 0.283, collapsed v1 0.0016)", flush=True)

for epoch in range(start_epoch, epochs):

    _vit_on = epoch >= FREEZE_VIT_EPOCHS
    for q in _vit_params:
        q.requires_grad_(_vit_on)
    model.freeze_vit = not _vit_on
    print(f"[v2] epoch {epoch+1}: DoRA adapters {'TRAINING' if _vit_on else 'FROZEN'}", flush=True)

    model.train()
    train_loss = 0.0
    train_abs_err = 0.0
    n_train_seen = 0

    for idx, (slices, sex, age) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]")):
        slices = slices.to(device, non_blocking=True)
        sex, age = sex.to(device, non_blocking=True), age.to(device, non_blocking=True)

        with amp_ctx():
            outputs = model(slices, sex)
        outputs = outputs.float()          # back to fp32 for loss + bookkeeping

        # The network predicts a STANDARDISED age; convert back to years for
        # every human-facing number so the curves stay comparable with v1.
        pred_years = outputs.squeeze(-1) * AGE_SD + AGE_MU
        for i in range(pred_years.size(0)):
            train_outputs[train_indices[idx * batch_size + i]] = pred_years[i].item()

        target_z = (age - AGE_MU) / AGE_SD
        loss = criterion(outputs.squeeze(-1), target_z)

        (loss / ACCUM_STEPS).backward()
        if (idx + 1) % ACCUM_STEPS == 0 or (idx + 1) == len(train_loader):
            if CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    [q for g in optimizer.param_groups for q in g["params"]], CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        train_loss += loss.item()
        train_abs_err += (pred_years.detach() - age).abs().sum().item()
        n_train_seen += pred_years.size(0)

    train_loss /= len(train_loader)
    train_mae = train_abs_err / max(1, n_train_seen)
    print(f"Epoch {epoch+1}/{epochs}, Train Huber(z): {train_loss:.4f}, Train MAE: {train_mae:.3f}y")

    model.eval()
    val_loss = 0.0
    _val_pred_years = []
    with torch.no_grad():
        for idx, (slices, sex, age) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]")):
            slices = slices.to(device, non_blocking=True)
            sex, age = sex.to(device, non_blocking=True), age.to(device, non_blocking=True)
            with amp_ctx():
                outputs = model(slices, sex)
            outputs = outputs.float()
            pred_years = outputs.squeeze(-1) * AGE_SD + AGE_MU
            loss = eval_crit(pred_years, age)      # MAE in years, as in v1
            val_loss += loss.item()

            _val_pred_years.append(pred_years.cpu())
            for i in range(pred_years.size(0)):
                val_outputs[val_indices[idx * batch_size + i]] = pred_years[i].item()

    _pred_sd = float(torch.cat(_val_pred_years).std()) if _val_pred_years else float("nan")
    _health = embedding_health()
    print(f"[v2] val prediction spread: {_pred_sd:.3f}y (actual age sd ~22.3) | "
          f"embedding health: {_health:.5f}", flush=True)
    if _health < COLLAPSE_WARN:
        print(f"[v2] *** WARNING: embedding health {_health:.5f} < {COLLAPSE_WARN} -- "
              f"the backbone is collapsing, as v1 did. Lower VIT_LR or raise "
              f"FREEZE_VIT_EPOCHS. ***", flush=True)

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")

    print("Saving last model...")
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "loss": val_loss,
        "t_rng_st": torch.get_rng_state(),
        "n_rng_st": np.random.get_state(),
        "cuda_rng_st": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        # schedule position, so the next resume continues it exactly
        "sched_epochs": SCHED_EPOCHS,
        "sched_step": scheduler.last_epoch,
    }
    with open(f"model_dumps/mix/{filename}_last_model_with_metadata.pkl", "wb") as f:
        pickle.dump(checkpoint, f)

    if val_loss < best_loss:
        best_loss = val_loss
        print(f"Validation loss improved to {best_loss:.4f}. Saving model...")
        with open(f"model_dumps/mix/{filename}_best_model_with_metadata.pkl", "wb") as f:
            pickle.dump(checkpoint, f)

    epoch_data.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                       "train_mae": train_mae, "pred_sd": _pred_sd,
                       "embedding_health": _health, "vit_training": int(_vit_on)})
    update_loss_plot(epoch_data, filename)

    max_index = max(train_outputs.keys())
    df_trn = pd.DataFrame(index=range(max_index + 1), columns=["Predicted_Age"])
    for index, value in train_outputs.items():
        df_trn.loc[index, "Predicted_Age"] = value
    df2 = df.copy()
    df2['Predicted_Age'] = df_trn['Predicted_Age']
    df2.loc[train_outputs.keys()].to_csv(f"model_dumps/mix/{filename}_predicted_ages_train.csv")

    max_index = max(val_outputs.keys())
    df_pred = pd.DataFrame(index=range(max_index + 1), columns=["Predicted_Age"])
    for index, value in val_outputs.items():
        df_pred.loc[index, "Predicted_Age"] = value
    df1 = df.copy()
    df1['Predicted_Age'] = df_pred['Predicted_Age']
    test_df = df1.loc[val_outputs.keys()]
    test_df.to_csv(f"model_dumps/mix/{filename}_predicted_ages_val.csv")

    unique_groups = test_df['ImageID'].str[:3].unique()
    group_to_color = {group: i for i, group in enumerate(unique_groups)}
    cmap = plt.get_cmap('tab10')
    colors = [cmap(group_to_color[group]) for group in test_df['ImageID'].str[:3]]

    plt.figure(figsize=(8, 6))
    plt.scatter(test_df['Age'], test_df['Predicted_Age'], color=colors, label='Predicted vs Actual')
    handles = [plt.Line2D([0], [0], marker='o', color=cmap(i), linestyle='', markersize=10)
               for i, group in enumerate(unique_groups)]
    plt.legend(handles, unique_groups, title="ImageID Groups")
    plt.xlabel('Age')
    plt.ylabel('Predicted Age')
    plt.title('Age vs Predicted Age')
    plt.grid(True)
    plt.savefig(f"model_dumps/mix/{filename}_age_vs_predicted.png")
    plt.close()
