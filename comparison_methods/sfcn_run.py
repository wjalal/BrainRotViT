"""
sfcn_run.py

Runs Han Peng's SFCN (UKBiobank_deep_pretrain) on the mixed-dataset train/val
split produced by sfcn_adjuster.py (identical to
vit_dora_train_feature_cnn_main_mix_roi.py).

Why fine-tune rather than pure inference:
  * The published SFCN checkpoint expects UK-Biobank T1 volumes registered to
    MNI152 1mm (182x218x182) and predicts ages in [42,82] only. Our volumes are
    skull-stripped/bias-corrected in NATIVE space (varying shapes) and span ages
    8-96. So we init the feature extractor from the pretrained weights, attach a
    fresh soft-classification head sized to our age range, and fine-tune on the
    train split -- exactly how the other baselines here are trained.

Preprocessing per volume (on the fly):
    reorient to RAS -> crop brain bounding box -> resample to (160,192,160)
    -> divide by mean (SFCN convention).
Soft-label KL-divergence training (num2vect / my_KLDivLoss), age read out as
softmax(logits) . bin_centers.

Usage:
    python sfcn_run.py [epochs] [batch_size] [smoke]
        epochs      default 30
        batch_size  default 4
        smoke       optional; run on 24 train / 24 val samples for a quick check
"""
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import zoom, label, find_objects
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
from tqdm import tqdm

sys.path.insert(0, "UKBiobank_deep_pretrain")
from dp_model.model_files.sfcn import SFCN
from dp_model import dp_loss as dpl
from dp_model import dp_utils as dpu

universal_seed = 69420
np.random.seed(universal_seed)
torch.manual_seed(universal_seed)
torch.cuda.manual_seed_all(universal_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "UKBiobank_deep_pretrain/data"
PRETRAINED = "UKBiobank_deep_pretrain/brain_age/run_20190719_00_epoch_best_mae.p"
TARGET_SHAPE = (160, 192, 160)
BIN_STEP = 1
SIGMA = 1
OUT_PREFIX = "model_dumps/mix/sfcn"
os.makedirs("model_dumps/mix", exist_ok=True)

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4
smoke = "smoke" in sys.argv[3:]
# eval-only: skip training, load the saved best model, and score the test set
test_only = "test_only" in sys.argv[3:]


# ---------------------------------------------------------------------------
# Preprocessing (matches the brain-crop used by the other pipelines)
# ---------------------------------------------------------------------------
def crop_brain(volume, intensity_threshold=0.1):
    v = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    mask = v > intensity_threshold
    lab, n = label(mask)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    largest = np.argmax(sizes)
    sl = find_objects((lab == largest).astype(int))[0]
    return volume[sl[0].start:sl[0].stop, sl[1].start:sl[1].stop, sl[2].start:sl[2].stop]


def load_volume(path):
    nii = nib.load(path)
    ornt = ornt_transform(io_orientation(nii.affine), axcodes2ornt(("R", "A", "S")))
    data = apply_orientation(nii.get_fdata(), ornt)
    data = crop_brain(data)
    factors = [t / s for t, s in zip(TARGET_SHAPE, data.shape)]
    data = zoom(data, factors, order=1)
    # guard against off-by-one from rounding
    data = data[:TARGET_SHAPE[0], :TARGET_SHAPE[1], :TARGET_SHAPE[2]]
    pad = [(0, t - s) for t, s in zip(TARGET_SHAPE, data.shape)]
    data = np.pad(data, pad, mode="constant")
    m = data.mean()
    data = data / m if m > 0 else data
    return data.astype(np.float32)


class SFCNDataset(Dataset):
    def __init__(self, meta_df, split, bin_range):
        self.df = meta_df[meta_df["split"] == split].reset_index(drop=True)
        self.split = split
        self.bin_range = bin_range

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(DATA_DIR, self.split, row["filename"])
        data = load_volume(path)
        x = torch.tensor(data[None], dtype=torch.float32)      # (1,160,192,160)
        y, _ = dpu.num2vect(row["Age"], self.bin_range, BIN_STEP, SIGMA)
        y = torch.tensor(y, dtype=torch.float32)               # soft label
        return x, y, float(row["Age"]), idx


# ---------------------------------------------------------------------------
# Data / bin range
# ---------------------------------------------------------------------------
meta = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
if smoke:
    meta = pd.concat([meta[meta.split == "train"].head(24),
                      meta[meta.split == "val"].head(24)]).reset_index(drop=True)
    epochs = min(epochs, 2)

train_ages = meta[meta.split == "train"]["Age"]
bin_start = int(np.floor(train_ages.min() - 2))
bin_end = int(np.ceil(train_ages.max() + 2))
bin_range = [bin_start, bin_end]
n_bins = (bin_end - bin_start) // BIN_STEP
_, bin_centers = dpu.num2vect(np.array([train_ages.iloc[0]]), bin_range, BIN_STEP, SIGMA)
bin_centers_t = torch.tensor(bin_centers, dtype=torch.float32, device=device)
print(f"Bin range {bin_range}  ({n_bins} bins, step {BIN_STEP})")

train_loader = DataLoader(SFCNDataset(meta, "train", bin_range), batch_size=batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(SFCNDataset(meta, "val", bin_range), batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(SFCNDataset(meta, "test", bin_range), batch_size=batch_size,
                         shuffle=False, num_workers=4, pin_memory=True)
print(f"train {len(train_loader.dataset)} / val {len(val_loader.dataset)} "
      f"/ test {len(test_loader.dataset)} samples")

# ---------------------------------------------------------------------------
# Model: pretrained SFCN feature extractor + fresh head for our bin count
# ---------------------------------------------------------------------------
model = SFCN(output_dim=n_bins)
model = nn.DataParallel(model)
if os.path.exists(PRETRAINED):
    pretrained = torch.load(PRETRAINED, map_location="cpu")
    # Drop the old classification head (40 bins) so shapes match; keep the rest.
    pretrained = {k: v for k, v in pretrained.items()
                  if not k.startswith("module.classifier.conv_6")}
    missing, unexpected = model.load_state_dict(pretrained, strict=False)
    print(f"Loaded pretrained SFCN (skipped head). missing={len(missing)} unexpected={len(unexpected)}")
else:
    print("WARNING: pretrained checkpoint not found; training from scratch.")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.3)


def predict_age(logits):
    # logits: (B, n_bins) log-softmax outputs -> expected age
    prob = torch.exp(logits)
    return (prob * bin_centers_t).sum(dim=1)


def run_eval(loader):
    model.eval()
    preds, trues, idxs = [], [], []
    with torch.no_grad():
        for x, y, age, idx in loader:
            x = x.to(device)
            out = model(x)[0].reshape(x.size(0), -1)
            preds.extend(predict_age(out).cpu().tolist())
            trues.extend(age.tolist())
            idxs.extend(idx.tolist())
    preds, trues = np.array(preds), np.array(trues)
    mae = np.mean(np.abs(preds - trues))
    return mae, preds, trues, idxs


best_mae = np.inf
val_df = meta[meta.split == "val"].reset_index(drop=True)
test_df = meta[meta.split == "test"].reset_index(drop=True)


def cohort_of(filename):
    return filename.split("_")[0]


def eval_test_and_save():
    """Load the best fine-tuned model, score the test set, report overall +
    per-cohort MAE, and write predictions."""
    ckpt = f"{OUT_PREFIX}_best.pth"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded best model from {ckpt} for test evaluation.")
    else:
        print("No best checkpoint found; evaluating current weights.")
    mae, preds, trues, idxs = run_eval(test_loader)
    out = test_df.iloc[idxs].copy()
    out["Predicted_Age"] = preds
    out["cohort"] = out["filename"].apply(cohort_of)
    out.to_csv(f"{OUT_PREFIX}_predicted_test.csv", index=False)
    print(f"\n=== TEST (OOD) results ===  n={len(out)}  overall MAE {mae:.3f}")
    for coh, g in out.groupby("cohort"):
        cmae = np.mean(np.abs(g["Predicted_Age"] - g["Age"]))
        print(f"  {coh:<10s} n={len(g):<4d} MAE {cmae:.3f}")
    print(f"Saved: {OUT_PREFIX}_predicted_test.csv")


if test_only:
    eval_test_and_save()
    sys.exit(0)

for epoch in range(epochs):
    model.train()
    running = 0.0
    for x, y, age, idx in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)[0].reshape(x.size(0), -1)   # (B, n_bins) log-softmax
        loss = dpl.my_KLDivLoss(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item()
    scheduler.step()

    mae, preds, trues, idxs = run_eval(val_loader)
    print(f"Epoch {epoch+1}: train_KL {running/len(train_loader):.4f}  val_MAE {mae:.3f}")

    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), f"{OUT_PREFIX}_best.pth")
        out_df = val_df.iloc[idxs].copy()
        out_df["Predicted_Age"] = preds
        out_df.to_csv(f"{OUT_PREFIX}_predicted_val.csv", index=False)
        print(f"  -> new best val MAE {best_mae:.3f}; saved model + predictions")

print(f"\nDone. Best val MAE: {best_mae:.3f}")
print(f"Predictions: {OUT_PREFIX}_predicted_val.csv")

# Evaluate the best fine-tuned model on the held-out OOD test cohorts.
eval_test_and_save()
