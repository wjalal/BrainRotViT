"""
3dresnet_test.py

Evaluate the trained 3D ResNet (saved best model) on the four EXTERNAL,
held-out cohorts used elsewhere in this project as the OOD test set:
    SALD, truecrime, agerisk, sudmex
(the same four scored by vit_dora_train_feature_cnn_main_mix_roi_test.py and
sfcn_run.py). None of these appear in the mixed train/val split, so this is a
pure out-of-distribution generalisation test.

The model architecture and the exact volume preprocessing (RAS reorient ->
brain-crop -> resample to 160x192x160 -> min-max -> white0 standardisation) are
imported directly from 3dresnet.py so training and testing stay identical.

Usage:
    python 3dresnet_test.py [model_name] [batch_size]
        model_name   checkpoint name under model_dumps/resnet/<name>/ (default 3dresnet)
        batch_size   default 4
"""
import os
import sys
import pickle
import importlib.util
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Import model + dataset + preprocessing from 3dresnet.py (its main() is guarded
# by __name__ == "__main__", so importing does not launch training).
# ---------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("resnet3d_train", "3dresnet.py")
r3d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3d)
resnet34 = r3d.resnet34
BrainAgeDataset = r3d.BrainAgeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = sys.argv[1] if len(sys.argv) > 1 else "3dresnet"
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4
out_dir = f"model_dumps/resnet/{model_name}"
ckpt_path = os.path.join(out_dir, f"{model_name}_best_model_with_metadata.pkl")


# ---------------------------------------------------------------------------
# Build the four external test cohorts (full cohorts, no sampling) -- filepath
# constructions match the test/adjuster scripts exactly.
# ---------------------------------------------------------------------------
def load_cohorts():
    df_sald = pd.read_csv("sald_storage/sald_brainrotnet_metadata.csv")
    df_sald["filepath"] = df_sald.apply(
        lambda r: f"sald_storage/SALD_bias_corrected/sub-{r['ImageID'][4:]}.stripped.N4.nii.gz", axis=1)

    df_truecrime = pd.read_csv("truecrime_storage/truecrime_brainrotnet_metadata.csv")
    df_truecrime["filepath"] = df_truecrime.apply(
        lambda r: f"truecrime_storage/truecrime_nii_gz_bias_corrected/{r['ImageID'][3:]}.stripped.N4.nii.gz", axis=1)

    df_agerisk = pd.read_csv("agerisk_storage/agerisk_brainrotnet_metadata.csv")
    df_agerisk["filepath"] = df_agerisk.apply(
        lambda r: f"agerisk_storage/agerisk_nii_gz_bias_corrected/{r['ImageID'][8:]}.N4.nii.gz", axis=1)

    df_sudmex = pd.read_csv("sudmex_storage/sudmex_brainrotnet_metadata.csv")
    df_sudmex["filepath"] = df_sudmex.apply(
        lambda r: f"sudmex_storage/sudmex_nii_gz_bias_corrected/{r['ImageID'][7:]}.stripped.N4.nii.gz", axis=1)

    return {"sald": df_sald, "truecrime": df_truecrime,
            "agerisk": df_agerisk, "sudmex": df_sudmex}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
if not os.path.exists(ckpt_path):
    sys.exit(f"Checkpoint not found: {ckpt_path}")

model = resnet34(num_classes=1, dropout=False).to(device)
with open(ckpt_path, "rb") as f:
    ckpt = pickle.load(f)
state = ckpt["model_state"]
# Strip any DataParallel 'module.' prefix so it loads into the plain model.
state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
model.load_state_dict(state)
model.eval()
print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch')}, saved val MAE {ckpt.get('val_mae'):.4f})")


@torch.no_grad()
def predict(dataset):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    preds = []
    for img, sex, age in loader:
        img = img.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(img)
        preds.extend(out.float().squeeze(1).cpu().tolist())
    return np.array(preds)


def stats(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = r2_score(y_true, y_pred)
    r = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else float("nan")
    rho = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else float("nan")
    return mae, rmse, r2, r, rho


# ---------------------------------------------------------------------------
# Evaluate each cohort, accumulate predictions
# ---------------------------------------------------------------------------
cohorts = load_cohorts()
all_rows = []
print(f"\n{'cohort':<11s}{'n':>6s}{'MAE':>9s}{'RMSE':>9s}{'R2':>8s}{'Pearson':>9s}{'Spearman':>10s}")
print("-" * 62)
for name, df in cohorts.items():
    ds = BrainAgeDataset(df, target_shape=(160, 192, 160))
    preds = predict(ds)
    truth = ds.df["Age"].to_numpy(dtype=float)
    mae, rmse, r2, r, rho = stats(truth, preds)
    print(f"{name:<11s}{len(ds):>6d}{mae:>9.3f}{rmse:>9.3f}{r2:>8.3f}{r:>9.3f}{rho:>10.3f}")

    out = ds.df[["ImageID", "Sex", "Age", "filepath"]].copy()
    out["Predicted_Age"] = preds
    out["cohort"] = name
    all_rows.append(out)

# ---------------------------------------------------------------------------
# Overall (pooled across the four cohorts) + save predictions
# ---------------------------------------------------------------------------
res = pd.concat(all_rows, ignore_index=True)
mae, rmse, r2, r, rho = stats(res["Age"], res["Predicted_Age"])
print("-" * 62)
print(f"{'OVERALL':<11s}{len(res):>6d}{mae:>9.3f}{rmse:>9.3f}{r2:>8.3f}{r:>9.3f}{rho:>10.3f}")

out_csv = os.path.join(out_dir, f"{model_name}_predicted_test.csv")
res.to_csv(out_csv, index=False)
print(f"\nSaved per-sample predictions: {out_csv}")
