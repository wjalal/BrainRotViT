"""
sfcn_adjuster.py

Reproduces the EXACT train/val split used by
vit_dora_train_feature_cnn_main_mix_roi.py (the mixed-dataset training file) and
organises the corresponding NIfTI volumes into the folder structure the SFCN
runner (sfcn_run.py) consumes -- using symlinks only, no file copies.

The split is a sample-level torch.utils.data.random_split with
generator seed 69420, 80/20, over the SAME concatenated `df` (same per-dataset
head/sort/dropna/dedup operations, SALD excluded, identical concat order). By
rebuilding `df` verbatim and re-running the identical random_split, the produced
train/val indices are guaranteed to match the ViT-DoRA run row-for-row.

Output:
    UKBiobank_deep_pretrain/data/train/<dataset>_<file>.nii.gz  (symlinks)
    UKBiobank_deep_pretrain/data/val/<dataset>_<file>.nii.gz    (symlinks)
    UKBiobank_deep_pretrain/data/dataset.csv  (filename, split, Age, sex, src)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from dataset_cls import ADNIDatasetViT

universal_seed = 69420
np.random.seed(universal_seed)
torch.manual_seed(universal_seed)

# ---------------------------------------------------------------------------
# Rebuild `df` EXACTLY as in vit_dora_train_feature_cnn_main_mix_roi.py
# ---------------------------------------------------------------------------
df_adni = pd.read_csv("adni_storage/adni_brainrotnet_metadata.csv")
df_adni['filepath'] = df_adni.apply(
    lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1)
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

# ---------------------------------------------------------------------------
# TEST cohorts (held-out OOD) -- exactly the four used by
# vit_dora_train_feature_cnn_main_mix_roi_test.py: full SALD + truecrime +
# agerisk + sudmex (no head/sort/sampling). None overlap the train/val `df`.
# ---------------------------------------------------------------------------
df_sald = pd.read_csv("sald_storage/sald_brainrotnet_metadata.csv")
df_sald['filepath'] = df_sald.apply(
    lambda row: f"sald_storage/SALD_bias_corrected/sub-{row['ImageID'][4:]}.stripped.N4.nii.gz", axis=1)

df_truecrime = pd.read_csv("truecrime_storage/truecrime_brainrotnet_metadata.csv")
df_truecrime['filepath'] = df_truecrime.apply(
    lambda row: f"truecrime_storage/truecrime_nii_gz_bias_corrected/{row['ImageID'][3:]}.stripped.N4.nii.gz", axis=1)

df_agerisk = pd.read_csv("agerisk_storage/agerisk_brainrotnet_metadata.csv")
df_agerisk['filepath'] = df_agerisk.apply(
    lambda row: f"agerisk_storage/agerisk_nii_gz_bias_corrected/{row['ImageID'][8:]}.N4.nii.gz", axis=1)

df_sudmex = pd.read_csv("sudmex_storage/sudmex_brainrotnet_metadata.csv")
df_sudmex['filepath'] = df_sudmex.apply(
    lambda row: f"sudmex_storage/sudmex_nii_gz_bias_corrected/{row['ImageID'][7:]}.stripped.N4.nii.gz", axis=1)

df_test = pd.concat([
    df_sald[['ImageID', 'Sex', 'Age', 'filepath']],
    df_truecrime[['ImageID', 'Sex', 'Age', 'filepath']],
    df_agerisk[['ImageID', 'Sex', 'Age', 'filepath']],
    df_sudmex[['ImageID', 'Sex', 'Age', 'filepath']],
], ignore_index=True)

# NOTE: SALD is intentionally excluded from train/val (commented out in the
# ViT-DoRA training file); it appears only in the test set above.
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

# Age_Group label list is only needed to build the identical dataset object that
# was passed to random_split (the labels themselves don't affect the indices).
df['Age_Group'] = df['Age'].astype(int).apply(lambda x: f"{x:03d}"[:-1] + "0") + df['Sex']
filepath_list = df['filepath'].tolist()
unique_labels = sorted(set(df['Age_Group'].tolist()))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
label_list = [label_to_idx[l] for l in df['Age_Group'].tolist()]

# ---------------------------------------------------------------------------
# Identical split: torch random_split, generator seed 69420, 80/20
# ---------------------------------------------------------------------------
vit_dataset = ADNIDatasetViT(filepath_list, label_list)
train_size = int(0.8 * len(vit_dataset))
val_size = len(vit_dataset) - train_size
generator = torch.Generator().manual_seed(universal_seed)
train_ds, val_ds = random_split(vit_dataset, [train_size, val_size], generator=generator)
train_indices = list(train_ds.indices)
val_indices = list(val_ds.indices)
print(f"Total {len(df)}  ->  train {len(train_indices)} / val {len(val_indices)}")


def infer_dataset_name(filepath):
    for part in filepath.split('/'):
        if part.endswith("_storage"):
            return part.replace("_storage", "")
    return "unknown"


# ---------------------------------------------------------------------------
# Symlink each volume into data/{train,val}/  (no copies)
# ---------------------------------------------------------------------------
base_dir = Path("UKBiobank_deep_pretrain/data")
records = []
missing = 0

# train/val rows come from the split indices into `df`; test rows are the full
# `df_test` cohorts. `frame` tells us which DataFrame to index for each split.
split_plan = [
    ("train", df, train_indices),
    ("val", df, val_indices),
    ("test", df_test, list(range(len(df_test)))),
]
for split_name, frame, indices in split_plan:
    target_dir = base_dir / split_name
    target_dir.mkdir(parents=True, exist_ok=True)
    for i in indices:
        row = frame.iloc[i]
        src = Path(row["filepath"])
        ds_name = infer_dataset_name(row["filepath"])
        dest_name = f"{ds_name}_{src.name}"
        dest_path = target_dir / dest_name
        if not src.exists():
            missing += 1
            continue
        if not dest_path.exists():
            try:
                os.symlink(src.resolve(), dest_path)
            except FileExistsError:
                pass
        records.append({
            "filename": dest_name,
            "split": split_name,
            "Age": row["Age"],
            "sex": 1 if row["Sex"] == "M" else 0,
            "src": str(src),
        })

meta = pd.DataFrame(records)
meta.to_csv(base_dir / "dataset.csv", index=False)
print(f"Wrote {base_dir/'dataset.csv'}  ({len(meta)} rows; {missing} source files missing)")
print("Per-split counts:", meta["split"].value_counts().to_dict())
test_src = meta[meta.split == "test"]["filename"].str.split("_").str[0]
print("Test cohort counts:", test_src.value_counts().to_dict())
print(f"Symlinks under {base_dir}/{{train,val,test}}")
print("Age range (all):", meta["Age"].min(), "-", meta["Age"].max())
