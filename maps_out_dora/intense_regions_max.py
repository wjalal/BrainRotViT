import nibabel as nib
import numpy as np
import pandas as pd
import re
import os

# --------- User paths (edit if needed) ----------
att_map_path = "attention_3d_mapped_backprop_dora_cropped_centered_resized.nii.gz"
aal_path     = "aal_crop_centered.nii"
output_csv   = "aal_grouped_weighted_attention_ranking.csv"  # set None to skip saving

# --------- Load NIfTI images ----------
if not os.path.exists(att_map_path):
    raise FileNotFoundError(f"Attention map not found: {att_map_path}")
if not os.path.exists(aal_path):
    raise FileNotFoundError(f"AAL atlas not found: {aal_path}")

att_img = nib.load(att_map_path)
aal_img = nib.load(aal_path)

att_data = att_img.get_fdata()
aal_data = aal_img.get_fdata()

# Ensure shapes match
if att_data.shape != aal_data.shape:
    raise ValueError(f"Shape mismatch: attention {att_data.shape} vs AAL {aal_data.shape}")

# --------- Hard-coded AAL label names (1..116) ----------
aal_labels = {
    1: "Precentral_L", 2: "Precentral_R",
    3: "Frontal_Sup_L", 4: "Frontal_Sup_R",
    5: "Frontal_Sup_Orb_L", 6: "Frontal_Sup_Orb_R",
    7: "Frontal_Mid_L", 8: "Frontal_Mid_R",
    9: "Frontal_Mid_Orb_L", 10: "Frontal_Mid_Orb_R",
    11: "Frontal_Inf_Oper_L", 12: "Frontal_Inf_Oper_R",
    13: "Frontal_Inf_Tri_L", 14: "Frontal_Inf_Tri_R",
    15: "Frontal_Inf_Orb_L", 16: "Frontal_Inf_Orb_R",
    17: "Rolandic_Oper_L", 18: "Rolandic_Oper_R",
    19: "Supp_Motor_Area_L", 20: "Supp_Motor_Area_R",
    21: "Olfactory_L", 22: "Olfactory_R",
    23: "Frontal_Sup_Medial_L", 24: "Frontal_Sup_Medial_R",
    25: "Frontal_Med_Orb_L", 26: "Frontal_Med_Orb_R",
    27: "Rectus_L", 28: "Rectus_R",
    29: "Insula_L", 30: "Insula_R",
    31: "Cingulum_Ant_L", 32: "Cingulum_Ant_R",
    33: "Cingulum_Mid_L", 34: "Cingulum_Mid_R",
    35: "Cingulum_Post_L", 36: "Cingulum_Post_R",
    37: "Hippocampus_L", 38: "Hippocampus_R",
    39: "ParaHippocampal_L", 40: "ParaHippocampal_R",
    41: "Amygdala_L", 42: "Amygdala_R",
    43: "Calcarine_L", 44: "Calcarine_R",
    45: "Cuneus_L", 46: "Cuneus_R",
    47: "Lingual_L", 48: "Lingual_R",
    49: "Occipital_Sup_L", 50: "Occipital_Sup_R",
    51: "Occipital_Mid_L", 52: "Occipital_Mid_R",
    53: "Occipital_Inf_L", 54: "Occipital_Inf_R",
    55: "Fusiform_L", 56: "Fusiform_R",
    57: "Postcentral_L", 58: "Postcentral_R",
    59: "Parietal_Sup_L", 60: "Parietal_Sup_R",
    61: "Parietal_Inf_L", 62: "Parietal_Inf_R",
    63: "SupraMarginal_L", 64: "SupraMarginal_R",
    65: "Angular_L", 66: "Angular_R",
    67: "Precuneus_L", 68: "Precuneus_R",
    69: "Paracentral_Lobule_L", 70: "Paracentral_Lobule_R",
    71: "Caudate_L", 72: "Caudate_R",
    73: "Putamen_L", 74: "Putamen_R",
    75: "Pallidum_L", 76: "Pallidum_R",
    77: "Thalamus_L", 78: "Thalamus_R",
    79: "Heschl_L", 80: "Heschl_R",
    81: "Temporal_Sup_L", 82: "Temporal_Sup_R",
    83: "Temporal_Pole_Sup_L", 84: "Temporal_Pole_Sup_R",
    85: "Temporal_Mid_L", 86: "Temporal_Mid_R",
    87: "Temporal_Pole_Mid_L", 88: "Temporal_Pole_Mid_R",
    89: "Temporal_Inf_L", 90: "Temporal_Inf_R",
    91: "Cerebelum_Crus1_L", 92: "Cerebelum_Crus1_R",
    93: "Cerebelum_Crus2_L", 94: "Cerebelum_Crus2_R",
    95: "Cerebelum_3_L", 96: "Cerebelum_3_R",
    97: "Cerebelum_4_5_L", 98: "Cerebelum_4_5_R",
    99: "Cerebelum_6_L", 100: "Cerebelum_6_R",
    101: "Cerebelum_7b_L", 102: "Cerebelum_7b_R",
    103: "Cerebelum_8_L", 104: "Cerebelum_8_R",
    105: "Cerebelum_9_L", 106: "Cerebelum_9_R",
    107: "Cerebelum_10_L", 108: "Cerebelum_10_R",
    109: "Vermis_1_2", 110: "Vermis_3",
    111: "Vermis_4_5", 112: "Vermis_6",
    113: "Vermis_7", 114: "Vermis_8",
    115: "Vermis_9", 116: "Vermis_10"
}

# --------- Detect numbering scheme (1..116 vs 2001..2116) ----------
region_ids = np.unique(aal_data)
region_ids = region_ids[region_ids > 0]  # ignore background
if region_ids.size == 0:
    raise RuntimeError("No positive region labels found in AAL atlas (all zeros?)")

numbering_offset = 2000 if np.min(region_ids) > 2000 else 0
print(f"Detected AAL numbering offset: {numbering_offset} (0 means 1..116 numbering)")

# --------- Compute per-ROI stats: mean intensity and voxel count ----------
rows = []
for rid in np.unique(region_ids):
    mask = (aal_data == rid)
    vox_count = int(mask.sum())
    if vox_count == 0:
        continue
    # max over voxels in attention map (ignore NaNs)
    vals = att_data[mask]
    if np.all(np.isnan(vals)):
        max_val = np.nan
    else:
        max_val = float(np.nanmax(vals))

    rid_corrected = int(rid - numbering_offset)
    name = aal_labels.get(rid_corrected, f"Region_{rid_corrected}")
    rows.append({
        "RawID": int(rid),
        "ID": rid_corrected,
        "RegionName": name,
        "VoxelCount": vox_count,
        "MaxIntensity": max_val
    })

roi_df = pd.DataFrame(rows)
if roi_df.empty:
    raise RuntimeError("No ROI statistics were computed. Check atlas numbering or that atlas overlaps attention map.")

# Drop ROIs with NaN mean (optional) — here we drop them
roi_df = roi_df.dropna(subset=["MaxIntensity"])
if roi_df.empty:
    raise RuntimeError("All ROIs have NaN mean intensities. Check attention map values.")

# --------- Grouping function (name-based) ----------
def base_group(name):
    """
    Collapse ROI names to their first word before the first underscore.
    Examples:
      - Precentral_L -> Precentral
      - Cerebelum_Crus1_L -> Cerebelum
      - Vermis_4_5 -> Vermis
    """
    return name.split("_")[0]
# Drop ROIs with NaN max intensity
roi_df = roi_df.dropna(subset=["MaxIntensity"])

# Add Group column
roi_df["Group"] = roi_df["RegionName"].apply(base_group)

# Compute group-level max (not weighted — direct max across subregions)
grouped = roi_df.groupby("Group").apply(
    lambda g: pd.Series({
        "TotalVoxels": int(g["VoxelCount"].sum()),
        "MaxIntensity": float(np.nanmax(g["MaxIntensity"]))
    })
).reset_index()

# Sort descending by MaxIntensity
grouped = grouped.sort_values("MaxIntensity", ascending=False).reset_index(drop=True)
grouped.insert(0, "Rank", np.arange(1, len(grouped) + 1))

# Exponentiate intensity over base 100: maps [0,1] -> [1,100] (monotonic, so the
# ranking is unchanged) for a more readable 1-100 intensity scale.
grouped["ExpIntensity"] = 100.0 ** grouped["MaxIntensity"]

# Print
print("\n=== Grouped AAL regions (ranked by max intensity, shown as 100**intensity) ===\n")
for _, r in grouped.iterrows():
    print(f"{int(r['Rank']):3d}. {r['Group']:25s}  | Intensity(100^) = {r['ExpIntensity']:.4f}  | TotalVoxels = {r['TotalVoxels']}")