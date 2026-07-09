import os
import pandas as pd
from pathlib import Path



import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
from dataset_cls import ADNIDataset, ADNIDatasetViT
from torch.utils.data import DataLoader, Dataset
from matplotlib.image import imread

def set_random_seed(seed=69420):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

universal_seed = 69420

set_random_seed(universal_seed)


def resample_nifti(img_data, target_slices = 160):
    # Determine the current number of slices along the z-axis (3rd dimension)
    current_slices = img_data.shape[0]
    # Calculate the zoom factor for resampling (only along the z-axis)
    zoom_factor = target_slices / current_slices
    # Resample the image data along the z-axis
    resampled_data = zoom(img_data, (zoom_factor, 1, 1), order=3)  # order=3 for cubic interpolation
    # Ensure that the resampled data has the target number of slices
    # print (resampled_data.shape)
    # resampled_data = resampled_data[:target_slices,:,:]
    # print (resampled_data.shape)
    return resampled_data


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV file into a pandas DataFrame
csv_path = "adni_storage/adni_brainrotnet_metadata.csv"
df_adni = pd.read_csv(csv_path)
# df = df.sample(n=1000, random_state=69420)
# Add a new column 'filepath' with the constructed file paths
df_adni['filepath'] = df_adni.apply(
    lambda row: f"adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1)
df_adni = df_adni.loc[
    df_adni.groupby('SubjectID')['Age'].apply(lambda x: (x - x.median()).abs().idxmin())
].reset_index(drop=True)
df_adni = df_adni.sort_values(by='Age', ascending=True).reset_index(drop=True).head(900)

# Load independent dataset metadata
metadata_path = "ixi_storage/ixi_brainrotnet_metadata.csv"
df_ixi = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_ixi['filepath'] = df_ixi.apply(
    lambda row: f"ixi_storage/IXI_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "abide_storage/abide_brainrotnet_metadata.csv"
df_abide = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_abide['filepath'] = df_abide.apply(
    lambda row: f"abide_storage/ABIDEII_bias_corrected/{row['ImageID'][7:]}.stripped.N4.nii.gz",
    axis=1
)
df_abide = df_abide.sort_values(by='Age', ascending=False).reset_index(drop=True)
df_abide = df_abide.head(750)
# df_abide=df_abide.sample(n=200)

metadata_path = "dlbs_storage/dlbs_brainrotnet_metadata.csv"
df_dlbs = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_dlbs['filepath'] = df_dlbs.apply(
    lambda row: f"dlbs_storage/DLBS_bias_corrected/{row['ImageID'][4:]}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "cobre_storage/cobre_brainrotnet_metadata.csv"
df_cobre = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_cobre['filepath'] = df_cobre.apply(
    lambda row: f"cobre_storage/COBRE_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "fcon1000_storage/fcon1000_brainrotnet_metadata.csv"
df_fcon = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_fcon['filepath'] = df_fcon.apply(
    lambda row: f"fcon1000_storage/fcon1000_bias_corrected/{row['ImageID'][8:]}.stripped.N4.nii.gz",
    axis=1
)
df_fcon = df_fcon.dropna()
# df_fcon = df_fcon.sort_values(by='Age', ascending=False).reset_index(drop=True).head(300)
# df_fcon = df_fcon.sample(n=300)

metadata_path = "sald_storage/sald_brainrotnet_metadata.csv"
df_sald = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_sald['filepath'] = df_sald.apply(
    lambda row: f"sald_storage/SALD_bias_corrected/sub-{row['ImageID'][4:]}.stripped.N4.nii.gz",
    axis=1
)
# df_sald = df_sald.sort_values(by='Age', ascending=False).reset_index(drop=True).head(300)
# df_sald = df_sald.sample(n=300)

metadata_path = "corr_storage/corr_brainrotnet_metadata.csv"
df_corr = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_corr['filepath'] = df_corr.apply(
    lambda row: f"corr_storage/CORR_bias_corrected/{row['ImageID'][5:]}.stripped.N4.nii.gz",
    axis=1
)
df_corr = df_corr.sort_values(by='Age', ascending=True).reset_index(drop=True)
# df_corr = df_corr.head(300)
# df_corr = df_corr.sample(n=200)


metadata_path = "oasis1_storage/oasis1_brainrotnet_metadata.csv"
df_oas1 = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_oas1['filepath'] = df_oas1.apply(
    lambda row: f"oasis1_storage/oasis_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)
df_oas1 = df_oas1.sort_values(by='Age', ascending=False)
df_oas1 = df_oas1.reset_index(drop=True).head(300)

metadata_path = "camcan_storage/camcan_brainrotnet_metadata.csv"
df_camcan = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_camcan['filepath'] = df_camcan.apply(
    lambda row: f"camcan_storage/CamCAN_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "nimh_storage/nimh_mprage_brainrotnet_metadata.csv"
df_nimh = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_nimh['filepath'] = df_nimh.apply(
    lambda row: f"nimh_storage/nimh_bias_corrected/{row['ImageID'][5:]}_ses-01_acq-MPRAGE_rec-SCIC_T1w.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "bold_storage/bold_brainrotnet_metadata.csv"
df_bold = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_bold['filepath'] = df_bold.apply(
    lambda row: f"bold_storage/bold_bias_corrected/{row['ImageID'][5:]}_T1w.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "truecrime_storage/truecrime_brainrotnet_metadata.csv"
df_truecrime = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_truecrime['filepath'] = df_truecrime.apply(
    lambda row: f"truecrime_storage/truecrime_nii_gz_bias_corrected/{row['ImageID'][3:]}.stripped.N4.nii.gz",
    axis=1
)

metadata_path = "agerisk_storage/agerisk_brainrotnet_metadata.csv"
df_agerisk = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_agerisk['filepath'] = df_agerisk.apply(
    lambda row: f"agerisk_storage/agerisk_nii_gz_bias_corrected/{row['ImageID'][8:]}.N4.nii.gz",
    axis=1
)

metadata_path = "sudmex_storage/sudmex_brainrotnet_metadata.csv"
df_sudmex = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_sudmex['filepath'] = df_sudmex.apply(
    lambda row: f"sudmex_storage/sudmex_nii_gz_bias_corrected/{row['ImageID'][7:]}.stripped.N4.nii.gz",
    axis=1
)

df = pd.concat ([
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
                 df_bold[['ImageID', 'Sex', 'Age', 'filepath']]
                 ], ignore_index=True)
print (df)
# Ensure 'Age' is an integer
df['Age_Group'] = df['Age'].astype(int).apply(lambda x: f"{x:03d}"[:-1] + "0")
df['Age_Group'] = df['Age_Group'] + df['Sex']
print (df['Age_Group'].unique())
# Prepare dataset and dataloaders
sex_encoded = df['Sex'].apply(lambda x: 0 if x == 'M' else 1).tolist()
age_list = df['Age'].tolist()
filepath_list = df['filepath'].tolist()
label_list = df['Age_Group'].tolist()

# Get unique labels and create a mapping
unique_labels = sorted(set(label_list))  # Ensure consistent ordering
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}  # Reverse mapping for decoding

# Convert labels to integers
numeric_labels = [label_to_idx[label] for label in label_list]
label_list = numeric_labels

roi = 160

# Transformation pipeline for ViT
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Convert to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize for ViT
])


# Function to extract 16 evenly spaced slices
def extract_slices(volume, num_slices=16):
    total_slices = volume.shape[0]
    indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
    return volume[indices, :, :]  # Select slices


def calculate_loose_bounding_box_from_volume(volume):
    # Find indices of non-zero values
    non_zero_indices = np.argwhere(volume > 0)

    # Calculate min and max indices along each dimension
    min_indices = np.min(non_zero_indices, axis=0)
    max_indices = np.max(non_zero_indices, axis=0)

    # Convert indices to integers
    min_indices = min_indices.astype(int)
    max_indices = max_indices.astype(int)

    return min_indices, max_indices

from scipy.ndimage import label, find_objects

def calculate_bounding_box_from_volume(volume, intensity_threshold=0.1):
    # Normalize the volume
    volume_normalized = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    # Apply intensity threshold
    binary_mask = volume_normalized > intensity_threshold

    # Label connected components
    labeled_array, num_features = label(binary_mask)

    # Find the largest connected component
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # Exclude background
    largest_component = np.argmax(component_sizes)

    # Create a mask for the largest component
    brain_mask = labeled_array == largest_component

    # Find the bounding box of the largest component
    slices = find_objects(brain_mask.astype(int))[0]
    min_indices = [s.start for s in slices]
    max_indices = [s.stop - 1 for s in slices]

    return min_indices, max_indices

def crop_brain_volumes(brain_data):
    

        # Calculate bounding box from the brain volume
    min_indices, max_indices = calculate_bounding_box_from_volume(brain_data)

        # Crop the volume
    cropped_brain = brain_data[min_indices[0]:max_indices[0] + 1,
                                   min_indices[1]:max_indices[1] + 1,
                                   min_indices[2]:max_indices[2] + 1]
    return cropped_brain


def crop_loose_brain_volumes(brain_data):
    

        # Calculate bounding box from the brain volume
    min_indices, max_indices = calculate_loose_bounding_box_from_volume(brain_data)

        # Crop the volume
    cropped_brain = brain_data[min_indices[0]:max_indices[0] + 1,
                                   min_indices[1]:max_indices[1] + 1,
                                   min_indices[2]:max_indices[2] + 1]
    return cropped_brain


# Instantiate Dataset
vit_dataset = ADNIDatasetViT(filepath_list, label_list)

# Split Dataset
train_size = int(0.8 * len(vit_dataset))
val_size = len(vit_dataset) - train_size
generator = torch.Generator().manual_seed(universal_seed)
vit_train_dataset, vit_val_dataset = torch.utils.data.random_split(vit_dataset, [train_size, val_size], generator=generator)


def save_mid_slice (data, image_title, path):
    # Determine the middle sagittal slice index
    mid_slice_idx = data.shape[0] // 2  # Assuming sagittal slices along the first axis
    # Extract the middle sagittal slice
    sagittal_slice = data[mid_slice_idx, :, :]
    # Normalize the slice for visualization
    sagittal_slice_normalized = (sagittal_slice - np.min(sagittal_slice)) / (np.max(sagittal_slice) - np.min(sagittal_slice))
    # Define the output filename based on the original NIfTI file path
    output_filename = f"{path}/{image_title}.png"
    # Save the slice as a PNG image
    plt.imsave(output_filename, sagittal_slice_normalized, cmap='gray')

# -----------------------
# 4. Extract validation dataframe from indices
# -----------------------
val_indices = vit_val_dataset.indices  # indices into df
df_val = df.iloc[val_indices].reset_index(drop=True)
df_train = df.iloc[vit_train_dataset.indices].reset_index(drop=True)

print(f"Train size: {len(df_train)}, Val size: {len(df_val)}")
print(df_val.head())










# --- Target paths ---
base_dir = Path("TSAN-brain-age-estimation")
data_dir = base_dir / "data"
train_dir = data_dir / "train"
val_dir = data_dir / "val"
test_dir = data_dir / "test"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# --- Utility: create symlinks ---
def make_links(df, split_name):
    target_dir = data_dir / split_name
    for _, row in df.iterrows():
        filepath = Path(row["filepath"])
        if not filepath.exists():
            print(f"⚠️ Skipping missing file: {filepath}")
            continue
        dest_name = f"{row['dataset_name']}_{filepath.name}"
        dest_path = target_dir / dest_name
        if not dest_path.exists():
            try:
                os.symlink(filepath.resolve(), dest_path)
            except FileExistsError:
                pass

# --- Helper: infer dataset name from filepath ---
def infer_dataset_name(filepath):
    parts = filepath.split('/')
    for part in parts:
        if part.endswith("_storage"):
            return part.replace("_storage", "")
    return "unknown"

# --- Helper: format DataFrame for output ---
def format_df(df):
    df = df.copy()
    df["dataset_name"] = df["filepath"].apply(infer_dataset_name)
    df["sex"] = df["Sex"].map({"M": 1, "F": 0})
    df["filename"] = df.apply(
        lambda r: f"{r['dataset_name']}_{Path(r['filepath']).name}",
        axis=1
    )
    return df[["filename", "Age", "sex", "filepath", "dataset_name"]]








# --- Prepare splits ---
train_meta = format_df(df_train)
val_meta = format_df(df_val)
test_meta = format_df(df_sudmex)  # df_sald is your test set

# --- Create symlinks ---
make_links(train_meta, "train")
make_links(val_meta, "val")
make_links(test_meta, "test")

# --- Combine and save metadata ---
combined_meta = pd.concat([train_meta, val_meta, test_meta], ignore_index=True)
combined_meta[["filename", "Age", "sex"]].to_excel(
    data_dir / "dataset.xls",
    index=False,
    header=True  # ✅ no header row
)

print("✅ dataset.xls written without header.")
print(f"📁 Path: {data_dir / 'dataset.xls'}")
print("✅ Symlinks created in train/, val/, and test/ folders.")
