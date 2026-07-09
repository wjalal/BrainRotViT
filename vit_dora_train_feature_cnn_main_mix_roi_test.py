import os
import pandas as pd
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from tqdm import tqdm
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import sys
import SimpleITK as sitk
from scipy.ndimage import zoom
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
    axis=1
)
df_adni = df_adni.sort_values(by='Age', ascending=True).reset_index(drop=True).head(500)
# df_adni=df_adni.sample(n=400)

# Load independent dataset metadata
metadata_path = "ixi_storage/ixi_brainrotnet_metadata.csv"
df_ixi = pd.read_csv(metadata_path)
# Update filepaths for the independent dataset
df_ixi['filepath'] = df_ixi.apply(
    lambda row: f"ixi_storage/IXI_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
    axis=1
)

# metadata_path = "abide_storage/abide_brainrotnet_metadata.csv"
# df_abide = pd.read_csv(metadata_path)
# # Update filepaths for the independent dataset
# df_abide['filepath'] = df_abide.apply(
#     lambda row: f"abide_storage/ABIDEII_bias_corrected/{row['ImageID'][7:]}.stripped.N4.nii.gz",
#     axis=1
# )
# df_abide = df_abide.sort_values(by='Age', ascending=False).reset_index(drop=True).head(500)
# df_abide=df_abide.sample(n=400)

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
df_corr = df_corr.sort_values(by='Age', ascending=True).reset_index(drop=True).head(300)
df_corr = df_corr.sample(n=200)


# metadata_path = "oasis1_storage/oasis1_brainrotnet_metadata.csv"
# df_oas1 = pd.read_csv(metadata_path)
# # Update filepaths for the independent dataset
# df_oas1['filepath'] = df_oas1.apply(
#     lambda row: f"oasis1_storage/oasis_nii_gz_bias_corrected/{row['ImageID']}.stripped.N4.nii.gz",
#     axis=1
# )
# df_oas1 = df_oas1.sort_values(by='Age', ascending=False).reset_index(drop=True).head(300)

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
                #  df_adni[['ImageID', 'Sex', 'Age', 'filepath']], 
                #  df_ixi[['ImageID', 'Sex', 'Age', 'filepath']], 
                #  df_abide[['ImageID', 'Sex', 'Age', 'filepath']],
                #  df_dlbs[['ImageID', 'Sex', 'Age', 'filepath']],
                #  df_cobre[['ImageID', 'Sex', 'Age', 'filepath']],
                #  df_fcon[['ImageID', 'Sex', 'Age', 'filepath']],
                 df_sald[['ImageID', 'Sex', 'Age', 'filepath']],
                #  df_corr[['ImageID', 'Sex', 'Age', 'filepath']], 
                #  df_oas1[['ImageID', 'Sex', 'Age', 'filepath']],
                # df_camcan[['ImageID', 'Sex', 'Age', 'filepath']],
                # df_nimh[['ImageID', 'Sex', 'Age', 'filepath']],
                # df_bold[['ImageID', 'Sex', 'Age', 'filepath']].
                df_truecrime[['ImageID', 'Sex', 'Age', 'filepath']],
                  df_agerisk[['ImageID', 'Sex', 'Age', 'filepath']],
                  df_sudmex[['ImageID', 'Sex', 'Age', 'filepath']],
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

# Function to preprocess data and dynamically expand slices while saving to disk
def preprocess_and_expand(dataset, transform, output_dir, num_slices=16):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    expanded_images, expanded_labels = [], []

    for filepath, label in tqdm(dataset, desc="Processing Slices"):
    # for filepath, label in dataset:
        # print (filepath)
        # Check if all slice files already exist
        all_slices_exist = True
        slice_filenames = [
            os.path.join(output_dir, f"{os.path.basename(filepath)}_slice_{i}.pt")
            for i in range(num_slices)
        ]
        if not all(os.path.exists(slice_file) for slice_file in slice_filenames):
            all_slices_exist = False

        # Skip processing if all slices exist
        if all_slices_exist:
            expanded_images.extend(slice_filenames)  # Add existing file paths
            expanded_labels.extend([label] * num_slices)
            continue

        # Load NIfTI image only if slices are missing
        nii_img = nib.load(filepath)
        orig_ornt = io_orientation(nii_img.affine)
        ras_ornt = axcodes2ornt(("R", "A", "S"))
        ornt_trans = ornt_transform(orig_ornt, ras_ornt)
        data = nii_img.get_fdata()  # Load image data
        data = apply_orientation(data, ornt_trans)

        data = crop_loose_brain_volumes(data)

        # Normalize and extract slices
        data = (data - data.min()) / (data.max() - data.min())
        slices = extract_slices(data, num_slices)

        # Transform each slice, save to file, and add to dataset
        for i, slice_data in enumerate(slices):
            slice_filename = slice_filenames[i]
            if not os.path.exists(slice_filename):
                transformed_slice = transform(slice_data)  # Transform slice
                torch.save(transformed_slice, slice_filename)  # Save to file
            expanded_images.append(slice_filename)  # Store file path
            expanded_labels.append(label)

        # print("Processed:")

    return expanded_images, expanded_labels

# Instantiate Dataset
vit_dataset = ADNIDatasetViT(filepath_list, label_list)

# Split Dataset
train_size = int(0.8 * len(vit_dataset))
val_size = len(vit_dataset) - train_size
generator = torch.Generator().manual_seed(universal_seed)
vit_train_dataset, vit_val_dataset = torch.utils.data.random_split(vit_dataset, [train_size, val_size], generator=generator)

# Create New Dataset with Filepaths
class ExpandedDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load the image from file
        image = torch.load(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Define output directory for slices
slice_count = 32
output_dir = f"processed_slices_{slice_count}"

# Preprocess and expand the training data
expanded_image_paths, expanded_labels = preprocess_and_expand(vit_train_dataset, transform, output_dir, slice_count)

# Create Expanded Dataset and DataLoader
expanded_train_dataset = ExpandedDataset(expanded_image_paths, expanded_labels)
expanded_train_loader = DataLoader(expanded_train_dataset, batch_size=8, shuffle=True)

# Print Sizes
print(f"Original Training Dataset Size: {len(vit_train_dataset)}")
print(f"Expanded Training Dataset Size: {len(expanded_train_dataset)}")

from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
# Load ViT model
num_classes = df['Age_Group'].nunique()  # Number of unique Age_Groups
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=20,
    ignore_mismatched_sizes=True,
)

# Apply DoRA (weight-decomposed LoRA) to the attention projections, matching the
# vit_dora training setup. After (optionally recovered) training the adapters
# are merged back so feature extraction can keep calling model.vit(...).
dora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
    use_dora=True,
)
model = get_peft_model(model, dora_config)
model.print_trainable_parameters()

model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Function to save checkpoint
def save_checkpoint(epoch, model, optimizer, path="model_dumps/vit_train_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at epoch {epoch+1}")

# Function to load checkpoint
def load_checkpoint(path="model_dumps/vit_train_checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    return start_epoch

# Check if recovery mode is enabled
checkpoint_path = "model_dumps/vit_dora_train_checkpoint.pth"
start_epoch = 0

start_epoch = 0

if len(sys.argv) > 4 and sys.argv[4] == "recover":
    start_epoch = load_checkpoint(path=checkpoint_path)

# Training loop
vit_train_epochs = 5

# Merge DoRA adapters into the base ViT so feature extraction below can use
# model.vit(...) (get_peft_model otherwise hides the .vit submodule).
model = model.merge_and_unload()

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
# model = ViTModel.from_pretrained("google/vit-base-patch16-224")
# model.to(device)  # Move the model to the GPU (if available)
model.eval()

# Update image transform for grayscale images to match ViT input requirements
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Convert to RGB directly
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

torch.cuda.empty_cache()  # Free GPU memory

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


# To store features and labels
features_list = []
labels_list = []


# # Directory to save processed images and features
# os.makedirs(f"adni_storage/ADNI_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
# # Process each row in the DataFrame
# for _, row in tqdm(df_adni.iterrows(), total=len(df_adni), desc="Processing images"):
#     filepath = row['filepath']
#     image_title = f"{row['ImageID'][4:]}_{row['SubjectID']}"

#     # Check if the feature file already exists
#     feature_file_path = f"adni_storage/ADNI_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     if os.path.exists(feature_file_path):
#         # If file exists, load the features from the file
#         features = np.load(feature_file_path)
#         features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
        
#         from PIL import Image
#         # Normalize the array to 0-255 for grayscale image
#         data_normalized = ((features - np.min(features)) / (np.max(features) - np.min(features)) * 255).astype(np.uint8)
#         data_normalized = np.repeat(data_normalized, 4, axis=0)
#         # Create an image from the array
#         img = Image.fromarray(np.transpose(data_normalized), mode='L')  # 'L' mode for grayscale
#         # Save the image
#         # img.save(f"adni_storage/ADNI_features/train_dora_e{vit_train_epochs}_s{slice_count}/featuremaps/{image_title}_fm.png")

#         features_list.append(features)  # Flatten the features and add to the list
#         labels_list.append(row['Age'])  # Add the corresponding age label
#     else:
#         # print ("hiii")
#         if os.path.exists(filepath):
#             try:
#                 # Load the NIfTI image
#                 nii_img = nib.load(filepath)

#                 # Get current orientation and reorient to RAS
#                 orig_ornt = io_orientation(nii_img.affine)
#                 ras_ornt = axcodes2ornt(("R", "A", "S"))
#                 ornt_trans = ornt_transform(orig_ornt, ras_ornt)

#                 data = nii_img.get_fdata()  # Load image data
#                 data = apply_orientation(data, ornt_trans)

#                 affine = nii_img.affine  # Affine transformation matrix

#                 data = crop_brain_volumes(data)

#                 # Resample the volume to 160 slices (if required)
#                 data = resample_nifti(data, target_slices=160)

#                 save_mid_slice (data, image_title, "adni_storage/ADNI_images")

#                 # Extract features for all sagittal slices
#                 features = []
#                 for slice_idx in range(data.shape[0]):
#                     slice_data = data[slice_idx, :, :]
#                     slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize

#                     # Transform slice for ViT input
#                     slice_tensor = transform(slice_data).unsqueeze(0).to(device)  # Add batch dimension and move to GPU

#                     # Extract features using ViT
#                     with torch.no_grad():
#                         # #outputs = model(slice_tensor)
#                         # slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Move output back to CPU
#                         slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#                         features.append(slice_features)

#                 # Save extracted features
#                 features = np.array(features)
#                 np.save(feature_file_path, features)
#                 features_list.append(features)
#                 labels_list.append(row['Age'])  # Target is 'Age'

#             except Exception as e:
#                 print(f"Error processing {filepath}: {e}")
#         else:
#             print(f"File not found: {filepath}")


# # Directory to save processed images and features
# os.makedirs(f"ixi_storage/IXI_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
# for _, row in tqdm(df_ixi.iterrows(), total=len(df_ixi), desc="Processing test images"):
#     filepath = row['filepath']    
#     image_title = f"{row['ImageID']}"
#         # Check if the feature file already exists
#     feature_file_path = f"ixi_storage/IXI_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     if os.path.exists(feature_file_path):
#         # If file exists, load the features from the file
#         features = np.load(feature_file_path)
        
#         features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
#         features_list.append(features)  # Flatten the features and add to the list
#         labels_list.append(row['Age'])  # Add the corresponding age label
#     else:
#         if os.path.exists(filepath):
#             try:
#                 # Load the NIfTI image
#                 nii_img = nib.load(filepath)

#                 # Get current orientation and reorient to RAS
#                 orig_ornt = io_orientation(nii_img.affine)
#                 ras_ornt = axcodes2ornt(("R", "A", "S"))
#                 ornt_trans = ornt_transform(orig_ornt, ras_ornt)

#                 data = nii_img.get_fdata()  # Load image data
#                 data = apply_orientation(data, ornt_trans)

#                 affine = nii_img.affine  # Affine transformation matrix

#                 data = crop_brain_volumes(data)

#                 # Resample the volume to 160 slices (if required)
#                 data = resample_nifti(data, target_slices=160)

#                 save_mid_slice (data, image_title, "ixi_storage/IXI_images")
#                 # Extract features for all slices
#                 features = []
#                 for slice_idx in range(data.shape[0]):
#                     slice_data = data[slice_idx, :, :]
#                     slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
#                     slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
#                     # Extract features using ViT
#                     with torch.no_grad():
#                         #outputs = model(slice_tensor)
#                         slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#                         features.append(slice_features)
#                 # Save extracted features
#                 features = np.array(features)
#                 np.save(feature_file_path, features)
#                 features_list.append(features)
#                 labels_list.append(row['Age'])  # Assuming 'Age' is the target

#             except Exception as e:
#                 print(f"Error processing {filepath}: {e}")
#         else:
#             print(f"File not found: {filepath}")


# # Directory to save processed images and features
# os.makedirs(f"abide_storage/ABIDEII_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
# for _, row in tqdm(df_abide.iterrows(), total=len(df_abide), desc="Processing test images"):
#     filepath = row['filepath']    
#     image_title = f"{row['ImageID'][7:]}"
#         # Check if the feature file already exists
#     feature_file_path = f"abide_storage/ABIDEII_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     # feature_file_path = f"ixi_storage/IXI_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     if os.path.exists(feature_file_path):
#         # If file exists, load the features from the file
#         features = np.load(feature_file_path)
        
#         features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
#         features_list.append(features)  # Flatten the features and add to the list
#         labels_list.append(row['Age'])  # Add the corresponding age label
#     else:
#         if os.path.exists(filepath):
#             try:
#                 # Load the NIfTI image
#                 nii_img = nib.load(filepath)

#                 # Get current orientation and reorient to RAS
#                 orig_ornt = io_orientation(nii_img.affine)
#                 ras_ornt = axcodes2ornt(("R", "A", "S"))
#                 ornt_trans = ornt_transform(orig_ornt, ras_ornt)

#                 data = nii_img.get_fdata()  # Load image data
#                 data = apply_orientation(data, ornt_trans)

#                 affine = nii_img.affine  # Affine transformation matrix

#                 # Resample the volume to 160 slices (if required)
#                 data = resample_nifti(data, target_slices=160)
#                 # Extract features for all slices
#                 features = []
#                 for slice_idx in range(data.shape[0]):
#                     slice_data = data[slice_idx, :, :]
#                     slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
#                     slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
#                     # Extract features using ViT
#                     with torch.no_grad():
#                         #outputs = model(slice_tensor)
#                         slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#                         features.append(slice_features)
#                 # Save extracted features
#                 features = np.array(features)
#                 np.save(feature_file_path, features)
#                 features_list.append(features)
#                 labels_list.append(row['Age'])  # Assuming 'Age' is the target

#             except Exception as e:
#                 print(f"Error processing {filepath}: {e}")
#         else:
#             print(f"File not found: {filepath}")

# os.makedirs(f"dlbs_storage/DLBS_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
# for _, row in tqdm(df_dlbs.iterrows(), total=len(df_dlbs), desc="Processing test images"):
#     filepath = row['filepath']    
#     image_title = f"{row['ImageID'][4:]}"
#         # Check if the feature file already exists
#     feature_file_path = f"dlbs_storage/DLBS_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     # feature_file_path = f"ixi_storage/IXI_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     if os.path.exists(feature_file_path):
#         # If file exists, load the features from the file
#         features = np.load(feature_file_path)
        
#         features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
#         features_list.append(features)  # Flatten the features and add to the list
#         labels_list.append(row['Age'])  # Add the corresponding age label
#     else:
#         if os.path.exists(filepath):
#             try:
#                 # Load the NIfTI image
#                 nii_img = nib.load(filepath)

#                 # Get current orientation and reorient to RAS
#                 orig_ornt = io_orientation(nii_img.affine)
#                 ras_ornt = axcodes2ornt(("R", "A", "S"))
#                 ornt_trans = ornt_transform(orig_ornt, ras_ornt)

#                 data = nii_img.get_fdata()  # Load image data
#                 data = apply_orientation(data, ornt_trans)

#                 affine = nii_img.affine  # Affine transformation matrix

#                 data = crop_brain_volumes(data)

#                 # Resample the volume to 160 slices (if required)
#                 data = resample_nifti(data, target_slices=160)

#                 save_mid_slice (data, image_title, "dlbs_storage/DLBS_images")
#                 # Extract features for all slices
#                 features = []
#                 for slice_idx in range(data.shape[0]):
#                     slice_data = data[slice_idx, :, :]
#                     slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
#                     slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
#                     # Extract features using ViT
#                     with torch.no_grad():
#                         #outputs = model(slice_tensor)
#                         slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#                         features.append(slice_features)
#                 # Save extracted features
#                 features = np.array(features)
#                 np.save(feature_file_path, features)
#                 features_list.append(features)
#                 labels_list.append(row['Age'])  # Assuming 'Age' is the target

#             except Exception as e:
#                 print(f"Error processing {filepath}: {e}")
#         else:
#             print(f"File not found: {filepath}")


# os.makedirs(f"cobre_storage/COBRE_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
# for _, row in tqdm(df_cobre.iterrows(), total=len(df_cobre), desc="Processing test images"):
#     filepath = row['filepath']    
#     image_title = f"{row['ImageID'][5:]}"
#         # Check if the feature file already exists
#     feature_file_path = f"cobre_storage/COBRE_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     # feature_file_path = f"ixi_storage/IXI_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     if os.path.exists(feature_file_path):
#         # If file exists, load the features from the file
#         features = np.load(feature_file_path)
        
#         features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
#         features_list.append(features)  # Flatten the features and add to the list
#         labels_list.append(row['Age'])  # Add the corresponding age label
#     else:
#         if os.path.exists(filepath):
#             try:
#                 # Load the NIfTI image
#                 nii_img = nib.load(filepath)

#                 # Get current orientation and reorient to RAS
#                 orig_ornt = io_orientation(nii_img.affine)
#                 ras_ornt = axcodes2ornt(("R", "A", "S"))
#                 ornt_trans = ornt_transform(orig_ornt, ras_ornt)

#                 data = nii_img.get_fdata()  # Load image data
#                 data = apply_orientation(data, ornt_trans)

#                 affine = nii_img.affine  # Affine transformation matrix

#                 # Resample the volume to 160 slices (if required)
#                 data = resample_nifti(data, target_slices=160)
#                 # Extract features for all slices
#                 features = []
#                 for slice_idx in range(data.shape[0]):
#                     slice_data = data[slice_idx, :, :]
#                     slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
#                     slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
#                     # Extract features using ViT
#                     with torch.no_grad():
#                         #outputs = model(slice_tensor)
#                         slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#                         features.append(slice_features)
#                 # Save extracted features
#                 features = np.array(features)
#                 np.save(feature_file_path, features)
#                 features_list.append(features)
#                 labels_list.append(row['Age'])  # Assuming 'Age' is the target

#             except Exception as e:
#                 print(f"Error processing {filepath}: {e}")
#         else:
#             print(f"File not found: {filepath}")

# os.makedirs(f"fcon1000_storage/fcon1000_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
# for _, row in tqdm(df_fcon.iterrows(), total=len(df_fcon), desc="Processing test images"):
#     filepath = row['filepath']    
#     image_title = f"{row['ImageID'][5:]}"
#         # Check if the feature file already exists
#     feature_file_path = f"fcon1000_storage/fcon1000_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     # feature_file_path = f"ixi_storage/IXI_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     if os.path.exists(feature_file_path):
#         # If file exists, load the features from the file
#         features = np.load(feature_file_path)
        
#         features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
#         features_list.append(features)  # Flatten the features and add to the list
#         labels_list.append(row['Age'])  # Add the corresponding age label
#     else:
#         if os.path.exists(filepath):
#             try:
#                 # Load the NIfTI image
#                 nii_img = nib.load(filepath)

#                 # Get current orientation and reorient to RAS
#                 orig_ornt = io_orientation(nii_img.affine)
#                 ras_ornt = axcodes2ornt(("R", "A", "S"))
#                 ornt_trans = ornt_transform(orig_ornt, ras_ornt)

#                 data = nii_img.get_fdata()  # Load image data
#                 data = apply_orientation(data, ornt_trans)

#                 affine = nii_img.affine  # Affine transformation matrix

#                 data = crop_brain_volumes(data)

#                 # Resample the volume to 160 slices (if required)
#                 data = resample_nifti(data, target_slices=160)

#                 save_mid_slice (data, image_title, "fcon1000_storage/fcon1000_images")
#                 # Extract features for all slices
#                 features = []
#                 for slice_idx in range(data.shape[0]):
#                     slice_data = data[slice_idx, :, :]
#                     slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
#                     slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
#                     # Extract features using ViT
#                     with torch.no_grad():
#                         #outputs = model(slice_tensor)
#                         slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#                         features.append(slice_features)
#                 # Save extracted features
#                 features = np.array(features)
#                 np.save(feature_file_path, features)
#                 features_list.append(features)
#                 labels_list.append(row['Age'])  # Assuming 'Age' is the target

#             except Exception as e:
#                 print(f"Error processing {filepath}: {e}")
#         else:
#             print(f"File not found: {filepath}")

os.makedirs(f"sald_storage/SALD_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
for _, row in tqdm(df_sald.iterrows(), total=len(df_sald), desc="Processing test images"):
    filepath = row['filepath']    
    image_title = f"{row['ImageID'][4:]}"
        # Check if the feature file already exists
    feature_file_path = f"sald_storage/SALD_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
    # feature_file_path = f"ixi_storage/IXI_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
    if os.path.exists(feature_file_path):
        # If file exists, load the features from the file
        features = np.load(feature_file_path)
        
        features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
        features_list.append(features)  # Flatten the features and add to the list
        labels_list.append(row['Age'])  # Add the corresponding age label
    else:
        if os.path.exists(filepath):
            try:
                # Load the NIfTI image
                nii_img = nib.load(filepath)

                # Get current orientation and reorient to RAS
                orig_ornt = io_orientation(nii_img.affine)
                ras_ornt = axcodes2ornt(("R", "A", "S"))
                ornt_trans = ornt_transform(orig_ornt, ras_ornt)

                data = nii_img.get_fdata()  # Load image data
                data = apply_orientation(data, ornt_trans)

                affine = nii_img.affine  # Affine transformation matrix
                data = crop_brain_volumes(data)
                # Resample the volume to 160 slices (if required)
                data = resample_nifti(data, target_slices=160)
                save_mid_slice (data, image_title, "sald_storage/SALD_images")
                # Extract features for all slices
                features = []
                for slice_idx in range(data.shape[0]):
                    slice_data = data[slice_idx, :, :]
                    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
                    slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
                    # Extract features using ViT
                    with torch.no_grad():
                        #outputs = model(slice_tensor)
                        slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                        features.append(slice_features)
                # Save extracted features
                features = np.array(features)
                np.save(feature_file_path, features)
                features_list.append(features)
                labels_list.append(row['Age'])  # Assuming 'Age' is the target

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")

# os.makedirs(f"corr_storage/CORR_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
# for _, row in tqdm(df_corr.iterrows(), total=len(df_corr), desc="Processing test images"):
#     filepath = row['filepath']    
#     image_title = f"{row['ImageID'][5:]}"
#         # Check if the feature file already exists
#     feature_file_path = f"corr_storage/CORR_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     # feature_file_path = f"ixi_storage/IXI_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     if os.path.exists(feature_file_path):
#         # If file exists, load the features from the file
#         features = np.load(feature_file_path)
        
#         features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
#         features_list.append(features)  # Flatten the features and add to the list
#         labels_list.append(row['Age'])  # Add the corresponding age label
#     else:
#         if os.path.exists(filepath):
#             try:
#                 # Load the NIfTI image
#                 nii_img = nib.load(filepath)

#                 # Get current orientation and reorient to RAS
#                 orig_ornt = io_orientation(nii_img.affine)
#                 ras_ornt = axcodes2ornt(("R", "A", "S"))
#                 ornt_trans = ornt_transform(orig_ornt, ras_ornt)

#                 data = nii_img.get_fdata()  # Load image data
#                 data = apply_orientation(data, ornt_trans)

#                 affine = nii_img.affine  # Affine transformation matrix

#                 # Resample the volume to 160 slices (if required)
#                 data = resample_nifti(data, target_slices=160)
#                 # Extract features for all slices
#                 features = []
#                 for slice_idx in range(data.shape[0]):
#                     slice_data = data[slice_idx, :, :]
#                     slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
#                     slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
#                     # Extract features using ViT
#                     with torch.no_grad():
#                         #outputs = model(slice_tensor)
#                         slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#                         features.append(slice_features)
#                 # Save extracted features
#                 features = np.array(features)
#                 np.save(feature_file_path, features)
#                 features_list.append(features)
#                 labels_list.append(row['Age'])  # Assuming 'Age' is the target

#             except Exception as e:
#                 print(f"Error processing {filepath}: {e}")
#         else:
#             print(f"File not found: {filepath}")


# os.makedirs(f"oasis1_storage/oasis1_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
# for _, row in tqdm(df_oas1.iterrows(), total=len(df_oas1), desc="Processing test images"):
#     filepath = row['filepath']    
#     image_title = f"{row['ImageID']}"
#         # Check if the feature file already exists
#     feature_file_path = f"oasis1_storage/oasis1_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     if os.path.exists(feature_file_path):
#         # If file exists, load the features from the file
#         features = np.load(feature_file_path)
        
#         features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
#         features_list.append(features)  # Flatten the features and add to the list
#         labels_list.append(row['Age'])  # Add the corresponding age label
#     else:
#         if os.path.exists(filepath):
#             try:
#                 # Load the NIfTI image
#                 nii_img = nib.load(filepath)

#                 # Get current orientation and reorient to RAS
#                 orig_ornt = io_orientation(nii_img.affine)
#                 ras_ornt = axcodes2ornt(("R", "A", "S"))
#                 ornt_trans = ornt_transform(orig_ornt, ras_ornt)

#                 data = nii_img.get_fdata()  # Load image data
#                 data = apply_orientation(data, ornt_trans)

#                 affine = nii_img.affine  # Affine transformation matrix

#                 # Resample the volume to 160 slices (if required)
#                 data = resample_nifti(data, target_slices=160)
#                 # Extract features for all slices
#                 features = []
#                 for slice_idx in range(data.shape[0]):
#                     slice_data = data[slice_idx, :, :]
#                     slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
#                     slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
#                     # Extract features using ViT
#                     with torch.no_grad():
#                         #outputs = model(slice_tensor)
#                         slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#                         features.append(slice_features)
#                 # Save extracted features
#                 features = np.array(features)
#                 np.save(feature_file_path, features)
#                 features_list.append(features)
#                 labels_list.append(row['Age'])  # Assuming 'Age' is the target

#             except Exception as e:
#                 print(f"Error processing {filepath}: {e}")
#         else:
#             print(f"File not found: {filepath}")

# os.makedirs(f"camcan_storage/CamCAN_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
# for _, row in tqdm(df_camcan.iterrows(), total=len(df_camcan), desc="Processing test images"):
#     filepath = row['filepath']    
#     image_title = f"{row['ImageID']}"
#         # Check if the feature file already exists
#     feature_file_path = f"camcan_storage/CamCAN_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     if os.path.exists(feature_file_path):
#         # If file exists, load the features from the file
#         features = np.load(feature_file_path)
        
#         features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
#         features_list.append(features)  # Flatten the features and add to the list
#         labels_list.append(row['Age'])  # Add the corresponding age label
#     else:
#         if os.path.exists(filepath):
#             try:
#                 # Load the NIfTI image
#                 nii_img = nib.load(filepath)

#                 # Get current orientation and reorient to RAS
#                 orig_ornt = io_orientation(nii_img.affine)
#                 ras_ornt = axcodes2ornt(("R", "A", "S"))
#                 ornt_trans = ornt_transform(orig_ornt, ras_ornt)

#                 data = nii_img.get_fdata()  # Load image data
#                 data = apply_orientation(data, ornt_trans)

#                 affine = nii_img.affine  # Affine transformation matrix

#                 data = crop_brain_volumes(data)

#                 # Resample the volume to 160 slices (if required)
#                 data = resample_nifti(data, target_slices=160)

#                 save_mid_slice (data, image_title, "camcan_storage/CamCAN_images")
#                 # Extract features for all slices
#                 features = []
#                 for slice_idx in range(data.shape[0]):
#                     slice_data = data[slice_idx, :, :]
#                     slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
#                     slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
#                     # Extract features using ViT
#                     with torch.no_grad():
#                         slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#                         features.append(slice_features)
#                 # Save extracted features
#                 features = np.array(features)
#                 np.save(feature_file_path, features)
#                 features_list.append(features)
#                 labels_list.append(row['Age'])  # Assuming 'Age' is the target

#             except Exception as e:
#                 print(f"Error processing {filepath}: {e}")
#         else:
#             print(f"File not found: {filepath}")


# os.makedirs(f"nimh_storage/nimh_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
# for _, row in tqdm(df_nimh.iterrows(), total=len(df_nimh), desc="Processing test images"):
#     filepath = row['filepath']    
#     image_title = f"{row['ImageID']}"
#         # Check if the feature file already exists
#     feature_file_path = f"nimh_storage/nimh_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     # feature_file_path = f"ixi_storage/IXI_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     if os.path.exists(feature_file_path):
#         # If file exists, load the features from the file
#         features = np.load(feature_file_path)
        
#         features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
#         features_list.append(features)  # Flatten the features and add to the list
#         labels_list.append(row['Age'])  # Add the corresponding age label
#     else:
#         if os.path.exists(filepath):
#             try:
#                 # Load the NIfTI image
#                 nii_img = nib.load(filepath)

#                 # Get current orientation and reorient to RAS
#                 orig_ornt = io_orientation(nii_img.affine)
#                 ras_ornt = axcodes2ornt(("R", "A", "S"))
#                 ornt_trans = ornt_transform(orig_ornt, ras_ornt)

#                 data = nii_img.get_fdata()  # Load image data
#                 data = apply_orientation(data, ornt_trans)

#                 affine = nii_img.affine  # Affine transformation matrix
#                 data = crop_brain_volumes(data)
#                 # Resample the volume to 160 slices (if required)
#                 data = resample_nifti(data, target_slices=160)
#                 save_mid_slice (data, image_title, "nimh_storage/nimh_images")
#                 # Extract features for all slices
#                 features = []
#                 for slice_idx in range(data.shape[0]):
#                     slice_data = data[slice_idx, :, :]
#                     slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
#                     slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
#                     # Extract features using ViT
#                     with torch.no_grad():
#                         #outputs = model(slice_tensor)
#                         slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#                         features.append(slice_features)
#                 # Save extracted features
#                 features = np.array(features)
#                 np.save(feature_file_path, features)
#                 features_list.append(features)
#                 labels_list.append(row['Age'])  # Assuming 'Age' is the target

#             except Exception as e:
#                 print(f"Error processing {filepath}: {e}")
#         else:
#             print(f"File not found: {filepath}")

# os.makedirs(f"bold_storage/bold_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
# for _, row in tqdm(df_bold.iterrows(), total=len(df_bold), desc="Processing test images"):
#     filepath = row['filepath']    
#     image_title = f"{row['ImageID']}"
#         # Check if the feature file already exists
#     feature_file_path = f"bold_storage/bold_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     # feature_file_path = f"ixi_storage/IXI_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
#     if os.path.exists(feature_file_path):
#         # If file exists, load the features from the file
#         features = np.load(feature_file_path)
        
#         features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
#         features_list.append(features)  # Flatten the features and add to the list
#         labels_list.append(row['Age'])  # Add the corresponding age label
#     else:
#         if os.path.exists(filepath):
#             try:
#                 # Load the NIfTI image
#                 nii_img = nib.load(filepath)

#                 # Get current orientation and reorient to RAS
#                 orig_ornt = io_orientation(nii_img.affine)
#                 ras_ornt = axcodes2ornt(("R", "A", "S"))
#                 ornt_trans = ornt_transform(orig_ornt, ras_ornt)

#                 data = nii_img.get_fdata()  # Load image data
#                 data = apply_orientation(data, ornt_trans)

#                 affine = nii_img.affine  # Affine transformation matrix
#                 data = crop_brain_volumes(data)
#                 # Resample the volume to 160 slices (if required)
#                 data = resample_nifti(data, target_slices=160)
#                 save_mid_slice (data, image_title, "bold_storage/bold_images")
#                 # Extract features for all slices
#                 features = []
#                 for slice_idx in range(data.shape[0]):
#                     slice_data = data[slice_idx, :, :]
#                     slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
#                     slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
#                     # Extract features using ViT
#                     with torch.no_grad():
#                         #outputs = model(slice_tensor)
#                         slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#                         features.append(slice_features)
#                 # Save extracted features
#                 features = np.array(features)
#                 np.save(feature_file_path, features)
#                 features_list.append(features)
#                 labels_list.append(row['Age'])  # Assuming 'Age' is the target

#             except Exception as e:
#                 print(f"Error processing {filepath}: {e}")
#         else:
#             print(f"File not found: {filepath}")



os.makedirs(f"truecrime_storage/truecrime_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
for _, row in tqdm(df_truecrime.iterrows(), total=len(df_truecrime), desc="Processing test images"):
    filepath = row['filepath']    
    image_title = f"{row['ImageID'][3:]}"
        # Check if the feature file already exists
    feature_file_path = f"truecrime_storage/truecrime_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
    # feature_file_path = f"ixi_storage/IXI_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
    if os.path.exists(feature_file_path):
        # If file exists, load the features from the file
        features = np.load(feature_file_path)
        
        features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
        features_list.append(features)  # Flatten the features and add to the list
        labels_list.append(row['Age'])  # Add the corresponding age label
    else:
        if os.path.exists(filepath):
            try:
                # Load the NIfTI image
                nii_img = nib.load(filepath)

                # Get current orientation and reorient to RAS
                orig_ornt = io_orientation(nii_img.affine)
                ras_ornt = axcodes2ornt(("R", "A", "S"))
                ornt_trans = ornt_transform(orig_ornt, ras_ornt)

                data = nii_img.get_fdata()  # Load image data
                data = apply_orientation(data, ornt_trans)

                affine = nii_img.affine  # Affine transformation matrix
                data = crop_brain_volumes(data)
                # Resample the volume to 160 slices (if required)
                data = resample_nifti(data, target_slices=160)
                save_mid_slice (data, image_title, "truecrime_storage/truecrime_images")
                # Extract features for all slices
                features = []
                for slice_idx in range(data.shape[0]):
                    slice_data = data[slice_idx, :, :]
                    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
                    slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
                    # Extract features using ViT
                    with torch.no_grad():
                        #outputs = model(slice_tensor)
                        slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                        features.append(slice_features)
                # Save extracted features
                features = np.array(features)
                np.save(feature_file_path, features)
                features_list.append(features)
                labels_list.append(row['Age'])  # Assuming 'Age' is the target

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")


os.makedirs(f"agerisk_storage/agerisk_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
for _, row in tqdm(df_agerisk.iterrows(), total=len(df_agerisk), desc="Processing test images"):
    filepath = row['filepath']    
    image_title = f"{row['ImageID'][8:]}"
        # Check if the feature file already exists
    feature_file_path = f"agerisk_storage/agerisk_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
    # feature_file_path = f"ixi_storage/IXI_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
    if os.path.exists(feature_file_path):
        # If file exists, load the features from the file
        features = np.load(feature_file_path)
        
        features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
        features_list.append(features)  # Flatten the features and add to the list
        labels_list.append(row['Age'])  # Add the corresponding age label
    else:
        if os.path.exists(filepath):
            try:
                # Load the NIfTI image
                nii_img = nib.load(filepath)

                # Get current orientation and reorient to RAS
                orig_ornt = io_orientation(nii_img.affine)
                ras_ornt = axcodes2ornt(("R", "A", "S"))
                ornt_trans = ornt_transform(orig_ornt, ras_ornt)

                data = nii_img.get_fdata()  # Load image data
                data = apply_orientation(data, ornt_trans)

                affine = nii_img.affine  # Affine transformation matrix
                data = crop_brain_volumes(data)
                # Resample the volume to 160 slices (if required)
                data = resample_nifti(data, target_slices=160)
                save_mid_slice (data, image_title, "agerisk_storage/agerisk_images")
                # Extract features for all slices
                features = []
                for slice_idx in range(data.shape[0]):
                    slice_data = data[slice_idx, :, :]
                    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
                    slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
                    # Extract features using ViT
                    with torch.no_grad():
                        #outputs = model(slice_tensor)
                        slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                        features.append(slice_features)
                # Save extracted features
                features = np.array(features)
                np.save(feature_file_path, features)
                features_list.append(features)
                labels_list.append(row['Age'])  # Assuming 'Age' is the target

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")



os.makedirs(f"sudmex_storage/sudmex_features/train_dora_e{vit_train_epochs}_s{slice_count}/", exist_ok=True)
for _, row in tqdm(df_sudmex.iterrows(), total=len(df_sudmex), desc="Processing test images"):
    filepath = row['filepath']    
    image_title = f"{row['ImageID'][7:]}"
        # Check if the feature file already exists
    feature_file_path = f"sudmex_storage/sudmex_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
    # feature_file_path = f"ixi_storage/IXI_features/train_dora_e{vit_train_epochs}_s{slice_count}/{image_title}_features.npy"
    if os.path.exists(feature_file_path):
        # If file exists, load the features from the file
        features = np.load(feature_file_path)
        
        features =  features[len(features) // 2 - roi//2 : len(features) // 2 + roi//2]
        features_list.append(features)  # Flatten the features and add to the list
        labels_list.append(row['Age'])  # Add the corresponding age label
    else:
        if os.path.exists(filepath):
            try:
                # Load the NIfTI image
                nii_img = nib.load(filepath)

                # Get current orientation and reorient to RAS
                orig_ornt = io_orientation(nii_img.affine)
                ras_ornt = axcodes2ornt(("R", "A", "S"))
                ornt_trans = ornt_transform(orig_ornt, ras_ornt)

                data = nii_img.get_fdata()  # Load image data
                data = apply_orientation(data, ornt_trans)

                affine = nii_img.affine  # Affine transformation matrix
                data = crop_brain_volumes(data)
                # Resample the volume to 160 slices (if required)
                data = resample_nifti(data, target_slices=160)
                save_mid_slice (data, image_title, "sudmex_storage/sudmex_images")
                # Extract features for all slices
                features = []
                for slice_idx in range(data.shape[0]):
                    slice_data = data[slice_idx, :, :]
                    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))  # Normalize
                    
                    slice_tensor = transform(slice_data).unsqueeze(0).to(device)
                    
                    # Extract features using ViT
                    with torch.no_grad():
                        #outputs = model(slice_tensor)
                        slice_features = model.vit(slice_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                        features.append(slice_features)
                # Save extracted features
                features = np.array(features)
                np.save(feature_file_path, features)
                features_list.append(features)
                labels_list.append(row['Age'])  # Assuming 'Age' is the target

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")

batch_size = 1

# print (features_list)
print (features_list[0].shape)

# Create Dataset and DataLoader
val_dataset = ADNIDataset(features_list, sex_encoded, age_list)
# Store the indices of the validation dataset
# val_indices = val_dataset.indices

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Tracking outputs for validation samples
val_outputs = {}

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###########################################
# THIS IS WHERE YOU CHOOSE A MODEL TO TEST 
###########################################

import importlib

# Assuming sys.argv[1] is the module name
module_name = sys.argv[1]  # Example: "my_model"
class_name = "AgePredictionCNN"  # The class you want to import

try:
    # Dynamically import the module
    module = importlib.import_module(module_name)
    
    # Dynamically get the class
    AgePredictionCNN = getattr(module, class_name)
    
    print(f"Successfully imported {class_name} from {module_name}.")

except ImportError:
    print(f"Module {module_name} could not be imported.")
except AttributeError:
    print(f"{class_name} does not exist in {module_name}.")

##############################
# MODEL IMPORTED DYNAMICALLY
##############################


print (features_list[0].shape)
model = AgePredictionCNN((1, features_list[0].shape[0], features_list[0].shape[1])).to(device)
criterion = nn.MSELoss()  # MAE Loss
eval_crit = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
best_loss = np.inf  # Initialize the best loss to infinity
start_epoch = 0


load_saved = sys.argv[2] # "last, "best"
if load_saved != "none":
    with open(f"model_dumps/mix/{sys.argv[1]}_best_model_with_metadata.pkl", "rb") as f:
        checkpoint = pickle.load(f)
    best_loss = checkpoint["loss"]

    # Load the checkpoint
    with open(f"model_dumps/mix/{sys.argv[1]}_{load_saved}_model_with_metadata.pkl", "rb") as f:
        checkpoint = pickle.load(f)

    # Restore model and optimizer state
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    # Restore RNG states
    torch.set_rng_state(checkpoint["t_rng_st"])
    np.random.set_state(checkpoint["n_rng_st"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_st"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_st"])

    # Retrieve metadata
    start_epoch = checkpoint["epoch"] + 1
    loaded_loss = checkpoint["loss"]

    print(f"Loaded model from epoch {start_epoch} with validation loss {loaded_loss:.4f}, best loss {best_loss:.4f}")

    # Perform validation before resuming training
    model.eval()
    val_loss = 0.0
    val_outputs = {}  # Ensure this is initialized
    with torch.no_grad():
        current_index = 0
        for features, sex, age in val_loader:
            features = features.unsqueeze(1).to(device)
            sex = sex.to(device)
            age = age.to(device)

            outputs = model(features, sex)
            loss = eval_crit(outputs.squeeze(), age)
            val_loss += loss.item()

            for i in range(outputs.size(0)):
                val_outputs[current_index] = outputs[i].item()
                current_index += 1

    val_loss /= len(val_loader)
    print(f"Validation Loss after loading: {val_loss:.4f}")

    # Save predictions and create DataFrame
    max_index = max(val_outputs.keys())
    df_pred = pd.DataFrame(index=range(max_index + 1), columns=["Predicted_Age"])
    for index, value in val_outputs.items():
        df_pred.loc[index, "Predicted_Age"] = value

    df1 = df.copy()
    df1['Predicted_Age'] = df_pred['Predicted_Age']
    test_df = df1.loc[val_outputs.keys()]
    test_df.to_csv(f"model_dumps/mix/{sys.argv[1]}_predicted_ages_test.csv")
    # show_problematic(test_df)

        # Map unique first 4 characters of ImageID to color codes
    unique_groups = test_df['ImageID'].str[:3].unique()
    group_to_color = {group: i for i, group in enumerate(unique_groups)}

    # Assign colors based on the mapping
    cmap = plt.get_cmap('tab10')  # Change colormap as desired
    colors = [cmap(group_to_color[group]) for group in test_df['ImageID'].str[:3]]

    # Check that the predictions have been added to the DataFrame
    # Plot Age vs. Predicted Age
    plt.figure(figsize=(8, 6))
    plt.scatter(test_df['Age'], test_df['Predicted_Age'], color=colors, label='Predicted vs Actual')
    # plt.plot(test_df['Age'], test_df['Age'], color='red', linestyle='--', label='Perfect Prediction')  # Optional: Line of perfect prediction
    # Add legend for colors based on ImageID groups
    handles = [plt.Line2D([0], [0], marker='o', color=cmap(i), linestyle='', markersize=10) 
            for i, group in enumerate(unique_groups)]
    plt.legend(handles, unique_groups, title="ImageID Groups")
    plt.xlabel('Age')
    plt.ylabel('Predicted Age')
    plt.title('Age vs Predicted Age')
    plt.grid(True)
    plt.savefig(f"model_dumps/mix/plots/vit_cnn_{sys.argv[1]}_TEST.png")