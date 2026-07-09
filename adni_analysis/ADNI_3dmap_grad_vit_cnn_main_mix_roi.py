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
csv_path = "../adni_storage/adni_brainrotnet_metadata.csv"
df_adni = pd.read_csv(csv_path)
# df = df.sample(n=1000, random_state=69420)
# Add a new column 'filepath' with the constructed file paths
df_adni['filepath'] = df_adni.apply(
    lambda row: f"../adni_storage/ADNI_nii_gz_bias_corrected/I{row['ImageID'][4:]}_{row['SubjectID']}.stripped.N4.nii.gz",
    axis=1
)

row_to_top = df_adni.iloc[[4190]]        # row 4190 as a DataFrame
rest = df_adni.drop(df_adni.index[4190])      # everything else

# Concatenate with row_to_top first
df_adni = pd.concat([row_to_top, rest], ignore_index=True)

df = pd.concat ([
                 df_adni[['ImageID', 'Sex', 'Age', 'filepath']], 
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


import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# 3. Load trained model
# ----------------------

# Example shape from training
input_shape = (1, 160, 768)  # (channels, H, W)

model = AgePredictionCNN(input_shape=input_shape).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)


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


model.eval()

# ----------------------
# 2. Guided Backprop helper
# ----------------------
class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.forward_relu_outputs = []
        self.hooks = []

        def relu_forward_hook(module, input, output):
            self.forward_relu_outputs.append(output)

        def relu_backward_hook(module, grad_input, grad_output):
            # Only allow positive gradients to flow back
            forward_output = self.forward_relu_outputs.pop()
            forward_mask = (forward_output > 0).float()
            modified_grad_output = grad_output[0].clamp(min=0.0)
            return (modified_grad_output * forward_mask,)

        # Replace all ReLUs/SiLUs with hooks
        for module in self.model.modules():
            if isinstance(module, nn.ReLU) or isinstance(module, nn.SiLU):
                self.hooks.append(module.register_forward_hook(relu_forward_hook))
                self.hooks.append(module.register_full_backward_hook(relu_backward_hook))

    def generate(self, input_tensor, sex_tensor):
        input_tensor.requires_grad = True

        output = self.model(input_tensor, sex_tensor)  # forward
        pred = output.squeeze()

        self.model.zero_grad()
        pred.backward(retain_graph=True)

        # Guided gradients wrt input
        guided_grads = input_tensor.grad.detach().cpu().numpy()[0, 0]  # (160,768)

        # Normalize for visualization
        guided_grads = (guided_grads - guided_grads.min()) / (guided_grads.max() - guided_grads.min() + 1e-8)
        return guided_grads, pred.item()

    def close(self):
        for hook in self.hooks:
            hook.remove()


# ----------------------
# 3. Guided Backprop instantiation
# ----------------------
guided_bp = GuidedBackprop(model)

# ----------------------
# 4. Example input (same as GradCAM)
# ----------------------
features = torch.randn(1, 1, 160, 768).to(device)  # fake example
sex = torch.tensor([1.0]).to(device)               # male/female encoding

# ----------------------
# 5. Generate Guided Backprop
# ----------------------
cnn_guided_map, pred = guided_bp.generate(features, sex)
print(f"Predicted Age: {pred:.2f}")

# ----------------------
# 6. Plot guided backprop saliency (rotated for better orientation)
# ----------------------
plt.figure(figsize=(6, 12))

# Transpose to swap axes: slices on x-axis, embedding on y-axis
plt.imshow(cnn_guided_map.T, cmap="jet", aspect="auto", origin="lower")

plt.colorbar(label="Guided Backprop Gradient")
plt.title("Guided Backpropagation Saliency Map (rotated)")
plt.xlabel("Sagittal slice index (160)")
plt.ylabel("Embedding dimension (768)")
# Save the figure
plt.savefig(f"gbp_{sys.argv[1]}_{sys.argv[2]}.png", bbox_inches='tight', dpi=300)
plt.show()


print (cnn_guided_map.shape)









from transformers import ViTForImageClassification
# Load ViT model
num_classes = df['Age_Group'].nunique()  # Number of unique Age_Groups
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True, 
    attn_implementation = "eager",
)

model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

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
checkpoint_path = "model_dumps/vit_train_checkpoint.pth"
start_epoch = 0

start_epoch = load_checkpoint(path=checkpoint_path)

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

import torch
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------
# Utility: Brain mask preprocessing
# --------------------------------------------------------
def apply_brain_mask(slice_tensor, brain_thresh=0.05):
    """
    Replaces background (pixels < brain_thresh * max) with slice mean.
    
    Args:
        slice_tensor: [1, 3, H, W] float tensor
        brain_thresh: fraction of max intensity for mask cutoff
    
    Returns:
        slice_tensor_masked: [1, 3, H, W]
    """
    img = slice_tensor.clone()
    gray = img.mean(dim=1, keepdim=True)  # [1,1,H,W]
    max_val = gray.max()
    mask = (gray > brain_thresh * max_val).float()

    # compute mean over brain region
    brain_mean = (gray * mask).sum() / (mask.sum() + 1e-8)

    # replace background with brain mean
    img = img * mask + brain_mean * (1 - mask)
    return img

import torch
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------
# 1. Map CNN saliency vector -> ViT patch tokens (with pre-hoc masking)
# --------------------------------------------------------
def embedding_importance_to_patch_map(vit_model, slice_tensor, G_vec, repr_type="mean", brain_thresh=0.05):
    """
    Args:
        vit_model: HuggingFace ViT or ViTForImageClassification
        slice_tensor: [1, 3, H, W] on device
        G_vec: numpy or torch, shape (D,)
        repr_type: how to combine token info (unused here, but could extend)
        brain_thresh: fraction of max intensity for masking

    Returns:
        patch_scores: (num_patches,) numpy array
        cls_attn: (num_patches,) numpy array from CLS->patch attention
    """
    # ---- pre-hoc brain masking ----
    # slice_tensor = apply_brain_mask(slice_tensor, brain_thresh=brain_thresh)

    vit_model.eval()
    with torch.no_grad():
        outputs = vit_model(slice_tensor, output_attentions=True)
        last_hidden = outputs.last_hidden_state  # [1, num_tokens, D]

    patch_tokens = last_hidden[0, 1:, :]  # drop CLS → (num_patches, D)

    # convert guided CNN vector to tensor
    if not torch.is_tensor(G_vec):
        G_vec_t = torch.tensor(G_vec, dtype=patch_tokens.dtype, device=patch_tokens.device)
    else:
        G_vec_t = G_vec.to(patch_tokens.device)

    # project CNN importance vector onto patch tokens
    patch_scores = (patch_tokens @ G_vec_t).detach().cpu().numpy()  # (num_patches,)

    # normalize to [0,1]
    if patch_scores.max() > patch_scores.min():
        patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min() + 1e-8)
    else:
        patch_scores = np.zeros_like(patch_scores)

    # CLS→patch attention
    cls_attn = outputs.attentions[-1][0, :, 0, 1:].mean(0).detach().cpu().numpy()  # (num_patches,)
    if cls_attn.max() > cls_attn.min():
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    else:
        cls_attn = np.zeros_like(cls_attn)

    return patch_scores, cls_attn


# --------------------------------------------------------
# 2. Convert patch scores to full-resolution image (with post-hoc masking)
# --------------------------------------------------------
def patch_map_to_image(patch_scores, cls_attn, slice_tensor, combine_mode="mul", brain_thresh=0.05):
    """
    Args:
        patch_scores: (num_patches,) numpy
        cls_attn: (num_patches,) numpy
        slice_tensor: [1, 3, H, W]
        combine_mode: 'mul', 'add', or 'patch_only'
        brain_thresh: threshold for masking background

    Returns:
        heat_img: (H, W) numpy, normalized to 0..1
    """
    H, W = slice_tensor.shape[-2:]
    num_patches = patch_scores.shape[0]
    grid = int(np.sqrt(num_patches))  # e.g. 14 for 224x224 w/16x16 patches

    # fuse scores
    if combine_mode == "mul":
        fused = patch_scores * cls_attn
    elif combine_mode == "add":
        fused = patch_scores + cls_attn
    elif combine_mode == "patch_only":
        fused = patch_scores
    else:
        raise ValueError(f"Unknown combine_mode: {combine_mode}")

    fused_grid = fused.reshape(grid, grid)

    # ---- post-hoc brain masking ----
    slice_img = slice_tensor.squeeze(0).cpu().numpy()  # (3, H, W)
    slice_gray = slice_img.mean(0)
    brain_mask = (slice_gray > brain_thresh).astype(float)

    # resize mask to patch grid
    brain_mask_patch = F.interpolate(
        torch.tensor(brain_mask)[None, None].float(),
        size=(grid, grid),
        mode="bilinear",
        align_corners=False
    ).squeeze().numpy()

    fused_grid *= brain_mask_patch  # suppress background patches
    # -------------------------------
    # upsample to slice resolution
    up = F.interpolate(
        torch.tensor(fused_grid)[None, None].float(),
        size=(H, W),
        mode="bilinear",
        align_corners=False
    ).squeeze().cpu().numpy()

    # normalize again
    if up.max() > up.min():
        up = (up - up.min()) / (up.max() - up.min() + 1e-8)
    else:
        up = np.zeros_like(up)

    return up

# ----------------------
# Full builder (uses above functions)
# ----------------------
def build_3d_interpretation_with_backprop(df_val, slice_count, model_vit_for_class, cnn_guided_map,
                                         device, transform, out_slice_dir="slice_maps",
                                         repr_type='cls', combine_mode='multiply',
                                         max_subjects_per_slice=50, layer_idx=-1, head_fusion='mean'):
    vit_backbone = model_vit_for_class.vit  # backbone (ViTModel)
    vit_backbone.to(device).eval()

    os.makedirs(out_slice_dir, exist_ok=True)
    per_slice_images = []

    for s in tqdm(range(slice_count), desc="Slices"):
        G_vec = cnn_guided_map[s]  # (D,)
        # print(f"\nSlice {s}: G_vec mean {G_vec.mean():.4f}, std {G_vec.std():.4f}")

        vit_patch_maps = []
        used = 0

        for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc=f"Slice {s} subjects", leave=False):
            if max_subjects_per_slice and used >= max_subjects_per_slice:
                break
            path = row['filepath']
            if not os.path.exists(path):
                continue
            try:
                nii_img = nib.load(path)

                # Reorient to RAS
                orig_ornt = io_orientation(nii_img.affine)
                ras_ornt = axcodes2ornt(("R", "A", "S"))
                ornt_trans = ornt_transform(orig_ornt, ras_ornt)
                
                data = nii_img.get_fdata()  # Load image data
                data = apply_orientation(data, ornt_trans)

                # Crop & resample
                data = crop_brain_volumes(data)
                data = resample_nifti(data, target_slices=160)
                if s >= data.shape[0]:
                    continue
                slice_data = (data[s] - data[s].min()) / (data[s].max() - data[s].min() + 1e-8)
                slice_tensor = transform(slice_data).unsqueeze(0).to(device).float()

                patch_scores, cls_attn = embedding_importance_to_patch_map(
                    vit_backbone, slice_tensor, G_vec, repr_type=repr_type
                )
                # print("   patch_scores:", patch_scores.min().item(), patch_scores.max().item(),
                    # "cls_attn:", cls_attn.min().item(), cls_attn.max().item())

                heat_img = patch_map_to_image(patch_scores, cls_attn, slice_tensor, combine_mode=combine_mode)
                # print("   heat_img stats:", heat_img.min(), heat_img.max(), "shape:", heat_img.shape)

                vit_patch_maps.append(heat_img)
                used += 1
            except Exception as e:
                tqdm.write(f"Skipping {path}: {e}")
                continue

        if len(vit_patch_maps) == 0:
            H, W = 224, 224
            per_slice_images.append(np.zeros((H, W)))
            continue

        mean_map = np.stack(vit_patch_maps, axis=0).mean(axis=0)
        print("Slice", s, "mean_map stats before norm:", mean_map.min(), mean_map.max())

        if mean_map.max() > mean_map.min():
            mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-8)
        else:
            mean_map = np.zeros_like(mean_map)

        per_slice_images.append(mean_map)
        np.save(os.path.join(out_slice_dir, f"slice_{s:03d}.npy"), mean_map)
        plt.imsave(os.path.join(out_slice_dir, f"slice_{s:03d}.png"), mean_map, cmap='hot')



    # stack into 3D volume: (slices, H, W)
    attention_volume = np.stack(per_slice_images, axis=0)
    if attention_volume.max() > attention_volume.min():
        attention_volume = (attention_volume - attention_volume.min()) / (attention_volume.max() - attention_volume.min() + 1e-8)
    else:
        attention_volume = np.zeros_like(attention_volume)

    # save NIfTI
    nif = nib.Nifti1Image(attention_volume, np.eye(4))
    nib.save(nif, os.path.join(out_slice_dir, "attention_3d_mapped_backprop.nii.gz"))
    print("Saved attention_3d_mapped_backprop.nii.gz")

    return attention_volume

# assume cnn_guided_map is numpy (160,768)
att_vol = build_3d_interpretation_with_backprop(df_val, slice_count=160, model_vit_for_class=model,
                                                cnn_guided_map=cnn_guided_map, device=device,
                                                transform=transform, out_slice_dir="maps_out",
                                                repr_type='mean', combine_mode='mul',
                                                max_subjects_per_slice=100)