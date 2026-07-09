import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
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


# -------------------------------
# Input and output paths
# -------------------------------
in_path = "attention_3d_mapped_backprop_dora.nii.gz"
out_path = "attention_3d_mapped_backprop_dora_cropped_centered_resized.nii.gz"

# Load image
img = nib.load(in_path)
data = img.get_fdata()
affine = img.affine.copy()
header = img.header.copy()

# Step 0: Resample sagittal slices
target_sagittal_slices = 146
orig_slices = data.shape[0]
zoom_factor_sagittal = target_sagittal_slices / orig_slices
data = zoom(data, (zoom_factor_sagittal, 1, 1), order=1)

# Step 1: Crop
data = crop_brain_volumes(data)

# Step 2: Resize in-plane slices
target_size = (178, 141)
orig_y, orig_z = data.shape[1], data.shape[2]
zoom_factors = (1, target_size[0] / orig_y, target_size[1] / orig_z)
resized_data = zoom(data, zoom_factors, order=1)

# Step 3: Center affine
shape = np.array(resized_data.shape)
voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)) if not np.allclose(affine, np.eye(4)) else np.ones(3)
center_shift = -(shape[:3] / 2) * voxel_sizes
new_affine = affine.copy()
new_affine[:3, 3] = center_shift

# Step 4: Save
new_img = nib.Nifti1Image(resized_data, new_affine, header=header)
nib.save(new_img, out_path)
print(f"Cropped, centered & resized image saved to: {out_path}")
