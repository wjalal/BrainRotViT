import os
from deepbrain import Extractor
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

import tensorflow as tf

# Check if GPU is available
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPUs:", gpu_devices)

session = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# Define the input and output directories
input_dir = "abide_storage/ABIDEII"
output_dir = "abide_storage/ABIDEII_stripped"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize the brain extractor
ext = Extractor()

# Process all .nii.gz files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".nii.gz"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".nii.gz", ".stripped.nii.gz"))
        
        # Skip processing if the output file already exists
        if os.path.exists(output_path):
            print(f"Output already exists, skipping: {output_path}")
            continue
        
        print(f"Processing: {input_path}")
        
        # Load the NIfTI image
        nii_img = nib.load(input_path)
        data = nii_img.get_fdata()

        # Run the extractor and get the mask
        prob = ext.run(data)
        brain_mask = prob > 0.5  # Threshold for mask

        # Apply the mask to extract the brain
        brain_only = data * brain_mask

        # Save the brain-extracted image
        brain_nii = nib.Nifti1Image(brain_only, nii_img.affine, nii_img.header)
        nib.save(brain_nii, output_path)
        
        print(f"Saved stripped image: {output_path}")

print("Processing complete.")
