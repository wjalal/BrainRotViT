#!/bin/bash

# Initialize count
count=0

# Define source and target directories
source_folder="adni_storage/ADNI_nii_gz_stripped"
target_folder="adni_storage/ADNI_nii_gz_bias_corrected"

# Path to N4BiasFieldCorrection binary
n4bias_path="ants/ants-2.5.4/bin/N4BiasFieldCorrection"

# Loop through all .nii.gz files in the source folder
for input_file in "${source_folder}"/*.stripped.nii.gz; do
    # Extract the base name without directory and extension
    base_name=$(basename "$input_file" .stripped.nii.gz)

    # Define the output file name and path
    output_file="${target_folder}/${base_name}.stripped.N4.nii.gz"

    # Check if the output file already exists
    if [ -f "$output_file" ]; then
        echo "Output file $output_file already exists. Skipping..."
    else
        echo "Processing $input_file -> $output_file"
        # Run the N4BiasFieldCorrection command
        eval "$n4bias_path -i $input_file -o $output_file"
    fi
    count=$((count + 1))
    echo "$count done"
done
