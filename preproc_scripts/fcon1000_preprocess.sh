#!/bin/bash

# Initialize count
count=0

# # Define source and target directories
source_folder="fcon1000_storage/fcon1000_stripped"
target_folder="fcon1000_storage/fcon1000_bias_corrected"

# source_folder="fcon1000_storage/fcon1000_skullstripped"
# target_folder="fcon1000_storage/fcon1000_skullstripped_bias_corrected"

# Path to N4BiasFieldCorrection binary
n4bias_path="ants/ants-2.5.4/bin/N4BiasFieldCorrection"

# Loop through all .nii.gz files in the source folder
for input_file in "${source_folder}"/*.nii.gz; do
    # Extract the base name without directory and extension
    base_name=$(basename "$input_file" .nii.gz)

    # Define the output file name and path
    output_file="${target_folder}/${base_name}.N4.nii.gz"

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
