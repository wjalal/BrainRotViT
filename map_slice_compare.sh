#!/bin/bash

# Folders
dir1="maps_out"
dir2="maps_out_pre-post_mask"
outdir="comparison_out"

# Create output folder
mkdir -p "$outdir"

# Loop through images in dir1
for img1 in "$dir1"/*.png; do
    filename=$(basename "$img1")
    img2="$dir2/$filename"

    # Check if matching file exists in dir2
    if [[ -f "$img2" ]]; then
        echo "Comparing: $filename"
        
        # Make side-by-side comparison
        montage "$img1" "$img2" -tile 2x1 -geometry +5+5 "$outdir/$filename"
    fi
done

echo "Done! Check the $outdir folder for side-by-side comparisons."
