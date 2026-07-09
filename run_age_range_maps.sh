#!/usr/bin/env bash
#
# run_age_range_maps.sh
# =====================
# Generate a per-age-range 3D attention map with the DoRA interpretability
# pipeline, then reduce each to its AAL-atlas-fit NIfTI + region ranking.
#
# For each age range [10,20), [20,30), ... [80,90):
#   1. Edit 3dmap_grad_vit_cnn_main_mix_roi_dora.py IN PLACE to
#        (a) filter df_val to that age range (right after df_val is built), and
#        (b) override the output directory to maps_out_dora_age_<lo>_<hi>.
#   2. Run it -> writes attention_3d_mapped_backprop_dora.nii.gz into that dir.
#   3. Copy center.py, intense_regions_max.py, aal_crop_centered.nii into the dir.
#   4. Run  center.py  (-> ..._cropped_centered_resized.nii.gz)  then
#           intense_regions_max.py  (prints AAL region ranking).
#   5. Delete everything in the dir EXCEPT the cropped/centered/resized .nii.gz.
#   6. Restore the .py to its original form.
# Finally, collect the per-range .nii.gz (and region rankings) into ./age_maps_collected/.
#
# Usage:
#   ./run_age_range_maps.sh [MAX_SUBJECTS]     (default 50 subjects per range)
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---- config ----
PYFILE="3dmap_grad_vit_cnn_main_mix_roi_dora.py"
CNN_MODULE="cnn_mx_bigdo_ch_sw_res"
LOAD_SAVED="best"
MAX_SUBJECTS="${1:-50}"
HELPER_DIR="maps_out_dora"                          # source of helper scripts + atlas
KEEP="attention_3d_mapped_backprop_dora_cropped_centered_resized.nii.gz"
ATLAS="aal_crop_centered.nii"                       # real filename read by intense_regions_max.py
COLLECT="$ROOT/age_maps_collected"

BACKUP="$(mktemp "${PYFILE}.orig.XXXXXX")"
cp "$PYFILE" "$BACKUP"
# Always revert the .py to its original form on exit (normal, error, or Ctrl-C).
trap 'cp "$BACKUP" "$ROOT/$PYFILE"; rm -f "$BACKUP"; echo "Reverted $PYFILE to original."' EXIT

mkdir -p "$COLLECT"

for lo in 10 20 30 40 50 60 70 80; do
    hi=$((lo + 10))
    outdir="maps_out_dora_age_${lo}_${hi}"
    echo ""
    echo "==================== AGE ${lo}-${hi}  ->  ${outdir} ===================="

    # 1. Start from a pristine copy every iteration, then apply the two edits.
    cp "$BACKUP" "$PYFILE"

    #   (a) filter df_val by age, inserted right after df_val is built
    sed -i "/^df_val = df\.iloc\[val_indices\]\.reset_index(drop=True)\$/a df_val = df_val[(df_val['Age'] >= ${lo}) & (df_val['Age'] < ${hi})].reset_index(drop=True)" "$PYFILE"
    #   (b) override the output directory
    sed -i "s|^    out_dir = .*|    out_dir = \"${outdir}\"|" "$PYFILE"

    # sanity: confirm both edits landed and the file still parses
    if ! grep -q "df_val\['Age'\] >= ${lo}" "$PYFILE" || ! grep -q "out_dir = \"${outdir}\"" "$PYFILE"; then
        echo "  !! edit failed for ${lo}-${hi}; skipping"; continue
    fi
    if ! python -m py_compile "$PYFILE"; then
        echo "  !! edited $PYFILE does not compile; skipping ${lo}-${hi}"; continue
    fi

    # 2. Run the interpretability pipeline for this age range.
    if ! python "$PYFILE" "$CNN_MODULE" "$LOAD_SAVED" "$MAX_SUBJECTS"; then
        echo "  !! run failed (likely no subjects in ${lo}-${hi}); skipping"; continue
    fi
    if [ ! -f "${outdir}/attention_3d_mapped_backprop_dora.nii.gz" ]; then
        echo "  !! no attention volume produced for ${lo}-${hi}; skipping"; continue
    fi

    # 3. Copy helper scripts + atlas into the output dir.
    cp "$HELPER_DIR/center.py" "$HELPER_DIR/intense_regions_max.py" "$HELPER_DIR/$ATLAS" "$outdir/"

    # 4. center.py (crop/center/resize) then intense_regions_max.py (AAL ranking).
    pushd "$outdir" >/dev/null
    if ! python center.py; then
        echo "  !! center.py failed for ${lo}-${hi}"; popd >/dev/null; continue
    fi
    python intense_regions_max.py | tee "$COLLECT/age_${lo}_${hi}_regions.txt" || true
    popd >/dev/null

    if [ ! -f "${outdir}/${KEEP}" ]; then
        echo "  !! ${KEEP} not produced for ${lo}-${hi}; skipping cleanup/collect"; continue
    fi

    # 5. Delete everything in outdir except the final resized .nii.gz.
    find "$outdir" -type f ! -name "$KEEP" -delete
    find "$outdir" -mindepth 1 -type d -empty -delete

    # 6. Collect the final .nii.gz (revert of the .py happens at loop top / on exit).
    cp "$outdir/$KEEP" "$COLLECT/age_${lo}_${hi}_${KEEP}"
    echo "  done: ${outdir}/${KEEP}  ->  $COLLECT/age_${lo}_${hi}_${KEEP}"
done

echo ""
echo "==================== SUMMARY ===================="
echo "Final per-age-range attention maps (also kept in each maps_out_dora_age_* dir):"
ls -1 "$COLLECT"/*.nii.gz 2>/dev/null || echo "  (none produced)"
echo "Region rankings: $COLLECT/age_*_regions.txt"
