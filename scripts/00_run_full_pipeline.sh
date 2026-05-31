#!/bin/bash
set -e

# ============================================================
# FULL MODIFIED C++ + OPENCV CUDA ORTHOMOSAIC PIPELINE
# WITH STEP-BY-STEP TIMING
# ============================================================
#
# This script prints:
#   1. Start time of each major step
#   2. End time of each major step
#   3. Elapsed time for each major step
#   4. Total pipeline computation time
#
# Timing is written to:
#   output/timing_report.txt
#
# ============================================================


# -----------------------------
# CHANGE THIS PATH IF NEEDED
# -----------------------------

PROJECT_DIR="/home/user/project"

IMAGE_DIR="$PROJECT_DIR/images"
OUTPUT_DIR="$PROJECT_DIR/output"

COLMAP_DB="$OUTPUT_DIR/database.db"

SPARSE_DIR="$OUTPUT_DIR/sparse"
SPARSE_TXT="$OUTPUT_DIR/sparse_txt"
GCP_CORRECTED_SPARSE="$OUTPUT_DIR/sparse_gcp_corrected"

DENSE_DIR="$OUTPUT_DIR/dense"
DSM_DIR="$OUTPUT_DIR/dsm"
ORTHO_MAP_DIR="$OUTPUT_DIR/ortho_maps"
ORTHO_IMAGE_DIR="$OUTPUT_DIR/ortho_images"
MOSAIC_DIR="$OUTPUT_DIR/mosaic"

RADIO_IMAGE_DIR="$OUTPUT_DIR/ortho_images_radiometric"
BLENDED_MOSAIC_DIR="$OUTPUT_DIR/mosaic_blended"

BUILD_DIR="$PROJECT_DIR/scripts/build"


# -----------------------------
# COORDINATE SYSTEM
# -----------------------------

TARGET_EPSG="EPSG:32646"


# -----------------------------
# PIXEL SIZE
# -----------------------------

DSM_RESOLUTION=0.05
ORTHO_RESOLUTION=0.05


# -----------------------------
# OPTIONAL GCP TRANSFORM
# -----------------------------

GCP_TRANSFORM="$OUTPUT_DIR/gcp_transform.txt"


# -----------------------------
# REMAP SETTINGS
# -----------------------------

BLOCK_SIZE=2048
MIN_VALID_PIXELS=500


# -----------------------------
# GPU INDEX
# -----------------------------

GPU_INDEX=0


# -----------------------------
# CREATE DIRECTORIES
# -----------------------------

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SPARSE_DIR"
mkdir -p "$SPARSE_TXT"
mkdir -p "$GCP_CORRECTED_SPARSE"
mkdir -p "$DENSE_DIR"
mkdir -p "$DSM_DIR"
mkdir -p "$ORTHO_MAP_DIR"
mkdir -p "$ORTHO_IMAGE_DIR"
mkdir -p "$MOSAIC_DIR"
mkdir -p "$RADIO_IMAGE_DIR"
mkdir -p "$BLENDED_MOSAIC_DIR"


# -----------------------------
# TIMING FUNCTIONS
# -----------------------------

TIMING_REPORT="$OUTPUT_DIR/timing_report.txt"

PIPELINE_START=$(date +%s)
PIPELINE_START_READABLE=$(date "+%Y-%m-%d %H:%M:%S")

STEP_NAMES=()
STEP_SECONDS=()

format_seconds() {
    local T=$1
    local H=$((T / 3600))
    local M=$(((T % 3600) / 60))
    local S=$((T % 60))
    printf "%02d:%02d:%02d" "$H" "$M" "$S"
}

start_step() {
    CURRENT_STEP_NAME="$1"
    CURRENT_STEP_START=$(date +%s)
    CURRENT_STEP_START_READABLE=$(date "+%Y-%m-%d %H:%M:%S")

    echo ""
    echo "============================================================"
    echo "$CURRENT_STEP_NAME"
    echo "Started: $CURRENT_STEP_START_READABLE"
    echo "============================================================"
}

end_step() {
    local STEP_END
    local STEP_END_READABLE
    local ELAPSED
    local ELAPSED_FMT

    STEP_END=$(date +%s)
    STEP_END_READABLE=$(date "+%Y-%m-%d %H:%M:%S")
    ELAPSED=$((STEP_END - CURRENT_STEP_START))
    ELAPSED_FMT=$(format_seconds "$ELAPSED")

    STEP_NAMES+=("$CURRENT_STEP_NAME")
    STEP_SECONDS+=("$ELAPSED")

    echo "------------------------------------------------------------"
    echo "$CURRENT_STEP_NAME completed."
    echo "Ended: $STEP_END_READABLE"
    echo "Elapsed: $ELAPSED_FMT"
    echo "------------------------------------------------------------"
}

write_timing_report() {
    local PIPELINE_END
    local TOTAL
    local TOTAL_FMT

    PIPELINE_END=$(date +%s)
    TOTAL=$((PIPELINE_END - PIPELINE_START))
    TOTAL_FMT=$(format_seconds "$TOTAL")

    {
        echo "ORTHOMOSAIC PIPELINE TIMING REPORT"
        echo "=================================="
        echo "Project directory: $PROJECT_DIR"
        echo "Started: $PIPELINE_START_READABLE"
        echo "Ended: $(date "+%Y-%m-%d %H:%M:%S")"
        echo "Total elapsed time: $TOTAL_FMT"
        echo ""
        echo "Step timings:"
        echo "-------------"

        for i in "${!STEP_NAMES[@]}"; do
            printf "%02d. %-65s %s\n" "$((i+1))" "${STEP_NAMES[$i]}" "$(format_seconds "${STEP_SECONDS[$i]}")"
        done

        echo ""
        echo "Output files:"
        echo "-------------"
        echo "Basic GDAL mosaic: $MOSAIC_DIR/final_orthomosaic_basic.tif"
        echo "Final blended mosaic: $BLENDED_MOSAIC_DIR/final_orthomosaic_blended.tif"
    } > "$TIMING_REPORT"

    echo ""
    echo "============================================================"
    echo "TOTAL PIPELINE TIME: $TOTAL_FMT"
    echo "Timing report saved to:"
    echo "$TIMING_REPORT"
    echo "============================================================"
}

trap 'echo "Pipeline failed. Partial timing report may be incomplete."; write_timing_report' ERR


start_step "STEP 0: Build C++ OpenCV CUDA tools"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake ..
make -j$(nproc)

end_step


start_step "STEP 0.1: Check C++ OpenCV CUDA availability"

"$BUILD_DIR/cuda_orthorectify" --check-cuda --gpu "$GPU_INDEX"

end_step


start_step "STEP 1: COLMAP GPU feature extraction"

colmap feature_extractor \
  --database_path "$COLMAP_DB" \
  --image_path "$IMAGE_DIR" \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 1 \
  --SiftExtraction.gpu_index "$GPU_INDEX" \
  --SiftExtraction.max_num_features 12000

end_step


start_step "STEP 2: COLMAP GPU feature matching"

colmap exhaustive_matcher \
  --database_path "$COLMAP_DB" \
  --SiftMatching.use_gpu 1 \
  --SiftMatching.gpu_index "$GPU_INDEX"

end_step


start_step "STEP 3: COLMAP sparse reconstruction"

rm -rf "$SPARSE_DIR"
mkdir -p "$SPARSE_DIR"

colmap mapper \
  --database_path "$COLMAP_DB" \
  --image_path "$IMAGE_DIR" \
  --output_path "$SPARSE_DIR" \
  --Mapper.ba_refine_focal_length 1 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 1 \
  --Mapper.filter_max_reproj_error 3.0 \
  --Mapper.filter_min_tri_angle 1.5 \
  --Mapper.min_num_matches 20 \
  --Mapper.init_min_num_inliers 100

SPARSE_MODEL="$SPARSE_DIR/0"

if [ ! -d "$SPARSE_MODEL" ]; then
    echo "ERROR: Sparse model not found:"
    echo "$SPARSE_MODEL"
    exit 1
fi

end_step


start_step "STEP 3.1: Final bundle adjustment"

colmap bundle_adjuster \
  --input_path "$SPARSE_MODEL" \
  --output_path "$SPARSE_MODEL" \
  --BundleAdjustment.refine_focal_length 1 \
  --BundleAdjustment.refine_principal_point 0 \
  --BundleAdjustment.refine_extra_params 1 \
  --BundleAdjustment.max_num_iterations 100

end_step


start_step "STEP 4: Export sparse model to TXT"

rm -rf "$SPARSE_TXT"
mkdir -p "$SPARSE_TXT"

colmap model_converter \
  --input_path "$SPARSE_MODEL" \
  --output_path "$SPARSE_TXT" \
  --output_type TXT

end_step


start_step "STEP 5: Optional GCP Helmert correction"

INPUT_SPARSE_FOR_DENSE=""

if [ -f "$GCP_TRANSFORM" ]; then

    echo "GCP transform found:"
    echo "$GCP_TRANSFORM"
    echo "Applying GCP correction..."

    rm -rf "$GCP_CORRECTED_SPARSE"
    mkdir -p "$GCP_CORRECTED_SPARSE"

    colmap model_transformer \
      --input_path "$SPARSE_MODEL" \
      --output_path "$GCP_CORRECTED_SPARSE" \
      --transform_path "$GCP_TRANSFORM"

    INPUT_SPARSE_FOR_DENSE="$GCP_CORRECTED_SPARSE"

else

    echo "WARNING: GCP transform file not found:"
    echo "$GCP_TRANSFORM"
    echo "Continuing WITHOUT GCP correction."

    INPUT_SPARSE_FOR_DENSE="$SPARSE_MODEL"

fi

end_step


start_step "STEP 6: COLMAP image undistortion"

rm -rf "$DENSE_DIR"
mkdir -p "$DENSE_DIR"

colmap image_undistorter \
  --image_path "$IMAGE_DIR" \
  --input_path "$INPUT_SPARSE_FOR_DENSE" \
  --output_path "$DENSE_DIR" \
  --output_type COLMAP

end_step


start_step "STEP 7: COLMAP GPU dense reconstruction"

colmap patch_match_stereo \
  --workspace_path "$DENSE_DIR" \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.filter true \
  --PatchMatchStereo.gpu_index "$GPU_INDEX" \
  --PatchMatchStereo.num_samples 15 \
  --PatchMatchStereo.num_iterations 5 \
  --PatchMatchStereo.window_radius 5 \
  --PatchMatchStereo.window_step 1 \
  --PatchMatchStereo.min_triangulation_angle 1.5

end_step


start_step "STEP 7.1: COLMAP stereo fusion"

colmap stereo_fusion \
  --workspace_path "$DENSE_DIR" \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path "$DENSE_DIR/fused_raw.ply" \
  --StereoFusion.min_num_pixels 5 \
  --StereoFusion.max_reproj_error 2.0 \
  --StereoFusion.max_depth_error 0.01 \
  --StereoFusion.max_normal_error 10

end_step


start_step "STEP 7.2: PDAL statistical outlier filtering"

FILTERED_PLY="$DENSE_DIR/fused_filtered.ply"

cat > "$DENSE_DIR/pdal_filter_outliers.json" <<EOF
[
  "$DENSE_DIR/fused_raw.ply",
  {
    "type": "filters.outlier",
    "method": "statistical",
    "mean_k": 12,
    "multiplier": 2.0
  },
  {
    "type": "filters.range",
    "limits": "Classification![7:7]"
  },
  {
    "type": "writers.ply",
    "filename": "$FILTERED_PLY"
  }
]
EOF

if pdal pipeline "$DENSE_DIR/pdal_filter_outliers.json"; then
    DENSE_POINT_CLOUD="$FILTERED_PLY"
else
    echo "WARNING: PDAL filtering failed."
    echo "Continuing with raw COLMAP fused point cloud."
    DENSE_POINT_CLOUD="$DENSE_DIR/fused_raw.ply"
fi

end_step


start_step "STEP 8: Create DSM from dense point cloud"

bash "$PROJECT_DIR/scripts/01_create_dsm.sh" \
  "$DENSE_POINT_CLOUD" \
  "$DSM_DIR/dsm.tif" \
  "$DSM_RESOLUTION"

end_step


start_step "STEP 9: Generate memory-safe remap grids"

rm -rf "$ORTHO_MAP_DIR"
mkdir -p "$ORTHO_MAP_DIR"

python3 "$PROJECT_DIR/scripts/02_generate_remap_maps.py" \
  --project "$PROJECT_DIR" \
  --dense "$DENSE_DIR" \
  --dsm "$DSM_DIR/dsm.tif" \
  --output "$ORTHO_MAP_DIR" \
  --resolution "$ORTHO_RESOLUTION" \
  --epsg "$TARGET_EPSG" \
  --block-size "$BLOCK_SIZE" \
  --min-valid-pixels "$MIN_VALID_PIXELS"

end_step


start_step "STEP 10: C++ OpenCV CUDA orthorectification"

rm -rf "$ORTHO_IMAGE_DIR"
mkdir -p "$ORTHO_IMAGE_DIR"

MAP_FILES=$(find "$ORTHO_MAP_DIR" -name "*.json" ! -name "remap_index.json")

if [ -z "$MAP_FILES" ]; then
    echo "ERROR: No remap JSON files found in:"
    echo "$ORTHO_MAP_DIR"
    exit 1
fi

ORTHO_TILE_COUNT=0

for MAP_JSON in $MAP_FILES; do

    TILE_START=$(date +%s)

    BASE=$(basename "$MAP_JSON" .json)

    IMG_PATH=$(python3 -c "import json; print(json.load(open('$MAP_JSON'))['image_path'])")
    XMAP=$(python3 -c "import json; print(json.load(open('$MAP_JSON'))['xmap'])")
    YMAP=$(python3 -c "import json; print(json.load(open('$MAP_JSON'))['ymap'])")
    WIDTH=$(python3 -c "import json; print(json.load(open('$MAP_JSON'))['width'])")
    HEIGHT=$(python3 -c "import json; print(json.load(open('$MAP_JSON'))['height'])")

    ULX=$(python3 -c "import json; print(json.load(open('$MAP_JSON'))['ulx'])")
    ULY=$(python3 -c "import json; print(json.load(open('$MAP_JSON'))['uly'])")
    LRX=$(python3 -c "import json; print(json.load(open('$MAP_JSON'))['lrx'])")
    LRY=$(python3 -c "import json; print(json.load(open('$MAP_JSON'))['lry'])")
    EPSG=$(python3 -c "import json; print(json.load(open('$MAP_JSON'))['epsg'])")

    OUT_IMG="$ORTHO_IMAGE_DIR/${BASE}_ortho_raw.tif"
    GEO_IMG="$ORTHO_IMAGE_DIR/${BASE}_ortho.tif"

    echo "C++ OpenCV CUDA orthorectifying tile:"
    echo "$BASE"

    "$BUILD_DIR/cuda_orthorectify" \
      --image "$IMG_PATH" \
      --xmap "$XMAP" \
      --ymap "$YMAP" \
      --width "$WIDTH" \
      --height "$HEIGHT" \
      --output "$OUT_IMG" \
      --gpu "$GPU_INDEX"

    gdal_translate \
      -a_srs "$EPSG" \
      -a_ullr "$ULX" "$ULY" "$LRX" "$LRY" \
      -co COMPRESS=LZW \
      -co TILED=YES \
      "$OUT_IMG" \
      "$GEO_IMG"

    rm "$OUT_IMG"

    TILE_END=$(date +%s)
    TILE_ELAPSED=$((TILE_END - TILE_START))
    echo "Tile $BASE elapsed: $(format_seconds "$TILE_ELAPSED")"

    ORTHO_TILE_COUNT=$((ORTHO_TILE_COUNT + 1))

done

echo "Orthorectified tile count: $ORTHO_TILE_COUNT"

end_step


start_step "STEP 11: Basic GDAL mosaic fallback"

bash "$PROJECT_DIR/scripts/04_make_mosaic.sh" \
  "$ORTHO_IMAGE_DIR" \
  "$MOSAIC_DIR/final_orthomosaic_basic.tif"

end_step


start_step "STEP 12: C++ OpenCV CUDA radiometric normalization"

rm -rf "$RADIO_IMAGE_DIR"
mkdir -p "$RADIO_IMAGE_DIR"

"$BUILD_DIR/radiometric_normalize_cuda" \
  --input "$ORTHO_IMAGE_DIR" \
  --output "$RADIO_IMAGE_DIR" \
  --nodata 0 \
  --gpu "$GPU_INDEX"

end_step


start_step "STEP 13: C++ OpenCV CUDA feather blended orthomosaic"

rm -rf "$BLENDED_MOSAIC_DIR"
mkdir -p "$BLENDED_MOSAIC_DIR"

"$BUILD_DIR/blend_mosaic_feather_cuda" \
  --input "$RADIO_IMAGE_DIR" \
  --output "$BLENDED_MOSAIC_DIR/final_orthomosaic_blended.tif" \
  --nodata 0 \
  --gpu "$GPU_INDEX"

end_step


write_timing_report

echo ""
echo "Basic GDAL mosaic:"
echo "$MOSAIC_DIR/final_orthomosaic_basic.tif"
echo ""
echo "Final C++ OpenCV CUDA corrected blended orthomosaic:"
echo "$BLENDED_MOSAIC_DIR/final_orthomosaic_blended.tif"
