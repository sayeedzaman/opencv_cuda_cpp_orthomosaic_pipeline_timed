#!/bin/bash
set -e

# ============================================================
# BASIC GDAL MOSAIC FALLBACK
# ============================================================

ORTHO_IMAGE_DIR="$1"
FINAL_ORTHO="$2"

if [ -z "$ORTHO_IMAGE_DIR" ] || [ -z "$FINAL_ORTHO" ]; then
    echo "Usage:"
    echo "  04_make_mosaic.sh ortho_image_dir final_orthomosaic.tif"
    exit 1
fi

MOSAIC_DIR=$(dirname "$FINAL_ORTHO")
mkdir -p "$MOSAIC_DIR"

VRT="$MOSAIC_DIR/orthomosaic.vrt"

echo "Building GDAL VRT..."

gdalbuildvrt \
  -resolution highest \
  -srcnodata 0 \
  -vrtnodata 0 \
  "$VRT" \
  "$ORTHO_IMAGE_DIR"/*_ortho.tif

echo "Creating final basic GeoTIFF..."

gdal_translate \
  -co COMPRESS=LZW \
  -co TILED=YES \
  -co BIGTIFF=IF_SAFER \
  "$VRT" \
  "$FINAL_ORTHO"

echo "Creating overviews..."

gdaladdo \
  -r average \
  "$FINAL_ORTHO" \
  2 4 8 16 32

echo "Final basic orthomosaic created:"
echo "$FINAL_ORTHO"
