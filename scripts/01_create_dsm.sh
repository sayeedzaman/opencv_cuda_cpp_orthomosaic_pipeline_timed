#!/bin/bash
set -e

# ============================================================
# CREATE DSM FROM DENSE POINT CLOUD
# ============================================================

POINT_CLOUD="$1"
DSM_OUT="$2"
RESOLUTION="$3"

if [ -z "$POINT_CLOUD" ] || [ -z "$DSM_OUT" ] || [ -z "$RESOLUTION" ]; then
    echo "Usage:"
    echo "  01_create_dsm.sh input_point_cloud.ply output_dsm.tif resolution"
    exit 1
fi

OUT_DIR=$(dirname "$DSM_OUT")
mkdir -p "$OUT_DIR"

PIPELINE_JSON="$OUT_DIR/pdal_create_dsm.json"

cat > "$PIPELINE_JSON" <<EOF
[
  "$POINT_CLOUD",
  {
    "type": "writers.gdal",
    "filename": "$DSM_OUT",
    "resolution": $RESOLUTION,
    "output_type": "max",
    "data_type": "float32",
    "nodata": -9999,
    "window_size": 3
  }
]
EOF

echo "Creating DSM with PDAL..."
pdal pipeline "$PIPELINE_JSON"

FILLED_DSM="${DSM_OUT%.tif}_filled.tif"

echo "Filling DSM no-data gaps..."
gdal_fillnodata.py \
  -md 5 \
  -si 2 \
  "$DSM_OUT" \
  "$FILLED_DSM"

mv "$FILLED_DSM" "$DSM_OUT"

echo "DSM created:"
echo "$DSM_OUT"
