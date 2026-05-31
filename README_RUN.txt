FULL MODIFIED OPENCV CUDA + C++ ORTHOMOSAIC PIPELINE
==================================================

This version uses C++ where it improves performance.

GPU/C++ executables:
  1. cuda_orthorectify
     - OpenCV CUDA remap
     - replaces Python orthorectification

  2. radiometric_normalize_cuda
     - GDAL GeoTIFF read/write
     - OpenCV CUDA per-pixel gain/offset correction

  3. blend_mosaic_feather_cuda
     - GDAL GeoTIFF read/write
     - OpenCV CUDA tile weighting
     - CPU-safe final mosaic accumulation

Python remains for:
  1. 02_generate_remap_maps.py
     - photogrammetric DSM + COLMAP camera projection
     - writes xmap/ymap .npy files

Required folder structure:

  /home/user/project/
  ├── images/
  ├── scripts/
  └── output/

Copy all files in scripts/ to:

  /home/user/project/scripts/

Change PROJECT_DIR in:

  scripts/00_run_full_pipeline.sh

Default:

  PROJECT_DIR="/home/user/project"

Install dependencies:

  sudo apt update

  sudo apt install -y \
    colmap \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    pdal \
    python3-numpy \
    cmake \
    build-essential \
    libopencv-dev

IMPORTANT:

  libopencv-dev from Ubuntu is usually CPU-only.
  For this pipeline, OpenCV must be compiled with CUDA.

Check your OpenCV CUDA C++ build with:

  opencv_version --verbose | grep -i cuda

or compile and run the pipeline. The C++ executables will fail early if CUDA is not available.

Build and run:

  chmod +x /home/user/project/scripts/*.sh
  chmod +x /home/user/project/scripts/*.py

  bash /home/user/project/scripts/00_run_full_pipeline.sh

Final output:

  /home/user/project/output/mosaic_blended/final_orthomosaic_blended.tif

Basic fallback mosaic:

  /home/user/project/output/mosaic/final_orthomosaic_basic.tif

GPU acceleration used in:

  - COLMAP SIFT extraction
  - COLMAP SIFT matching
  - COLMAP PatchMatch stereo
  - C++ OpenCV CUDA orthorectification
  - C++ OpenCV CUDA radiometric correction
  - C++ OpenCV CUDA feather weighting

Still CPU / I/O:

  - COLMAP mapper / bundle adjustment
  - PDAL filtering
  - DSM creation
  - Remap-grid generation
  - GDAL GeoTIFF writing
  - Final large-mosaic accumulation to avoid VRAM overflow


TIMING REPORT
=============

This timed version writes a timing summary to:

  /home/user/project/output/timing_report.txt

The report includes:

  - Total pipeline time
  - Time for each major step
  - Output file locations

The console also prints timing for:

  - Each major pipeline step
  - Each orthorectified tile
  - Each radiometric correction tile
  - Each feather-blending tile

