# Build the orthomosaic pipeline Docker image
# Run this once. Takes 30-60 min (OpenCV CUDA build).

docker build -t orthomosaic-pipeline:latest .
