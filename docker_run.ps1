# Run the orthomosaic pipeline in Docker with GPU access

$ProjectDir = $PSScriptRoot

docker run --rm --gpus all `
    -m 14g `
    -e NVIDIA_VISIBLE_DEVICES=all `
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility `
    -e QT_QPA_PLATFORM=offscreen `
    -v "${ProjectDir}:/project" `
    orthomosaic-pipeline:latest
