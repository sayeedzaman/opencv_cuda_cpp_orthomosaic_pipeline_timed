cmake_minimum_required(VERSION 3.18)

project(OpenCVCUDAOrthomosaic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GDAL REQUIRED)

add_executable(cuda_orthorectify
    cuda_orthorectify.cpp
)

add_executable(radiometric_normalize_cuda
    radiometric_normalize_cuda.cpp
)

add_executable(blend_mosaic_feather_cuda
    blend_mosaic_feather_cuda.cpp
)

target_include_directories(cuda_orthorectify PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(radiometric_normalize_cuda PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(blend_mosaic_feather_cuda PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cuda_orthorectify
    ${OpenCV_LIBS}
)

target_link_libraries(radiometric_normalize_cuda
    ${OpenCV_LIBS}
    GDAL::GDAL
)

target_link_libraries(blend_mosaic_feather_cuda
    ${OpenCV_LIBS}
    GDAL::GDAL
)
