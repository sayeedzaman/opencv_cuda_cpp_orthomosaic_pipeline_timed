FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PROJECT_DIR=/project
ENV QT_QPA_PLATFORM=offscreen
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}

# Base dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev python3-gdal \
    pdal \
    python3-pip python3-numpy \
    cmake build-essential ninja-build \
    libgtk-3-dev pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libboost-all-dev \
    git wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Build COLMAP with CUDA support
RUN git clone --depth 1 --branch 3.7 https://github.com/colmap/colmap.git /opt/colmap && \
    mkdir /opt/colmap/build && cd /opt/colmap/build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_CUDA=ON \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DCUDA_ARCH=86 \
      -DWITH_OPENGL=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /opt/colmap

# Build OpenCV 4.10 with CUDA 8.6 (RTX 3060)
RUN git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git /opt/opencv && \
    git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv_contrib.git /opt/opencv_contrib && \
    mkdir /opt/opencv/build && cd /opt/opencv/build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
      -DWITH_CUDA=ON \
      -DCUDA_ARCH_BIN=8.6 \
      -DWITH_CUDNN=OFF \
      -DOPENCV_DNN_CUDA=OFF \
      -DWITH_OPENGL=OFF \
      -DBUILD_opencv_python3=ON \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /opt/opencv /opt/opencv_contrib

WORKDIR /project

CMD ["bash", "scripts/00_run_full_pipeline.sh"]
