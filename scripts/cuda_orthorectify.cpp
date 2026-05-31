// ============================================================
// C++ OPENCV CUDA ORTHORECTIFICATION
// ============================================================
//
// Uses:
//   cv::cuda::remap
//
// Input:
//   --image source.jpg
//   --xmap xmap.npy
//   --ymap ymap.npy
//   --width output_width
//   --height output_height
//   --output raw_ortho.tif
//
// Output:
//   raw orthorectified TIFF without georeference
//
// GDAL georeference is assigned later by gdal_translate.
//
// ============================================================

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

#include "npy_reader.hpp"

static std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];

        if (key.rfind("--", 0) == 0) {
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                args[key] = argv[i + 1];
                ++i;
            } else {
                args[key] = "true";
            }
        }
    }

    return args;
}

static void check_cuda_or_throw(int gpu) {
    int count = cv::cuda::getCudaEnabledDeviceCount();

    if (count <= 0) {
        throw std::runtime_error("OpenCV CUDA device count is 0. OpenCV may not be compiled with CUDA.");
    }

    if (gpu >= count) {
        throw std::runtime_error("Requested GPU index is out of range.");
    }

    cv::cuda::setDevice(gpu);

    std::cout << "OpenCV CUDA check passed." << std::endl;
    std::cout << "CUDA device count: " << count << std::endl;
    std::cout << "Using GPU: " << gpu << std::endl;
}

int main(int argc, char** argv) {
    try {
        auto program_start = std::chrono::steady_clock::now();
        auto args = parse_args(argc, argv);

        int gpu = 0;

        if (args.count("--gpu")) {
            gpu = std::stoi(args["--gpu"]);
        }

        if (args.count("--check-cuda")) {
            check_cuda_or_throw(gpu);
            return 0;
        }

        const std::string image_path = args.at("--image");
        const std::string xmap_path = args.at("--xmap");
        const std::string ymap_path = args.at("--ymap");
        const int width = std::stoi(args.at("--width"));
        const int height = std::stoi(args.at("--height"));
        const std::string output_path = args.at("--output");

        check_cuda_or_throw(gpu);

        cv::Mat src = cv::imread(image_path, cv::IMREAD_COLOR);

        if (src.empty()) {
            throw std::runtime_error("Could not read source image: " + image_path);
        }

        cv::Mat xmap = read_npy_float32_2d(xmap_path, width, height);
        cv::Mat ymap = read_npy_float32_2d(ymap_path, width, height);

        cv::cuda::GpuMat d_src;
        cv::cuda::GpuMat d_xmap;
        cv::cuda::GpuMat d_ymap;
        cv::cuda::GpuMat d_dst;

        d_src.upload(src);
        d_xmap.upload(xmap);
        d_ymap.upload(ymap);

        std::cout << "Running C++ OpenCV CUDA remap: " << image_path << std::endl;

        cv::cuda::remap(
            d_src,
            d_dst,
            d_xmap,
            d_ymap,
            cv::INTER_CUBIC,
            cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0)
        );

        cv::Mat dst;
        d_dst.download(dst);

        if (!cv::imwrite(output_path, dst)) {
            throw std::runtime_error("Could not write output image: " + output_path);
        }

        auto program_end = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(program_end - program_start).count();

        std::cout << "C++ OpenCV CUDA orthorectification complete: " << output_path << std::endl;
        std::cout << "cuda_orthorectify elapsed seconds: " << elapsed_sec << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
