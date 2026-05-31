// ============================================================
// C++ OPENCV CUDA RADIOMETRIC NORMALIZATION
// ============================================================
//
// Uses:
//   GDAL for GeoTIFF I/O
//   OpenCV CUDA for per-pixel gain/offset correction
//
// Formula:
//   corrected = image * gain + offset
//
// gain   = global_std / local_std
// offset = global_mean - local_mean * gain
//
// ============================================================

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>

#include <gdal_priv.h>
#include <cpl_conv.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

namespace fs = std::filesystem;

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
        throw std::runtime_error("OpenCV CUDA device count is 0.");
    }

    if (gpu >= count) {
        throw std::runtime_error("Requested GPU index is out of range.");
    }

    cv::cuda::setDevice(gpu);

    std::cout << "OpenCV CUDA check passed. Using GPU: " << gpu << std::endl;
}

static std::vector<fs::path> list_ortho_files(const fs::path& dir) {
    std::vector<fs::path> files;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;

        std::string name = entry.path().filename().string();

        if (name.size() >= 10 && name.find("_ortho.tif") != std::string::npos) {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());

    return files;
}

struct GeoImage {
    cv::Mat rgb;
    std::array<double, 6> gt;
    std::string projection;
    int width = 0;
    int height = 0;
    int bands = 0;
};

static GeoImage read_geotiff_rgb(const fs::path& path) {
    GDALDataset* ds = static_cast<GDALDataset*>(
        GDALOpen(path.string().c_str(), GA_ReadOnly)
    );

    if (!ds) {
        throw std::runtime_error("Could not open GeoTIFF: " + path.string());
    }

    GeoImage img;

    img.width = ds->GetRasterXSize();
    img.height = ds->GetRasterYSize();
    img.bands = std::min(3, ds->GetRasterCount());

    double gt_raw[6];
    ds->GetGeoTransform(gt_raw);

    for (int i = 0; i < 6; ++i) {
        img.gt[i] = gt_raw[i];
    }

    const char* proj = ds->GetProjectionRef();
    img.projection = proj ? std::string(proj) : "";

    std::vector<cv::Mat> channels;

    for (int b = 1; b <= img.bands; ++b) {
        cv::Mat band(img.height, img.width, CV_8UC1);

        CPLErr err = ds->GetRasterBand(b)->RasterIO(
            GF_Read,
            0, 0,
            img.width, img.height,
            band.data,
            img.width, img.height,
            GDT_Byte,
            0, 0
        );

        if (err != CE_None) {
            GDALClose(ds);
            throw std::runtime_error("GDAL RasterIO read failed: " + path.string());
        }

        channels.push_back(band);
    }

    cv::merge(channels, img.rgb);

    GDALClose(ds);

    return img;
}

static void write_geotiff_rgb(
    const fs::path& path,
    const GeoImage& ref,
    const cv::Mat& rgb,
    int nodata
) {
    fs::create_directories(path.parent_path());

    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");

    if (!driver) {
        throw std::runtime_error("GTiff driver not available.");
    }

    char** options = nullptr;
    options = CSLSetNameValue(options, "COMPRESS", "LZW");
    options = CSLSetNameValue(options, "TILED", "YES");
    options = CSLSetNameValue(options, "BIGTIFF", "IF_SAFER");

    GDALDataset* ds = driver->Create(
        path.string().c_str(),
        rgb.cols,
        rgb.rows,
        rgb.channels(),
        GDT_Byte,
        options
    );

    CSLDestroy(options);

    if (!ds) {
        throw std::runtime_error("Could not create output GeoTIFF: " + path.string());
    }

    ds->SetGeoTransform(const_cast<double*>(ref.gt.data()));
    ds->SetProjection(ref.projection.c_str());

    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);

    for (int b = 0; b < rgb.channels(); ++b) {
        GDALRasterBand* band = ds->GetRasterBand(b + 1);
        band->SetNoDataValue(nodata);

        CPLErr err = band->RasterIO(
            GF_Write,
            0, 0,
            rgb.cols, rgb.rows,
            channels[b].data,
            rgb.cols, rgb.rows,
            GDT_Byte,
            0, 0
        );

        if (err != CE_None) {
            GDALClose(ds);
            throw std::runtime_error("GDAL RasterIO write failed: " + path.string());
        }
    }

    GDALClose(ds);
}

static cv::Mat valid_mask(const cv::Mat& rgb, int nodata) {
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);

    cv::Mat mask = cv::Mat::zeros(rgb.rows, rgb.cols, CV_8UC1);

    for (const auto& ch : channels) {
        cv::Mat m;
        cv::compare(ch, nodata, m, cv::CMP_NE);
        cv::bitwise_or(mask, m, mask);
    }

    return mask;
}

static std::array<double, 3> mean_for_file(const cv::Mat& rgb, const cv::Mat& mask) {
    cv::Scalar m = cv::mean(rgb, mask);

    return {m[0], m[1], m[2]};
}

static std::array<double, 3> std_for_file(const cv::Mat& rgb, const cv::Mat& mask) {
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);

    std::array<double, 3> out{1.0, 1.0, 1.0};

    for (int c = 0; c < std::min(3, static_cast<int>(channels.size())); ++c) {
        cv::Scalar mean, stddev;
        cv::meanStdDev(channels[c], mean, stddev, mask);
        out[c] = stddev[0] + 1e-6;
    }

    return out;
}

static cv::Mat normalize_cuda(
    const cv::Mat& rgb,
    const std::array<double, 3>& global_mean,
    const std::array<double, 3>& global_std,
    int nodata
) {
    cv::Mat mask = valid_mask(rgb, nodata);

    auto local_mean = mean_for_file(rgb, mask);
    auto local_std = std_for_file(rgb, mask);

    cv::cuda::GpuMat d_src;
    d_src.upload(rgb);

    cv::cuda::GpuMat d_float;
    d_src.convertTo(d_float, CV_32F);

    std::vector<cv::cuda::GpuMat> d_channels;
    cv::cuda::split(d_float, d_channels);

    for (int c = 0; c < std::min(3, static_cast<int>(d_channels.size())); ++c) {
        double gain = global_std[c] / local_std[c];
        double offset = global_mean[c] - local_mean[c] * gain;

        cv::cuda::GpuMat scaled;
        cv::cuda::multiply(d_channels[c], cv::Scalar(gain), scaled);

        cv::cuda::add(scaled, cv::Scalar(offset), d_channels[c]);
    }

    cv::cuda::GpuMat d_merged;
    cv::cuda::merge(d_channels, d_merged);

    cv::cuda::GpuMat d_u8;
    d_merged.convertTo(d_u8, CV_8U);

    cv::Mat corrected;
    d_u8.download(corrected);

    corrected.setTo(cv::Scalar(nodata, nodata, nodata), mask == 0);

    return corrected;
}

int main(int argc, char** argv) {
    try {
        auto program_start = std::chrono::steady_clock::now();
        GDALAllRegister();

        auto args = parse_args(argc, argv);

        fs::path input_dir = args.at("--input");
        fs::path output_dir = args.at("--output");

        int nodata = args.count("--nodata") ? std::stoi(args["--nodata"]) : 0;
        int gpu = args.count("--gpu") ? std::stoi(args["--gpu"]) : 0;

        check_cuda_or_throw(gpu);

        auto files = list_ortho_files(input_dir);

        if (files.empty()) {
            throw std::runtime_error("No *_ortho.tif files found in input directory.");
        }

        std::vector<std::array<double, 3>> means;
        std::vector<std::array<double, 3>> stds;

        std::cout << "Computing global radiometric statistics..." << std::endl;

        for (const auto& file : files) {
            GeoImage img = read_geotiff_rgb(file);
            cv::Mat mask = valid_mask(img.rgb, nodata);

            if (cv::countNonZero(mask) < 100) continue;

            means.push_back(mean_for_file(img.rgb, mask));
            stds.push_back(std_for_file(img.rgb, mask));
        }

        if (means.empty()) {
            throw std::runtime_error("Could not compute global stats.");
        }

        std::array<double, 3> global_mean{0, 0, 0};
        std::array<double, 3> global_std{0, 0, 0};

        for (const auto& m : means) {
            for (int c = 0; c < 3; ++c) global_mean[c] += m[c];
        }

        for (const auto& s : stds) {
            for (int c = 0; c < 3; ++c) global_std[c] += s[c];
        }

        for (int c = 0; c < 3; ++c) {
            global_mean[c] /= means.size();
            global_std[c] /= stds.size();
        }

        std::cout << "Global mean: "
                  << global_mean[0] << ", "
                  << global_mean[1] << ", "
                  << global_mean[2] << std::endl;

        std::cout << "Global std: "
                  << global_std[0] << ", "
                  << global_std[1] << ", "
                  << global_std[2] << std::endl;

        fs::create_directories(output_dir);

        for (const auto& file : files) {
            auto tile_start = std::chrono::steady_clock::now();

            std::cout << "C++ OpenCV CUDA radiometric correction: "
                      << file << std::endl;

            GeoImage img = read_geotiff_rgb(file);

            cv::Mat corrected = normalize_cuda(
                img.rgb,
                global_mean,
                global_std,
                nodata
            );

            fs::path out_file = output_dir / file.filename();

            write_geotiff_rgb(out_file, img, corrected, nodata);

            auto tile_end = std::chrono::steady_clock::now();
            double tile_sec = std::chrono::duration<double>(tile_end - tile_start).count();

            std::cout << "Radiometric tile elapsed seconds: " << tile_sec << std::endl;
        }

        auto program_end = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(program_end - program_start).count();

        std::cout << "C++ OpenCV CUDA radiometric normalization complete." << std::endl;
        std::cout << "radiometric_normalize_cuda elapsed seconds: " << elapsed_sec << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
