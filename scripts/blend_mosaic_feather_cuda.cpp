// ============================================================
// C++ OPENCV CUDA FEATHER-BLENDED MOSAIC
// ============================================================
//
// Uses:
//   GDAL for GeoTIFF I/O
//   OpenCV CUDA for tile * feather_weight
//
// Final full-mosaic accumulation is CPU-side to avoid VRAM overflow.
//
// ============================================================

#include <algorithm>
#include <filesystem>
#include <iostream>
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

        if (name.find("_ortho.tif") != std::string::npos) {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());

    return files;
}

struct TileInfo {
    fs::path path;
    std::array<double, 6> gt;
    std::string projection;
    int width = 0;
    int height = 0;
    int bands = 0;
    double minx = 0;
    double maxx = 0;
    double miny = 0;
    double maxy = 0;
    double pixel_width = 0;
    double pixel_height = 0;
};

static TileInfo get_tile_info(const fs::path& path) {
    GDALDataset* ds = static_cast<GDALDataset*>(
        GDALOpen(path.string().c_str(), GA_ReadOnly)
    );

    if (!ds) {
        throw std::runtime_error("Could not open GeoTIFF: " + path.string());
    }

    TileInfo info;
    info.path = path;
    info.width = ds->GetRasterXSize();
    info.height = ds->GetRasterYSize();
    info.bands = std::min(3, ds->GetRasterCount());

    double gt_raw[6];
    ds->GetGeoTransform(gt_raw);

    for (int i = 0; i < 6; ++i) info.gt[i] = gt_raw[i];

    const char* proj = ds->GetProjectionRef();
    info.projection = proj ? std::string(proj) : "";

    info.minx = info.gt[0];
    info.maxy = info.gt[3];
    info.maxx = info.gt[0] + info.width * info.gt[1];
    info.miny = info.gt[3] + info.height * info.gt[5];

    info.pixel_width = info.gt[1];
    info.pixel_height = std::abs(info.gt[5]);

    GDALClose(ds);

    return info;
}

static cv::Mat read_tile_rgb(const TileInfo& info) {
    GDALDataset* ds = static_cast<GDALDataset*>(
        GDALOpen(info.path.string().c_str(), GA_ReadOnly)
    );

    if (!ds) {
        throw std::runtime_error("Could not open GeoTIFF: " + info.path.string());
    }

    std::vector<cv::Mat> channels;

    for (int b = 1; b <= info.bands; ++b) {
        cv::Mat band(info.height, info.width, CV_8UC1);

        CPLErr err = ds->GetRasterBand(b)->RasterIO(
            GF_Read,
            0, 0,
            info.width, info.height,
            band.data,
            info.width, info.height,
            GDT_Byte,
            0, 0
        );

        if (err != CE_None) {
            GDALClose(ds);
            throw std::runtime_error("GDAL read failed: " + info.path.string());
        }

        channels.push_back(band);
    }

    GDALClose(ds);

    cv::Mat rgb;
    cv::merge(channels, rgb);

    return rgb;
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

static cv::Mat create_feather_weight(const cv::Mat& mask) {
    cv::Mat mask_u8;
    mask.convertTo(mask_u8, CV_8U, 1.0 / 255.0);

    if (cv::countNonZero(mask_u8) == 0) {
        return cv::Mat::zeros(mask.size(), CV_32FC1);
    }

    cv::Mat dist;
    cv::distanceTransform(mask_u8, dist, cv::DIST_L2, 5);

    double min_val, max_val;
    cv::minMaxLoc(dist, &min_val, &max_val);

    if (max_val > 0) {
        dist /= max_val;
    }

    cv::pow(dist, 0.7, dist);

    cv::Mat weight;
    dist.convertTo(weight, CV_32FC1);

    weight.setTo(0, mask == 0);

    return weight;
}

static cv::Mat apply_weight_cuda(const cv::Mat& rgb, const cv::Mat& weight) {
    cv::Mat rgb_float;
    rgb.convertTo(rgb_float, CV_32F);

    std::vector<cv::Mat> weight_channels(rgb.channels(), weight);
    cv::Mat weight_3;
    cv::merge(weight_channels, weight_3);

    cv::cuda::GpuMat d_img;
    cv::cuda::GpuMat d_weight;
    cv::cuda::GpuMat d_weighted;

    d_img.upload(rgb_float);
    d_weight.upload(weight_3);

    cv::cuda::multiply(d_img, d_weight, d_weighted);

    cv::Mat weighted;
    d_weighted.download(weighted);

    return weighted;
}

static void write_output_geotiff(
    const fs::path& output_path,
    const cv::Mat& out,
    const TileInfo& ref,
    double minx,
    double maxy,
    double pixel_width,
    double pixel_height,
    int nodata
) {
    fs::create_directories(output_path.parent_path());

    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");

    if (!driver) {
        throw std::runtime_error("GTiff driver not available.");
    }

    char** options = nullptr;
    options = CSLSetNameValue(options, "COMPRESS", "LZW");
    options = CSLSetNameValue(options, "TILED", "YES");
    options = CSLSetNameValue(options, "BIGTIFF", "IF_SAFER");

    GDALDataset* ds = driver->Create(
        output_path.string().c_str(),
        out.cols,
        out.rows,
        out.channels(),
        GDT_Byte,
        options
    );

    CSLDestroy(options);

    if (!ds) {
        throw std::runtime_error("Could not create output: " + output_path.string());
    }

    double gt[6] = {
        minx,
        pixel_width,
        0.0,
        maxy,
        0.0,
        -pixel_height
    };

    ds->SetGeoTransform(gt);
    ds->SetProjection(ref.projection.c_str());

    std::vector<cv::Mat> channels;
    cv::split(out, channels);

    for (int b = 0; b < out.channels(); ++b) {
        GDALRasterBand* band = ds->GetRasterBand(b + 1);
        band->SetNoDataValue(nodata);

        CPLErr err = band->RasterIO(
            GF_Write,
            0, 0,
            out.cols, out.rows,
            channels[b].data,
            out.cols, out.rows,
            GDT_Byte,
            0, 0
        );

        if (err != CE_None) {
            GDALClose(ds);
            throw std::runtime_error("GDAL write failed: " + output_path.string());
        }
    }

    // Build overviews.
    int overview_list[5] = {2, 4, 8, 16, 32};
    ds->BuildOverviews("AVERAGE", 5, overview_list, 0, nullptr, nullptr, nullptr);

    GDALClose(ds);
}

int main(int argc, char** argv) {
    try {
        auto program_start = std::chrono::steady_clock::now();
        GDALAllRegister();

        auto args = parse_args(argc, argv);

        fs::path input_dir = args.at("--input");
        fs::path output_path = args.at("--output");

        int nodata = args.count("--nodata") ? std::stoi(args["--nodata"]) : 0;
        int gpu = args.count("--gpu") ? std::stoi(args["--gpu"]) : 0;

        check_cuda_or_throw(gpu);

        auto files = list_ortho_files(input_dir);

        if (files.empty()) {
            throw std::runtime_error("No *_ortho.tif files found.");
        }

        std::vector<TileInfo> infos;

        for (const auto& f : files) {
            infos.push_back(get_tile_info(f));
        }

        double pixel_width = infos[0].pixel_width;
        double pixel_height = infos[0].pixel_height;

        double minx = infos[0].minx;
        double maxx = infos[0].maxx;
        double miny = infos[0].miny;
        double maxy = infos[0].maxy;

        for (const auto& info : infos) {
            minx = std::min(minx, info.minx);
            maxx = std::max(maxx, info.maxx);
            miny = std::min(miny, info.miny);
            maxy = std::max(maxy, info.maxy);
        }

        int mosaic_width = static_cast<int>(std::ceil((maxx - minx) / pixel_width));
        int mosaic_height = static_cast<int>(std::ceil((maxy - miny) / pixel_height));
        int bands = infos[0].bands;

        std::cout << "Mosaic size: "
                  << mosaic_width << " x "
                  << mosaic_height << std::endl;

        cv::Mat accum = cv::Mat::zeros(mosaic_height, mosaic_width, CV_32FC(bands));
        cv::Mat weight_sum = cv::Mat::zeros(mosaic_height, mosaic_width, CV_32FC1);

        for (const auto& info : infos) {
            auto tile_start = std::chrono::steady_clock::now();

            std::cout << "C++ OpenCV CUDA feather weighting tile: "
                      << info.path << std::endl;

            cv::Mat rgb = read_tile_rgb(info);
            cv::Mat mask = valid_mask(rgb, nodata);

            if (cv::countNonZero(mask) == 0) {
                continue;
            }

            cv::Mat weight = create_feather_weight(mask);
            cv::Mat weighted = apply_weight_cuda(rgb, weight);

            int xoff = static_cast<int>(std::round((info.minx - minx) / pixel_width));
            int yoff = static_cast<int>(std::round((maxy - info.maxy) / pixel_height));

            int dst_x1 = std::max(0, xoff);
            int dst_y1 = std::max(0, yoff);
            int dst_x2 = std::min(mosaic_width, xoff + info.width);
            int dst_y2 = std::min(mosaic_height, yoff + info.height);

            int src_x1 = std::max(0, -xoff);
            int src_y1 = std::max(0, -yoff);
            int src_x2 = src_x1 + (dst_x2 - dst_x1);
            int src_y2 = src_y1 + (dst_y2 - dst_y1);

            if (dst_x2 <= dst_x1 || dst_y2 <= dst_y1) {
                continue;
            }

            cv::Rect src_roi(src_x1, src_y1, src_x2 - src_x1, src_y2 - src_y1);
            cv::Rect dst_roi(dst_x1, dst_y1, dst_x2 - dst_x1, dst_y2 - dst_y1);

            accum(dst_roi) += weighted(src_roi);
            weight_sum(dst_roi) += weight(src_roi);

            auto tile_end = std::chrono::steady_clock::now();
            double tile_sec = std::chrono::duration<double>(tile_end - tile_start).count();

            std::cout << "Feather blending tile elapsed seconds: " << tile_sec << std::endl;
        }

        std::cout << "Finalizing blended mosaic..." << std::endl;

        cv::Mat out = cv::Mat::zeros(mosaic_height, mosaic_width, CV_8UC(bands));

        std::vector<cv::Mat> accum_channels;
        cv::split(accum, accum_channels);

        std::vector<cv::Mat> out_channels;

        for (int c = 0; c < bands; ++c) {
            cv::Mat band_float = cv::Mat::zeros(mosaic_height, mosaic_width, CV_32FC1);

            for (int y = 0; y < mosaic_height; ++y) {
                const float* acc_row = accum_channels[c].ptr<float>(y);
                const float* w_row = weight_sum.ptr<float>(y);
                float* out_row = band_float.ptr<float>(y);

                for (int x = 0; x < mosaic_width; ++x) {
                    if (w_row[x] > 0.0f) {
                        out_row[x] = acc_row[x] / std::max(w_row[x], 1e-6f);
                    } else {
                        out_row[x] = 0.0f;
                    }
                }
            }

            cv::Mat band_u8;
            band_float.convertTo(band_u8, CV_8UC1);
            out_channels.push_back(band_u8);
        }

        cv::merge(out_channels, out);

        write_output_geotiff(
            output_path,
            out,
            infos[0],
            minx,
            maxy,
            pixel_width,
            pixel_height,
            nodata
        );

        auto program_end = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(program_end - program_start).count();

        std::cout << "C++ OpenCV CUDA feather blended mosaic created: "
                  << output_path << std::endl;
        std::cout << "blend_mosaic_feather_cuda elapsed seconds: " << elapsed_sec << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
