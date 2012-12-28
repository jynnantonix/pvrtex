/*==========================================================================*/
/*                                                                          */
/* @file            Compressor.cpp                                          */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)              */
/* @date            2012/11/05                                              */
/* @version         0.4                                                     */
/* @brief           Class implementation for pvr texture compressor         */
/*                                                                          */
/*==========================================================================*/

#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <limits>

#include "../inc/Compressor.h"
#include "../inc/Util.h"
#include "../inc/Optimizer.h"

namespace pvrtex {
/* Initializes private variables to invalid values because the default */
/* constructor should never be called. */
Compressor::Compressor() :
width_(-1),
height_(-1),
format_(PVR_UNDEFINED),
data_(NULL)
{
}
  
Compressor::Compressor(int w, int h, IMAGE_FORMAT f, unsigned int *d)
    :
    width_(w),
    height_(h),
    format_(f),
    data_(d)
{
}
  
Compressor::~Compressor()
{
}
  
Eigen::MatrixXf Compressor::ComputeModulation(const Eigen::MatrixXi &orig,
                                              const Eigen::MatrixXi &dark,
                                              const Eigen::MatrixXi &bright) {
  Eigen::VectorXf modulation_values;
  Eigen::MatrixXf result(height_, width_);
  
  if (format_ == YUV_EXT_4BPP || format_ == YUV_2BPP) {
    modulation_values = Eigen::VectorXf(8);
    modulation_values << 0.0f, 0.14285f, 0.28571f, 0.42847f, 0.57142f,
    0.71428f, 0.85714f, 1.0f;
  } else {
    modulation_values = Eigen::VectorXf(4);
    modulation_values << 0.0f, 0.375f, 0.625f, 1.0f;
  }
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      /* Get the original, dark, and bright pixel colors */
      Eigen::Vector3f o, d, b;
      o = util::MakeColorVector(orig(y, x), util::PVR888).cast<float>();
      d = util::MakeColorVector(dark(y, x), util::PVR888).cast<float>();
      b = util::MakeColorVector(bright(y, x), util::PVR888).cast<float>();
      
      if (format_ == YUV_EXT_4BPP || format_ == YUV_2BPP) {
        o(1) = d(1) = b(1) =  0.0f;
        o(2) = d(2) = b(2) = 0.0f;
      }
      
      /* Set the appropriate modulation value */
      float delta, delta_min;
      delta_min = std::numeric_limits<float>::max();
      for (int k = 0; k < modulation_values.size(); ++k) {
        delta = static_cast<float>((util::lerp<Eigen::Vector3f>(d, b,
                    modulation_values(k)) - o).squaredNorm());
        if (delta < delta_min) {
          result(y, x) = modulation_values(k);
          delta_min = delta;
        }
      }
    }
  }
    
  return result;
}
  
void Compressor::Compress(unsigned int *out) {
  /* Create the matrix with the original data */
  Eigen::MatrixXi orig(height_, width_);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      orig(y, x) = data_[y*width_ + x];
    }
  }

  /* Get the internal data format */
  util::DATA_FORMAT df = util::ImageToData(format_);
  
  /* Convert to YUV if necessary */
  Eigen::MatrixXi bits;
  if (format_ == PVRTC_4BPP || format_ == PVRTC_2BPP) {
    bits = orig;
  } else {
    bits = util::RGBtoYUV(orig);
  }
    
  /* Wavelet filter */
  Eigen::MatrixXi result = util::Downscale(bits);
    
  /* Initial dark and bright prototypes */
  Eigen::Vector3i offset(32, 32, 32);
  Eigen::MatrixXi dark(result.rows(), result.cols());
  Eigen::MatrixXi bright(result.rows(), result.cols());

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for(int y = 0; y < result.rows(); ++y) {
    for(int x = 0; x < result.cols(); ++x) {
      Eigen::Vector3i color = util::MakeColorVector(result(y, x), util::PVR888);
      dark(y, x) = util::MakeRGB(color - offset, df);
      bright(y, x) = util::MakeRGB(color + offset, df);
    }
  }
  
  // Eigen::MatrixXi dark = result - offset;
  // Eigen::MatrixXi bright = result + offset;
    
  /* Iterative optimization */
  Optimizer opt(bits, dark, bright, Optimizer::SVD, df);
  float prev_err = std::numeric_limits<float>::max();
  float curr_err = std::numeric_limits<float>::max();
  for (int k = 0; (k < 4 || (prev_err - curr_err) > 1e-10) && (k < 20); ++k) {
    /* Least squares optimization */
    opt.Optimize(ComputeModulation(bits,
                                   util::Upscale4x4(opt.dark(), df),
                                   util::Upscale4x4(opt.bright(), df)));

    /* Get the current error */
    result = util::ModulateImage(util::Upscale4x4(opt.dark(), df),
                                 util::Upscale4x4(opt.bright(), df),
                                 opt.mod());
    prev_err = curr_err;
    curr_err = util::ComputeError(orig, result);
  }
    
  /* Write the final output */
  result = util::ModulateImage(util::Upscale4x4(opt.dark(), df),
                               util::Upscale4x4(opt.bright(), df),
                               opt.mod());

  /* Convert back to RGB if necessary */
  if (!(format_ == PVRTC_4BPP || format_ == PVRTC_2BPP)) {
    result = util::YUVtoRGB(result);
  } 
  for (int y = 0; y < result.rows(); ++y) {
    for (int x = 0; x < result.cols(); ++x) {
      out[y*width_ + x] = result(y, x);
    }
  }
    
  /* Get the error */
  std::cout << "RMS Error: " << util::ComputeError(orig, result) << std::endl;
}
  
} /* namespace pvrtex */
