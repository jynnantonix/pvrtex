/*==========================================================================*/
/*                                                                          */
/* @file            Compressor.cpp                                          */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)              */
/* @date            2012/11/05                                              */
/* @version         0.4                                                     */
/* @brief           Class implementation for pvr texture compressor         */
/*                                                                          */
/*==========================================================================*/

#include <iostream>
#include <cfloat>

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
  static const float kModulationValues[] = { 0.0f, 0.375f, 0.625f, 1.0f };
  Eigen::MatrixXf result(height_, width_);
  Eigen::Vector3f o, d, b;
  float delta, delta_min;
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      /* Get the original, dark, and bright pixel colors */
      o = util::MakeColorVector(orig(y, x), util::PVR888).cast<float>();
      d = util::MakeColorVector(dark(y, x), util::PVR888).cast<float>();
      b = util::MakeColorVector(bright(y, x), util::PVR888).cast<float>();
        
      /* Set the appropriate modulation value */
      delta_min = FLT_MAX;
      for (int k = 0; k < 4; ++k) {
        delta = (util::lerp<Eigen::Vector3f>(d, b, kModulationValues[k]) -
                 o).squaredNorm();
        if (delta < delta_min) {
          result(y, x) = kModulationValues[k];
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
  const Eigen::MatrixXi offset = Eigen::MatrixXi::Constant(height_>>2,
                                                           width_>>2,
                                                           0x30303030);
  Eigen::MatrixXi dark = result - offset;
  Eigen::MatrixXi bright = result + offset;
    
  /* Iterative optimization */
  util::DATA_FORMAT df = util::ImageToData(format_);
  Optimizer opt(bits, dark, bright, Optimizer::SVD, df);
  for (int k = 0; k < 12; ++k) {
    /* Least squares optimization */
    opt.Optimize(ComputeModulation(bits,
                                   util::Upscale4x4(opt.dark(), df),
                                   util::Upscale4x4(opt.bright(), df)));
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
