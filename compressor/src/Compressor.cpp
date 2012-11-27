/*==========================================================================*/
/*                                                                          */
/* @file            Compressor.cpp                                          */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)              */
/* @date            2012/11/05                                              */
/* @version         0.3                                                     */
/* @brief           Class implementation for pvr texture compressor         */
/*                                                                          */
/*==========================================================================*/

#include "../inc/Compressor.h"

namespace pvrtex {
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
      o = util::MakeColorVector(orig(y, x)).cast<float>();
      d = util::MakeColorVector(dark(y, x)).cast<float>();
      b = util::MakeColorVector(bright(y, x)).cast<float>();
        
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
  
void Compressor::Compress(unsigned int *out, IMAGE_FORMAT format) {
  /* Create the matrix with the original data */
  Eigen::MatrixXi bits(height_, width_);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      bits(y, x) = data_[y*width_ + x];
    }
  }
    
  /* Convert to YUV */
  Eigen::MatrixXi yuv_bits = util::RGBtoYUV(bits);
    
  /* Wavelet filter */
  Eigen::MatrixXi result = util::Downscale(yuv_bits);
    
  /* Initial dark and bright prototypes */
  const Eigen::MatrixXi offset = Eigen::MatrixXi::Constant(height_>>2,
                                                           width_>>2,
                                                           0x30303030);
  Eigen::MatrixXi dark = result - offset;
  Eigen::MatrixXi bright = result + offset;
    
  /* Iterative optimization */
  Optimizer opt(yuv_bits, dark, bright);
  for (int k = 0; k < 12; ++k) {
    /* Least squares optimization */
    opt.Optimize(ComputeModulation(yuv_bits,
                                   util::Upscale4x4(opt.dark()),
                                   util::Upscale4x4(opt.bright())));
  }
    
  /* Write the final output */
  result = util::ModulateImage(util::Upscale4x4(opt.dark()),
                               util::Upscale4x4(opt.bright()),
                               opt.mod());

  /* Convert back to RGB */
  result = util::YUVtoRGB(result);
  for (int y = 0; y < result.rows(); ++y) {
    for (int x = 0; x < result.cols(); ++x) {
      out[y*width_ + x] = result(y, x);
    }
  }
    
  /* Get the error */
  std::cout << "RMS Error: " << util::ComputeError(bits, result) << std::endl;
}
  
} /* namespace pvrtex */
