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
    Eigen::Vector4f o, d, b;
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
          delta = (util::lerp<Eigen::Vector4f>(d, b, kModulationValues[k]) -
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
    //unsigned int *bits = (unsigned int*) malloc(m_height * m_scanWidth);
    //bits = (unsigned int*)m_data;
    //memcpy((void*)out, (void*)m_data, m_height * m_scanWidth);
    
    /* Create the matrix with the original data */
    Eigen::MatrixXi bits(height_, width_);
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        bits(y, x) = data_[y*width_ + x];
      }
    }
    
    /* Wavelet filter */
    Wavelet filter(Wavelet::BASIC);
    Eigen::MatrixXi result = filter.Downscale(bits);
    
    /* Initial dark and bright prototypes */
    const Eigen::MatrixXi offset = Eigen::MatrixXi::Constant(height_>>2,
                                                             width_>>2,
                                                             0x20202020);
    Eigen::MatrixXi dark = result - offset;
    Eigen::MatrixXi bright = result + offset;
    
    /* Iterative optimization */
    Eigen::MatrixXf mod;
    for (int k = 0; k < 12; ++k) {
      /* Calculate the modulation image */
      mod = ComputeModulation(bits,
                              filter.Upscale(dark),
                              filter.Upscale(bright));
      
      /* Least squares optimization */
      Optimizer opt(bits, mod);
      opt.Optimize();
      dark = opt.dark();
      bright = opt.bright();
    }
    
    /* Write the final output */
    result = ModulateImage(filter.Upscale(dark),
                           filter.Upscale(bright),
                           mod);
    for (int y = 0; y < result.rows(); ++y) {
      for (int x = 0; x < result.cols(); ++x) {
        out[y*width_ + x] = result(y, x);
      }
    }
    
    /* Get the error */
    std::cout << "RMS Error: " << ComputeError(bits, result) << std::endl;
  }
  
  Eigen::MatrixXi Compressor::ModulateImage(const Eigen::MatrixXi &dark,
                                            const Eigen::MatrixXi &bright,
                                            const Eigen::MatrixXf &mod) {
    Eigen::MatrixXi result(height_, width_);
    Eigen::Vector4f d, b;
    Eigen::Vector4i r;
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        d = util::MakeColorVector(dark(y, x)).cast<float>();
        b = util::MakeColorVector(bright(y, x)).cast<float>();
        r = util::lerp<Eigen::Vector4f>(d,b, mod(y, x)).cast<int>();
        result(y, x) = util::MakeRGBA(r);
      }
    }
    return result;
  }
  
  float Compressor::ComputeError(const Eigen::MatrixXi &orig,
                                 const Eigen::MatrixXi &compressed) {
    float result = 0.0f;
    Eigen::Vector4i o, c;
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        result += (util::MakeColorVector(orig(y, x)) -
                   util::MakeColorVector(compressed(y, x))).squaredNorm();
      }
    }
    
    return sqrtf(result / (width_*height_));
  }
  
  void Compressor::WriteToFile(const char *filename)
  {
    
  }
} /* namespace pvrtex */

