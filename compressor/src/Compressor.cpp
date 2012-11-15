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
  Compressor::Compressor(int w, int h, int sw, IMAGE_FORMAT f, BYTE *d)
  :
  width_(w),
  height_(h),
  scan_width_(sw),
  format_(f),
  data_(d)
  {
  }
  
  Compressor::~Compressor()
  {
  }
  
  inline void Compressor::set_width(int w)
  {
    width_ = w;
  }
  
  inline void Compressor::set_height(int h)
  {
    height_ = h;
  }
  
  inline void Compressor::set_scan_width(int sw)
  {
    scan_width_ = sw;
  }
  
  inline void Compressor::set_data(BYTE *d)
  {
    data_ = d;
  }
  
  inline int Compressor::width()
  {
    return width_;
  }
  
  inline int Compressor::height()
  {
    return height_;
  }
  
  inline int Compressor::scan_width()
  {
    return scan_width_;
  }
  
  inline Compressor::IMAGE_FORMAT Compressor::format()
  {
    return format_;
  }
  
  inline BYTE* Compressor::data()
  {
    return data_;
  }
  
  Eigen::MatrixXf Compressor::ComputeModulation(Eigen::MatrixXi orig,
                                                Eigen::MatrixXi dark,
                                                Eigen::MatrixXi bright) {
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
          delta = (((1-kModulationValues[k])*d +
                    kModulationValues[k]*b) - o).squaredNorm();
          if (delta < delta_min) {
            result(y, x) = kModulationValues[k];
            delta_min = delta;
          }
        }
      }
    }
    
    return result;
  }
  
  void Compressor::Compress(BYTE *out, IMAGE_FORMAT format) {
    //unsigned int *bits = (unsigned int*) malloc(m_height * m_scanWidth);
    //bits = (unsigned int*)m_data;
    //memcpy((void*)out, (void*)m_data, m_height * m_scanWidth);
    
    /* Create the matrix with the original data */
    Eigen::MatrixXi bits(height_, width_);
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        unsigned int pixel = 0;
        int idx = 4 * (y*width_ + x);
        for (int k = 0; k < 4; ++k) {
          pixel = (pixel << 8) | data_[idx + k];
        }
        bits(y, x) = pixel;
      }
    }
    
    /* Wavelet filter */
    Wavelet filter(Wavelet::BASIC);
    Eigen::MatrixXi result = filter.Upscale(filter.Downscale(bits));
    
    /* Initial dark and bright prototypes */
    Eigen::MatrixXi dark = Eigen::MatrixXi::Constant(height_>>2,
                                                     width_>>2,
                                                     0x20202020);
    Eigen::MatrixXi bright = Eigen::MatrixXi::Constant(height_>>2,
                                                       width_>>2,
                                                       0xDFDFDFDF);
    
    /* Iterative optimization */
    for (int k = 0; k < 10; ++k) {
      /* Upscale images */
      dark = filter.Upscale(dark);
      bright = filter.Upscale(bright);
      
      /* Calculate the initial modulation image */
      Eigen::MatrixXf mod = ComputeModulation(bits, dark, bright);
      
      /* Least squares optimization */
      Optimizer opt(bits, mod);
      opt.Optimize();
      dark = opt.dark();
      bright = opt.bright();
    }
    
    /* Write the final output */
    for (int y = 0; y < dark.rows(); ++y) {
      for (int x = 0; x < dark.cols(); ++x) {
        int idx = 4*(y*width_ + x);
        unsigned int pixel = dark(y, x);
        out[idx] = util::MakeAlpha(pixel);
        out[idx+1] = util::MakeRed(pixel);
        out[idx+2] = util::MakeGreen(pixel);
        out[idx+3] = util::MakeBlue(pixel);
      }
    }
    
    for (int y = 0; y < bright.rows(); ++y) {
      for (int x = 0; x < bright.cols(); ++x) {
        int idx = (1024*512)+(4*(y*width_ + x));
        unsigned int pixel = bright(y, x);
        out[idx] = util::MakeAlpha(pixel);
        out[idx+1] = util::MakeRed(pixel);
        out[idx+2] = util::MakeGreen(pixel);
        out[idx+3] = util::MakeBlue(pixel);
      }
    }
    
  }
  
  void Compressor::WriteToFile(const char *filename)
  {
    
  }
} /* namespace pvrtex */

