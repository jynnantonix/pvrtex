/*==========================================================================*/
/*                                                                          */
/* @file            Compressor.cpp                                          */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)              */
/* @date            2012/11/05                                              */
/* @version         0.2                                                     */
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
    Eigen::MatrixXf result(height_, width_);
    Eigen::Vector4i o, d, b;
    int delta_d, delta_b, delta_38, delta_58, delta_min;
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        /* Get the original, dark, and bright pixel colors */
        o = MakeColorVector(orig(y, x));
        d = MakeColorVector(dark(y, x));
        b = MakeColorVector(bright(y, x));
        
        /* Get the error for each modulation value and find the minimum*/
        delta_d = static_cast<int>((d - o).squaredNorm());
        delta_b = static_cast<int>((b - o).squaredNorm());
        delta_38 = static_cast<int>(
                      ((FIVE_EIGHTHS*d)+(THREE_EIGHTHS*b) - o).squaredNorm());
        delta_58 = static_cast<int>(
                      ((THREE_EIGHTHS*d)+(FIVE_EIGHTHS*b) - o).squaredNorm());
        delta_min = static_cast<int>(fminf(fminf(static_cast<float>(delta_d),
                                                 static_cast<float>(delta_b)),
                                           fminf(static_cast<float>(delta_38),
                                                 static_cast<float>(delta_58))));
        
        /* Now set the modbit accordingly */
        if (delta_min == delta_d) {
          result(y, x) = 0.0f;
        } else if (delta_min == delta_38) {
          result(y, x) = THREE_EIGHTHS;
        } else if (delta_min == delta_58) {
          result(y, x) = FIVE_EIGHTHS;
        } else if (delta_min == delta_b) {
          result(y, x) = 1.0f;
        } else {
          std::cerr << "Invalid modulation error found: " << delta_min
                    << std::endl;
          std::cerr << "delta_d: " << delta_d << std::endl;
          std::cerr << "delta_38: " << delta_38 << std::endl;
          std::cerr << "delta_58: " << delta_58 << std::endl;
          std::cerr << "delta_b: " << delta_b << std::endl;
          std::cerr << "Position is <" << y << "," << x << ">" << std::endl;
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
    
    Eigen::MatrixXi dark = Eigen::MatrixXi::Constant(height_, width_, 0);
    Eigen::MatrixXi bright = Eigen::MatrixXi::Constant(height_,
                                                       width_,
                                                       0xFFFFFFFF);
    
    /* Write the final output */
    for (int y = 0; y < result.rows(); ++y) {
      for (int x = 0; x < result.cols(); ++x) {
        int idx = 4 * (y*width_ + x);
        unsigned int pixel = result(y, x);
        out[idx] = pvrtex::MakeAlpha(pixel);
        out[idx+1] = pvrtex::MakeRed(pixel);
        out[idx+2] = pvrtex::MakeGreen(pixel);
        out[idx+3] = pvrtex::MakeBlue(pixel);
      }
    }
    
  }
  
  void Compressor::WriteToFile(const char *filename)
  {
    
  }
} /* namespace pvrtex */

