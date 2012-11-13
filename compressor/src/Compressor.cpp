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
  
  inline IMAGE_FORMAT Compressor::format()
  {
    return format_;
  }
  
  inline BYTE* Compressor::data()
  {
    return data_;
  }
  
  inline BYTE Compressor::MakeAlpha(unsigned int p)
  {
    return ((p & FI_RGBA_ALPHA_MASK)>>FI_RGBA_ALPHA_SHIFT);
  }
  
  inline BYTE Compressor::MakeRed(unsigned int p)
  {
    return ((p & FI_RGBA_RED_MASK)>>FI_RGBA_RED_SHIFT);
  }
  
  inline BYTE Compressor::MakeGreen(unsigned int p)
  {
    return ((p & FI_RGBA_GREEN_MASK)>>FI_RGBA_GREEN_SHIFT);
  }
  
  inline BYTE Compressor::MakeBlue(unsigned int p)
  {
    return ((p & FI_RGBA_BLUE_MASK)>>FI_RGBA_BLUE_SHIFT);
  }
  
  void Compressor::Compress(BYTE *out, IMAGE_FORMAT format)
  {
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
    
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        int idx = 4 * (y*width_ + x);
        unsigned int pixel = bits(y, x);
        out[idx] = MakeAlpha(pixel);
        out[idx+1] = MakeRed(pixel);
        out[idx+2] = MakeGreen(pixel);
        out[idx+3] = MakeBlue(pixel);
      }
    }
    
  }
  
  void Compressor::WriteToFile(const char *filename)
  {
    
  }
} /* namespace PVRTEX */

