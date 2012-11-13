/*=========================================================================*/
/*                                                                         */
/* @file            Wavelet.cpp                                            */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/13                                             */
/* @version         0.1                                                    */
/* @brief           Class implementation for wavelet filters               */
/*                                                                         */
/*=========================================================================*/

#include "../inc/Wavelet.h"

namespace pvrtex {
  // simple wavelet filter
  static const float kBasicFilter[] = { 0.5f, 0.5f };
  
  Wavelet::Wavelet(FILTER f) :
  filter_type_(f)
  {
    switch (filter_type_) {
      case BASIC: {
        filter_ = kBasicFilter;
        break;
      }
      default:
        break;
    }
  }
  
  Wavelet::~Wavelet() {
  }
  
  void Wavelet::Init() {
  }
  
  Eigen::MatrixXi Wavelet::Downscale(Eigen::MatrixXi orig) {
    Eigen::MatrixXi result(orig.rows()>>2, orig.cols()>>2);
    for (int j = 0; j < result.rows(); ++j) {
      for (int i = 0; i < result.cols(); ++i) {
        result(j, i) = orig(j*4, i*4);
      }
    }
    
    return result;
  }
  
  Eigen::MatrixXi Wavelet::Upscale(Eigen::MatrixXi orig) {
    Eigen::MatrixXi result(orig.rows() * 4, orig.cols() * 4);
    int x, y, x1, y1;
    float x_diff, y_diff;
    Eigen::Vector4i a, b, c, d, color;
    for (int j = 0; j < result.rows(); ++j) {
      for (int i = 0; i < result.cols(); ++i) {
        /* Get the indices of the four neighboring pixels */
        x = (i>>2);   /* (i/4) */
        y = (j>>2);   /* (j/4) */
        x1 = pvrtex::Clamp(x+1, 0, orig.cols()-1);
        y1 = pvrtex::Clamp(y+1, 0, orig.rows()-1);
        x_diff = (i - (x*4)) * ONE_FOURTH;
        y_diff = (j - (y*4)) * ONE_FOURTH;
        
        /* Get the colors of the neighboring pixels */
        a = pvrtex::MakeColorVector(orig(y, x));
        b = pvrtex::MakeColorVector(orig(y, x1));
        c = pvrtex::MakeColorVector(orig(y1, x));
        d = pvrtex::MakeColorVector(orig(y1, x1));
        
        /* Do the bilinear interpolation for each channel */
        for (int k = 0; k < 4; ++k) {
          color(k) = static_cast<int>(
                         pvrtex::lerp(pvrtex::lerp(static_cast<float>(a(k)),
                                                   static_cast<float>(b(k)),
                                                   x_diff),
                                      pvrtex::lerp(static_cast<float>(c(k)),
                                                   static_cast<float>(d(k)),
                                                   x_diff),
                         y_diff));
        }
        
        /* Store the result */
        result(j, i) = pvrtex::MakeRGBA(color);
      }
    }
    
    return result;
  }
}