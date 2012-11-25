/*=========================================================================*/
/*                                                                         */
/* @file            Util.h                                                 */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/13                                             */
/* @version         0.1                                                    */
/* @brief           Preprocessor definitions and utility functions for     */
/*                  handling pixels                                        */
/*                                                                         */
/*=========================================================================*/

#ifndef pvrtex_Util_h
#define pvrtex_Util_h

#include <cmath>
#include <FreeImage.h>
#include <Eigen/Dense>

#define ONE_FOURTH          0.25f
#define THREE_EIGHTHS       0.375f
#define FIVE_EIGHTHS        0.625f
namespace pvrtex {
  namespace util {
    inline int Clamp(int x, int a, int b) {
      return static_cast<int>(fminf(fmaxf(static_cast<float>(x),
                                          static_cast<float>(a)),
                                    static_cast<float>(b)));
    }
    
    template<typename T>
    inline T lerp(const T &a, const T &b, const float delta) {
      return ((1.0f-delta) * a) + (delta*b);
    }
    
    inline int MakeAlpha(unsigned int p) {
      return ((p & FI_RGBA_ALPHA_MASK)>>FI_RGBA_ALPHA_SHIFT);
    }
    
    inline int MakeRed(unsigned int p) {
      return ((p & FI_RGBA_RED_MASK)>>FI_RGBA_RED_SHIFT);
    }
    
    inline int MakeGreen(unsigned int p) {
      return ((p & FI_RGBA_GREEN_MASK)>>FI_RGBA_GREEN_SHIFT);
    }
    
    inline int MakeBlue(unsigned int p) {
      return ((p & FI_RGBA_BLUE_MASK)>>FI_RGBA_BLUE_SHIFT);
    }
    
    inline Eigen::Vector4i MakeColorVector(unsigned int p) {
      return Eigen::Vector4i(MakeAlpha(p),
                             MakeRed(p),
                             MakeGreen(p),
                             MakeBlue(p));
    }
    
    inline unsigned int MakeRGBA(const Eigen::Vector4i &p) {
      return ((Clamp(p(0), 0, 255)<<FI_RGBA_ALPHA_SHIFT) |
              (Clamp(p(1), 0, 255)<<FI_RGBA_RED_SHIFT) |
              (Clamp(p(2), 0, 255)<<FI_RGBA_GREEN_SHIFT) |
              (Clamp(p(3), 0, 255)<<FI_RGBA_BLUE_SHIFT));
    }
    
    inline unsigned int MakeRGBA(unsigned int a, unsigned int r,
                                 unsigned int g, unsigned int b) {
      return (((a & 0x000000FF)<<FI_RGBA_ALPHA_SHIFT) |
              ((r & 0x000000FF)<<FI_RGBA_RED_SHIFT) |
              ((g & 0x000000FF)<<FI_RGBA_GREEN_SHIFT) |
              ((b & 0x000000FF)<<FI_RGBA_BLUE_SHIFT));
    }
    
    static const Eigen::MatrixXi ModulateImage(const Eigen::MatrixXi &dark,
                                               const Eigen::MatrixXi &bright,
                                               const Eigen::MatrixXf &mod) {
      Eigen::MatrixXi result(mod.rows(), mod.cols());
      Eigen::Vector4f d, b;
      Eigen::Vector4i r;
      for (int y = 0; y < mod.rows(); ++y) {
        for (int x = 0; x < mod.cols(); ++x) {
          d = MakeColorVector(dark(y, x)).cast<float>();
          b = MakeColorVector(bright(y, x)).cast<float>();
          r = lerp<Eigen::Vector4f>(d,b, mod(y, x)).cast<int>();
          result(y, x) = MakeRGBA(r);
        }
      }
      return result;
    }
    
    static float ComputeError(const Eigen::MatrixXi &orig,
                              const Eigen::MatrixXi &compressed) {
      float result = 0.0f;
      Eigen::Vector4i o, c;
      for (int y = 0; y < orig.rows(); ++y) {
        for (int x = 0; x < orig.cols(); ++x) {
          result += (MakeColorVector(orig(y, x)) -
                     MakeColorVector(compressed(y, x))).squaredNorm();
        }
      }
      
      return sqrtf(result / (orig.rows()*orig.cols()));
    }
    
    static const Eigen::MatrixXi Downscale(Eigen::MatrixXi &orig) {
      Eigen::MatrixXi result(orig.rows()>>2, orig.cols()>>2);
      for (int j = 0; j < result.rows(); ++j) {
        for (int i = 0; i < result.cols(); ++i) {
          result(j, i) = orig(j*4, i*4);
        }
      }
      
      return result;
    }
    
    static const Eigen::MatrixXi Upscale4x4(Eigen::MatrixXi &orig) {
      Eigen::MatrixXi result(orig.rows() * 4, orig.cols() * 4);
      int x, y, x1, y1;
      float x_diff, y_diff;
      Eigen::Vector4i a, b, c, d, color;
      for (int j = 0; j < result.rows(); ++j) {
        for (int i = 0; i < result.cols(); ++i) {
          /* Get the indices of the four neighboring pixels */
          x = Clamp(i-2, 0, result.cols()-1)>>2;   /* (i/4) */
          y = Clamp(j-2, 0, result.rows()-1)>>2;   /* (j/4) */
          x1 = x+1;
          y1 = y+1;
          x_diff = (Clamp(i-2, 0, result.cols()) - (x*4)) * ONE_FOURTH;
          y_diff = (Clamp(j-2, 0, result.rows()) - (y*4)) * ONE_FOURTH;
          
          /* Get the colors of the neighboring pixels */
          a = MakeColorVector(orig(y, x));
          b = MakeColorVector(orig(y, x1));
          c = MakeColorVector(orig(y1, x));
          d = MakeColorVector(orig(y1, x1));
          
          /* Do the bilinear interpolation for each channel */
          for (int k = 0; k < 4; ++k) {
            color(k) = static_cast<int>(lerp(lerp(static_cast<float>(a(k)),
                                                  static_cast<float>(b(k)),
                                                  x_diff),
                                             lerp(static_cast<float>(c(k)),
                                                  static_cast<float>(d(k)),
                                                  x_diff),
                                             y_diff));
          }
          
          /* Store the result */
          result(j, i) = MakeRGBA(color);
        }
      }
      
      return result;
    }

  } /* namespace util */
} /* namespace pvrtex */


#endif /* defined(pvrtex_Util_h) */
