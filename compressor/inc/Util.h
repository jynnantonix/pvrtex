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
    
    inline int MakeRed(unsigned int p) {
      return ((p & FI_RGBA_RED_MASK)>>FI_RGBA_RED_SHIFT);
    }
    
    inline int MakeGreen(unsigned int p) {
      return ((p & FI_RGBA_GREEN_MASK)>>FI_RGBA_GREEN_SHIFT);
    }
    
    inline int MakeBlue(unsigned int p) {
      return ((p & FI_RGBA_BLUE_MASK)>>FI_RGBA_BLUE_SHIFT);
    }
    
    inline Eigen::Vector3i MakeColorVector(unsigned int p) {
      return Eigen::Vector3i(MakeRed(p),
                             MakeGreen(p),
                             MakeBlue(p));
    }
    
    inline unsigned int MakeRGB(const Eigen::Vector3i &p) {
      return (((0x000000FF)<<FI_RGBA_ALPHA_SHIFT) |
              (Clamp(p(0), 0, 255)<<FI_RGBA_RED_SHIFT) |
              (Clamp(p(1), 0, 255)<<FI_RGBA_GREEN_SHIFT) |
              (Clamp(p(2), 0, 255)<<FI_RGBA_BLUE_SHIFT));
    }
    
    inline unsigned int MakeRGB(unsigned int r, unsigned int g,
                                unsigned int b) {
      return (((0x000000FF)<<FI_RGBA_ALPHA_SHIFT) |
              ((r & 0x000000FF)<<FI_RGBA_RED_SHIFT) |
              ((g & 0x000000FF)<<FI_RGBA_GREEN_SHIFT) |
              ((b & 0x000000FF)<<FI_RGBA_BLUE_SHIFT));
    }
    
    inline int Make565Red(unsigned int p) {
      return (((p & FI16_565_RED_MASK)>>FI16_565_RED_SHIFT)<<3);
    }
    
    inline int Make565Green(unsigned int p) {
      return (((p & FI16_565_GREEN_MASK)>>FI16_565_GREEN_SHIFT)<<2);
    }
    
    inline int Make565Blue(unsigned int p) {
      return (((p & FI16_565_BLUE_MASK)>>FI16_565_BLUE_SHIFT)<<3);
    }
    
    inline Eigen::Vector3i Make565ColorVector(unsigned int p) {
      return Eigen::Vector3i(Make565Red(p),
                             Make565Green(p),
                             Make565Blue(p));
    }
    
    inline unsigned int Make565RGB(const Eigen::Vector3i &p) {
      return (((Clamp(p(0), 0, 255)>>3) << FI16_565_RED_SHIFT) |
              ((Clamp(p(1), 0, 255)>>2) << FI16_565_GREEN_SHIFT) |
              ((Clamp(p(2), 0, 255)>>3) << FI16_565_BLUE_SHIFT));
    }
    
    static const Eigen::MatrixXi ModulateImage(const Eigen::MatrixXi &dark,
                                               const Eigen::MatrixXi &bright,
                                               const Eigen::MatrixXf &mod) {
      Eigen::MatrixXi result(mod.rows(), mod.cols());
      Eigen::Vector3f d, b;
      Eigen::Vector3i r;
      for (int y = 0; y < mod.rows(); ++y) {
        for (int x = 0; x < mod.cols(); ++x) {
          d = MakeColorVector(dark(y, x)).cast<float>();
          b = MakeColorVector(bright(y, x)).cast<float>();
          r = lerp<Eigen::Vector3f>(d,b, mod(y, x)).cast<int>();
          result(y, x) = MakeRGB(r);
        }
      }
      return result;
    }
    
    static float ComputeError(const Eigen::MatrixXi &orig,
                              const Eigen::MatrixXi &compressed) {
      float result = 0.0f;
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
    
    static const Eigen::MatrixXi Upscale4x4(const Eigen::MatrixXi &orig) {
      Eigen::MatrixXi result(orig.rows() * 4, orig.cols() * 4);
      int x, y, x1, y1;
      float x_diff, y_diff;
      Eigen::Vector3f a, b, c, d;
      Eigen::Vector3i color;
      for (int j = 0; j < result.rows(); ++j) {
        for (int i = 0; i < result.cols(); ++i) {
          /* Get the indices of the four neighboring pixels */
          x = Clamp(i-2, 0, result.cols()-1)>>2;   /* (i/4) */
          y = Clamp(j-2, 0, result.rows()-1)>>2;   /* (j/4) */
          x1 = Clamp(x+1, 0, orig.cols()-1);
          y1 = Clamp(y+1, 0, orig.rows()-1);
          x_diff = (Clamp(i-2, 0, result.cols()) - (x*4)) * ONE_FOURTH;
          y_diff = (Clamp(j-2, 0, result.rows()) - (y*4)) * ONE_FOURTH;
          
          /* Get the colors of the neighboring pixels */
          a = Make565ColorVector(orig(y, x)).cast<float>();
          b = Make565ColorVector(orig(y, x1)).cast<float>();
          c = Make565ColorVector(orig(y1, x)).cast<float>();
          d = Make565ColorVector(orig(y1, x1)).cast<float>();
          
          /* Do the bilinear interpolation for each channel */
          for (int k = 0; k < 3; ++k) {
            color(k) = static_cast<int>(lerp<float>(lerp<float>(a(k),
                                                                b(k),
                                                                x_diff),
                                                    lerp<float>(c(k),
                                                                d(k),
                                                                x_diff),
                                                    y_diff));
          }
          
          /* Store the result */
          result(j, i) = MakeRGB(color);
        }
      }
      
      return result;
    }

  } /* namespace util */
} /* namespace pvrtex */


#endif /* defined(pvrtex_Util_h) */
