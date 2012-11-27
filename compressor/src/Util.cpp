/*=========================================================================*/
/*                                                                         */
/* @file            Util.cpp                                               */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/27                                             */
/* @version         0.2                                                    */
/* @brief           Preprocessor definitions and utility functions for     */
/*                  handling pixels                                        */
/*                                                                         */
/*=========================================================================*/

#include "../inc/Util.h"

namespace pvrtex {
  namespace util {
    Eigen::MatrixXi ModulateImage(const Eigen::MatrixXi &dark,
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
    
    Eigen::MatrixXi Downscale(Eigen::MatrixXi &orig) {
      Eigen::MatrixXi result(orig.rows()>>2, orig.cols()>>2);
      for (int j = 0; j < result.rows(); ++j) {
        for (int i = 0; i < result.cols(); ++i) {
          result(j, i) = orig(j*4, i*4);
        }
      }
      
      return result;
    }
    
    Eigen::MatrixXi Upscale4x4(const Eigen::MatrixXi &orig) {
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

    float ComputeError(const Eigen::MatrixXi &orig,
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
  }
}