/*=========================================================================*/
/*                                                                         */
/* @file            Util.h                                                 */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/13                                             */
/* @version         0.2                                                    */
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

Eigen::MatrixXi ModulateImage(const Eigen::MatrixXi &dark,
                              const Eigen::MatrixXi &bright,
                              const Eigen::MatrixXf &mod);

Eigen::MatrixXi Downscale(Eigen::MatrixXi &orig);

Eigen::MatrixXi Upscale4x4(const Eigen::MatrixXi &orig);

Eigen::MatrixXi RGBtoYUV(const Eigen::MatrixXi &orig);
Eigen::MatrixXi YUVtoRGB(const Eigen::MatrixXi &orig);

float ComputeError(const Eigen::MatrixXi &orig,
                   const Eigen::MatrixXi &compressed);


} /* namespace util */
} /* namespace pvrtex */


#endif /* defined(pvrtex_Util_h) */
