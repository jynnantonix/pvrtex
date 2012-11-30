/*=========================================================================*/
/*                                                                         */
/* @file            Util.h                                                 */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/13                                             */
/* @version         0.3                                                    */
/* @brief           Utility functions for handling pixels                  */
/*                                                                         */
/*=========================================================================*/

#ifndef pvrtex_Util_h
#define pvrtex_Util_h

//#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

#include "Compressor.h"

namespace pvrtex {
namespace util {
enum DATA_FORMAT { PVR888, PVR565, PVR655, PVR844, PVR444 };
  
int MakeRed(unsigned int p, DATA_FORMAT f);
int MakeGreen(unsigned int p, DATA_FORMAT f);
int MakeBlue(unsigned int p, DATA_FORMAT f);
unsigned int MakeRGB(const Eigen::Vector3i &p, DATA_FORMAT f);

DATA_FORMAT ImageToData(Compressor::IMAGE_FORMAT f);
Eigen::MatrixXi ModulateImage(const Eigen::MatrixXi &dark,
                              const Eigen::MatrixXi &bright,
                              const Eigen::MatrixXf &mod);

Eigen::MatrixXi Downscale(Eigen::MatrixXi &orig);
Eigen::MatrixXi Upscale4x4(const Eigen::MatrixXi &orig, DATA_FORMAT f);

Eigen::MatrixXi RGBtoYUV(const Eigen::MatrixXi &orig);
Eigen::MatrixXi YUVtoRGB(const Eigen::MatrixXi &orig);

float ComputeError(const Eigen::MatrixXi &orig,
                   const Eigen::MatrixXi &compressed);

inline int Clamp(int x, int a, int b) {
  return std::min<int>(std::max<int>(x, a), b);
}

template<typename T>
inline T lerp(const T &a, const T &b, const float delta) {
  return ((1.0f-delta) * a) + (delta*b);
}

  
inline Eigen::Vector3i MakeColorVector(unsigned int p, DATA_FORMAT f) {
  return Eigen::Vector3i(MakeRed(p, f),
                         MakeGreen(p, f),
                         MakeBlue(p, f));
}

} /* namespace util */
} /* namespace pvrtex */


#endif /* defined(pvrtex_Util_h) */
