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
  inline float Clamp(unsigned int x, unsigned int a, unsigned int b) {
    return static_cast<int>(fminf(fmaxf(static_cast<float>(x),
                                        static_cast<float>(a)),
                                  static_cast<float>(b)));
  }
  
  inline float lerp(const float a, const float b, const float delta) {
    return ((1-delta) * a) + (delta*b);
  }
  
  inline BYTE MakeAlpha(unsigned int p) {
    return ((p & FI_RGBA_ALPHA_MASK)>>FI_RGBA_ALPHA_SHIFT);
  }
  
  inline BYTE MakeRed(unsigned int p) {
    return ((p & FI_RGBA_RED_MASK)>>FI_RGBA_RED_SHIFT);
  }
  
  inline BYTE MakeGreen(unsigned int p) {
    return ((p & FI_RGBA_GREEN_MASK)>>FI_RGBA_GREEN_SHIFT);
  }
  
  inline BYTE MakeBlue(unsigned int p) {
    return ((p & FI_RGBA_BLUE_MASK)>>FI_RGBA_BLUE_SHIFT);
  }
  
  inline Eigen::Vector4i MakeColorVector(unsigned int p) {
    return Eigen::Vector4i(static_cast<int>(MakeAlpha(p)),
                           static_cast<int>(MakeRed(p)),
                           static_cast<int>(MakeGreen(p)),
                           static_cast<int>(MakeBlue(p)));
  }
  
  inline unsigned int MakeRGBA(Eigen::Vector4i p) {
    return (((p(0) & 0x000000FF)<<FI_RGBA_ALPHA_SHIFT) |
            ((p(1) & 0x000000FF)<<FI_RGBA_RED_SHIFT) |
            ((p(2) & 0x000000FF)<<FI_RGBA_GREEN_SHIFT) |
            ((p(3) & 0x000000FF)<<FI_RGBA_BLUE_SHIFT));
  }
}


#endif /* defined(pvrtex_Util_h) */
