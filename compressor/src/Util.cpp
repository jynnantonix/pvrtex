/*=========================================================================*/
/*                                                                         */
/* @file            Util.cpp                                               */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/27                                             */
/* @version         0.3                                                    */
/* @brief           Utility functions for handling pixels                  */
/*                                                                         */
/*=========================================================================*/

#include <FreeImage.h>

#include "../inc/Util.h"

#define ONE_FOURTH            0.25f
#define THREE_EIGHTHS         0.375f
#define FIVE_EIGHTHS          0.625f

#define PVR_565_RED_MASK      0xF800
#define PVR_565_GREEN_MASK    0x07E0
#define PVR_565_BLUE_MASK     0x001F
#define PVR_565_RED_SHIFT     11
#define PVR_565_GREEN_SHIFT   5
#define PVR_565_BLUE_SHIFT    0
#define PVR_565_RED_8SHIFT    3
#define PVR_565_GREEN_8SHIFT  2
#define PVR_565_BLUE_8SHIFT   3

#define PVR_655_RED_MASK      0xFC00
#define PVR_655_GREEN_MASK    0x03E0
#define PVR_655_BLUE_MASK     0x001F
#define PVR_655_RED_SHIFT     10
#define PVR_655_GREEN_SHIFT   5
#define PVR_655_BLUE_SHIFT    0
#define PVR_655_RED_8SHIFT    2
#define PVR_655_GREEN_8SHIFT  3
#define PVR_655_BLUE_8SHIFT   3

#define PVR_844_RED_MASK      0xFF00
#define PVR_844_GREEN_MASK    0x00F0
#define PVR_844_BLUE_MASK     0x000F
#define PVR_844_RED_SHIFT     8
#define PVR_844_GREEN_SHIFT   4
#define PVR_844_BLUE_SHIFT    0
#define PVR_844_RED_8SHIFT    0
#define PVR_844_GREEN_8SHIFT  4
#define PVR_844_BLUE_8SHIFT   4

#define PVR_444_RED_MASK      0x0F00
#define PVR_444_GREEN_MASK    0x00F0
#define PVR_444_BLUE_MASK     0x000F
#define PVR_444_RED_SHIFT     8
#define PVR_444_GREEN_SHIFT   4
#define PVR_444_BLUE_SHIFT    0
#define PVR_444_RED_8SHIFT    4
#define PVR_444_GREEN_8SHIFT  4
#define PVR_444_BLUE_8SHIFT   4

namespace pvrtex {
namespace util {
int MakeRed(unsigned int p, DATA_FORMAT f) {
  int result;
  switch (f) {
    case PVR888: {
      result = ((p & FI_RGBA_RED_MASK)>>FI_RGBA_RED_SHIFT);
      break;
    }
    case PVR565: {
      result = (((p & PVR_565_RED_MASK)>>PVR_565_RED_SHIFT)
                <<PVR_565_RED_8SHIFT);
      break;
    }
    case PVR655: {
      result = (((p & PVR_655_RED_MASK)>>PVR_655_RED_SHIFT)
                <<PVR_655_RED_8SHIFT);
      break;
    }
    case PVR844: {
      result = (((p & PVR_844_RED_MASK)>>PVR_844_RED_SHIFT)
                <<PVR_844_RED_8SHIFT);
      break;
    }
    case PVR444: {
      result = (((p & PVR_444_RED_MASK)>>PVR_444_RED_SHIFT)
                <<PVR_444_RED_8SHIFT);
      break;
    }
    default: {
      assert(false);
    }
  }
  return result;
}
  
int MakeGreen(unsigned int p, DATA_FORMAT f) {
  int result;
  switch (f) {
    case PVR888: {
      result = ((p & FI_RGBA_GREEN_MASK)>>FI_RGBA_GREEN_SHIFT);
      break;
    }
    case PVR565: {
      result = (((p & PVR_565_GREEN_MASK)>>PVR_565_GREEN_SHIFT)
                <<PVR_565_GREEN_8SHIFT);
      break;
    }
    case PVR655: {
      result = (((p & PVR_655_GREEN_MASK)>>PVR_655_GREEN_SHIFT)
                <<PVR_655_GREEN_8SHIFT);
      break;
    }
    case PVR844: {
      result = (((p & PVR_844_GREEN_MASK)>>PVR_844_GREEN_SHIFT)
                <<PVR_844_GREEN_8SHIFT);
      break;
    }
    case PVR444: {
      result = (((p & PVR_444_GREEN_MASK)>>PVR_444_GREEN_SHIFT)
                <<PVR_444_GREEN_8SHIFT);
      break;
    }
    default: {
      assert(false);
    }
  }
  return result;
}
  
int MakeBlue(unsigned int p, DATA_FORMAT f) {
  int result;
  switch (f) {
    case PVR888: {
      result = ((p & FI_RGBA_BLUE_MASK)>>FI_RGBA_BLUE_SHIFT);
      break;
    }
    case PVR565: {
      result = (((p & PVR_565_BLUE_MASK)>>PVR_565_BLUE_SHIFT)
                <<PVR_565_BLUE_8SHIFT);
      break;
    }
    case PVR655: {
      result = (((p & PVR_655_BLUE_MASK)>>PVR_655_BLUE_SHIFT)
                <<PVR_655_BLUE_8SHIFT);
      break;
    }
    case PVR844: {
      result = (((p & PVR_844_BLUE_MASK)>>PVR_844_BLUE_SHIFT)
                <<PVR_844_BLUE_8SHIFT);
      break;
    }
    case PVR444: {
      result = (((p & PVR_444_BLUE_MASK)>>PVR_444_BLUE_SHIFT)
                <<PVR_444_BLUE_8SHIFT);
      break;
    }
    default: {
      assert(false);
    }
  }
  return result;
}
  
unsigned int MakeRGB(const Eigen::Vector3i &p, DATA_FORMAT f) {
  unsigned int result;
  
  /* Masks off the appropriate number of lower order bits and then moves */
  /* the remaining higher order bits to the correct position */
  switch (f) {
    case PVR888: {
      result  = (((0x000000FF)<<FI_RGBA_ALPHA_SHIFT) |
                 (Clamp(p(0),0,255)<<FI_RGBA_RED_SHIFT) |
                 (Clamp(p(1),0,255)<<FI_RGBA_GREEN_SHIFT) |
                 (Clamp(p(2),0,255)<<FI_RGBA_BLUE_SHIFT));
      break;
    }
    case PVR565: {
      result = (((Clamp(p(0),0,255)>>PVR_565_RED_8SHIFT) << PVR_565_RED_SHIFT) |
                ((Clamp(p(1),0,255)>>PVR_565_GREEN_8SHIFT) << PVR_565_GREEN_SHIFT) |
                ((Clamp(p(2),0,255)>>PVR_565_BLUE_8SHIFT) << PVR_565_BLUE_SHIFT));
      break;
    }
    case PVR655: {
      result = (((Clamp(p(0),0,255)>>PVR_655_RED_8SHIFT) << PVR_655_RED_SHIFT) |
                ((Clamp(p(1),0,255)>>PVR_655_GREEN_8SHIFT) << PVR_655_GREEN_SHIFT) |
                ((Clamp(p(2),0,255)>>PVR_655_BLUE_8SHIFT) << PVR_655_BLUE_SHIFT));
      break;
    }
    case PVR844: {
      result = (((Clamp(p(0),0,255)>>PVR_844_RED_8SHIFT) << PVR_844_RED_SHIFT) |
                ((Clamp(p(1),0,255)>>PVR_844_GREEN_8SHIFT) << PVR_844_GREEN_SHIFT) |
                ((Clamp(p(2),0,255)>>PVR_844_BLUE_8SHIFT) << PVR_844_BLUE_SHIFT));
      break;
    }
    case PVR444: {
      result = (((Clamp(p(0),0,255)>>PVR_444_RED_8SHIFT) << PVR_444_RED_SHIFT) |
                ((Clamp(p(1),0,255)>>PVR_444_GREEN_8SHIFT) << PVR_444_GREEN_SHIFT) |
                ((Clamp(p(2),0,255)>>PVR_444_BLUE_8SHIFT) << PVR_444_BLUE_SHIFT));
      break;
    }
    default:
      assert(false);
  }
  
  return result;
}
  
DATA_FORMAT ImageToData(Compressor::IMAGE_FORMAT f) {
  DATA_FORMAT result;
  switch (f) {
    case Compressor::PVRTC_2BPP:
    case Compressor::PVRTC_4BPP: {
      result = PVR565;
      break;
    }
    case Compressor::YUV_4BPP: {
      result = PVR655;
      break;
    }
    case Compressor::YUV_OPT_4BPP: {
      result = PVR844;
      break;
    }
    case Compressor::YUV_EXT_4BPP:
    case Compressor::YUV_2BPP: {
      result = PVR444;
      break;
    }
      
    default: {
      assert(false);
    }
  }
  return result;
}
  
Eigen::MatrixXi ModulateImage(const Eigen::MatrixXi &dark,
                              const Eigen::MatrixXi &bright,
                              const Eigen::MatrixXf &mod) {
  Eigen::MatrixXi result(mod.rows(), mod.cols());
  Eigen::Vector3f d, b;
  Eigen::Vector3i r;
  for (int y = 0; y < mod.rows(); ++y) {
    for (int x = 0; x < mod.cols(); ++x) {
      d = MakeColorVector(dark(y, x), PVR888).cast<float>();
      b = MakeColorVector(bright(y, x), PVR888).cast<float>();
      r = lerp<Eigen::Vector3f>(d,b, mod(y, x)).cast<int>();
      result(y, x) = MakeRGB(r, PVR888);
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
    
Eigen::MatrixXi Upscale4x4(const Eigen::MatrixXi &orig, DATA_FORMAT f) {
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
      a = MakeColorVector(orig(y, x), f).cast<float>();
      b = MakeColorVector(orig(y, x1), f).cast<float>();
      c = MakeColorVector(orig(y1, x), f).cast<float>();
      d = MakeColorVector(orig(y1, x1), f).cast<float>();
          
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
      result(j, i) = MakeRGB(color, PVR888);
    }
  }
      
  return result;
}

Eigen::MatrixXi RGBtoYUV(const Eigen::MatrixXi &orig) {
  Eigen::MatrixXi result(orig.rows(), orig.cols());
      
  // Transformation table
  Eigen::Matrix3f table;
  table << 0.257, 0.504, 0.098,
      -0.148, -0.291, 0.439,
      0.439, -0.368, -0.071;
  Eigen::Vector3f offset(16, 128, 128);
      
  // Do the tranformation
  Eigen::Vector3f color;
  for (int y = 0; y < orig.rows(); ++y) {
    for (int x = 0; x < orig.cols(); ++x) {
      color = table * MakeColorVector(orig(y, x), PVR888).cast<float>();
          
      result(y, x) = MakeRGB((color + offset).cast<int>(), PVR888);
    }
  }
      
  return result;
}
    
Eigen::MatrixXi YUVtoRGB(const Eigen::MatrixXi &orig) {
  Eigen::MatrixXi result(orig.rows(), orig.cols());
      
  // Transformation table
  Eigen::Matrix3f table;
  table <<1.164, 0.0, 1.596,
      1.164, -0.391, -0.813,
      1.164, 2.018, 0.0;
  Eigen::Vector3f offset(16, 128, 128);
      
  // Do the tranformation
  Eigen::Vector3f color;
  for (int y = 0; y < orig.rows(); ++y) {
    for (int x = 0; x < orig.cols(); ++x) {
      // Reverse the storage offsets first
      color = MakeColorVector(orig(y, x), PVR888).cast<float>();
      color -= offset;
          
      // Now convert back to RGB
      result(y, x) = MakeRGB((table * color).cast<int>(), PVR888);
    }
  }
      
  return result;
}
    
float ComputeError(const Eigen::MatrixXi &orig,
                   const Eigen::MatrixXi &compressed) {
  float result = 0.0f;
  for (int y = 0; y < orig.rows(); ++y) {
    for (int x = 0; x < orig.cols(); ++x) {
      result += (MakeColorVector(orig(y, x), PVR888) -
                 MakeColorVector(compressed(y, x), PVR888)).squaredNorm();
    }
  }
      
  return sqrtf(result / (orig.rows()*orig.cols()));
}
}
}
