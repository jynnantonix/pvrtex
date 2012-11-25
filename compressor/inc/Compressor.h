/*=========================================================================*/
/*                                                                         */
/* @file            Compressor.h                                           */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/05                                             */
/* @version         0.3                                                    */
/* @brief           Class declaration for pvr texture compressor           */
/*                                                                         */
/*=========================================================================*/
#ifndef __pvrtex__Compressor__
#define __pvrtex__Compressor__

#include <iostream>
#include <cfloat>
#include <FreeImage.h>
#include <Eigen/Dense>

#include "Util.h"
#include "Optimizer.h"

namespace pvrtex
{
  class Compressor
  {
  public:
    enum IMAGE_FORMAT { A8R8G8B8, R8G8B8, PVRTC4, PVRTC2 };
    
    Compressor(int w, int h, IMAGE_FORMAT f, unsigned int *d);
    ~Compressor();
    
    inline void set_width(int w) { width_ = w; }
    inline void set_height(int h) { height_ = h; }
    inline void set_format(IMAGE_FORMAT f) { format_ = f; }
    inline void set_data(unsigned int *d) { data_ = d; }
    
    inline int width() { return width_; }
    inline int height() { return height_; }
    inline IMAGE_FORMAT format() { return format_; }
    inline unsigned int* data() { return data_; }
    
    void Compress(unsigned int *out, IMAGE_FORMAT format);
    
    void WriteToFile(const char *filename);
  private:
    Compressor();
    
    Eigen::MatrixXf ComputeModulation(const Eigen::MatrixXi &orig,
                                      const Eigen::MatrixXi &dark,
                                      const Eigen::MatrixXi &bright);
    
    int width_;
    int height_;
    IMAGE_FORMAT format_;
    unsigned int *data_;
  };
} /* namespace pvrtex */
#endif /* defined(__pvrtex__Compressor__) */

