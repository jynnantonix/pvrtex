/*=========================================================================*/
/*                                                                         */
/* @file            Compressor.h                                           */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/05                                             */
/* @version         0.4                                                    */
/* @brief           Class declaration for pvr texture compressor           */
/*                                                                         */
/*=========================================================================*/
#ifndef __pvrtex__Compressor__
#define __pvrtex__Compressor__

#include <Eigen/Dense>

namespace pvrtex {
class Compressor {
public:
  enum IMAGE_FORMAT { PVRTC_4BPP,   /* Standard 4bpp PVR texture format.     */
                      PVRTC_2BPP,   /* Standard 2bpp PVR texture format.     */
                      YUV_4BPP,     /* 4bpp compression in YUV color space.  */
                      YUV_OPT_4BPP, /* Same as YUV_4BPP but bit allocation   */
                                    /* for each channel is optimized         */
                      YUV_EXT_4BPP, /* Sacrifice color choices to get more   */
                                    /* choices of modulation values.         */
                      YUV_2BPP,     /* Sacrifice color choices and encode    */
                                    /* modulation values alternately in a    */
                                    /* checkerboard pattern to save space.   */
                      PVR_UNDEFINED /* Something went wrong, this shold not  */
                                    /* be chosen.                            */
  };
  
  Compressor();
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
  
  void Compress(unsigned int *out);
  
private:
  /* Don't allow copy contructor or assignment */
  Compressor(const Compressor &c);
  void operator=(const Compressor &c);
  
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

