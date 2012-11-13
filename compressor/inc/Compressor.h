#ifndef __pvrtex__Compressor__
#define __pvrtex__Compressor__

/*=========================================================================*/
/*                                                                         */
/* @file            Compressor.h                                           */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/05                                             */
/* @version         0.2                                                    */
/* @brief           Class declaration for pvr texture compressor           */
/*                                                                         */
/*=========================================================================*/

#include <iostream>
#include <FreeImage.h>
#include <Eigen/Dense>

namespace pvrtex
{
  class Compressor
  {
  public:
    enum IMAGE_FORMAT { A8R8G8B8, R8G8B8, PVRTC4, PVRTC2 };
    
    Compressor(int w, int h, int sw, IMAGE_FORMAT f, BYTE *d);
    ~Compressor();
    
    inline void set_width(int w);
    inline void set_height(int h);
    inline void set_scan_width(int sw);
    inline void set_format(IMAGE_FORMAT f);
    inline void set_data(BYTE *d);
    
    inline int width();
    inline int height();
    inline int scan_width();
    inline IMAGE_FORMAT format();
    inline BYTE* data();
    
    void Compress(BYTE *out, IMAGE_FORMAT format = PVRTC4);
    
    void WriteToFile(const char *filename);
  private:
    Compressor();
    
    inline BYTE MakeAlpha(unsigned int p);
    inline BYTE MakeRed(unsigned int p);
    inline BYTE MakeGreen(unsigned int p);
    inline BYTE MakeBlue(unsigned int p);
    
    int width_;
    int height_;
    int scan_width_;
    IMAGE_FORMAT format_;
    BYTE *data_;
  };
} /* namespace PVRTEX */
#endif /* defined(__pvrtex__Compressor__) */

