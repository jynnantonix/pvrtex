/*=========================================================================*/
/*                                                                         */
/* @file            Wavelet.h                                              */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/13                                             */
/* @version         0.1                                                    */
/* @brief           Class declaration for wavelet filters                  */
/*                                                                         */
/*=========================================================================*/
#ifndef __pvrtex__Wavelet__
#define __pvrtex__Wavelet__

#include <Eigen/Dense>

#include "Util.h"
namespace pvrtex
{  
  class Wavelet
  {
  public:
    enum FILTER { BIOR, DAUBECHIES, BASIC };
    
    Wavelet();
    Wavelet(FILTER f);
    ~Wavelet();
    
    void Init();
    
    Eigen::MatrixXi Downscale(Eigen::MatrixXi orig);
    
    Eigen::MatrixXi Upscale(Eigen::MatrixXi orig);
    
  private:
    FILTER filter_type_;
    
    static const float kBasicFilter[];
  };
} /* namespace pvrtex */
#endif /* defined(__pvrtex__Wavelet__) */
