#ifndef __pvrtex__Wavelet__
#define __pvrtex__Wavelet__

/*=========================================================================*/
/*                                                                         */
/* @file            Wavelet.h                                              */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/13                                             */
/* @version         0.1                                                    */
/* @brief           Class declaration for wavelet filters                  */
/*                                                                         */
/*=========================================================================*/

#include <iostream>
#include <Eigen/Dense>

namespace pvrtex
{
  class Compressor;
  
  class Wavelet
  {
  public:
    enum FILTER { BIOR, DAUBECHIES, BASIC };
    
    Wavelet(FILTER f = BASIC);
    ~Wavelet();
    
    void Init();
    
    void Downscale(Eigen::MatrixXi orig, Eigen::MatrixXi result);
    
    void Upscale(Eigen::MatrixXi orig, Eigen::MatrixXi result);
    
  private:
    
    FILTER filter_type_;
    const float *filter_;
  };
}
#endif /* defined(__pvrtex__Wavelet__) */
