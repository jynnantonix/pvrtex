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
namespace pvrtex
{
  class Wavelet
  {
  public:
    enum FILTER { BIOR, DAUBECHIES, BASIC };
    
    Wavelet(FILTER f = BASIC);
    ~Wavelet();
    
    void Init();
  private:
    FILTER filter_type_;
    const float *filter_;
  };
}
#endif /* defined(__pvrtex__Wavelet__) */
