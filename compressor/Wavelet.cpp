/*=========================================================================*/
/*                                                                         */
/* @file            Wavelet.cpp                                            */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/13                                             */
/* @version         0.1                                                    */
/* @brief           Class implementation for wavelet filters               */
/*                                                                         */
/*=========================================================================*/

#include "Wavelet.h"

namespace pvrtex {
  // simple wavelet filter
  static const float kBasicFilter[] = { 0.5f, 0.5f };
  
  Wavelet::Wavelet(FILTER f)
  :
  filter_type_(f)
  {
    switch (filter_type_) {
      case BASIC: {
        filter_ = kBasicFilter;
        break;
      }
      default:
        break;
    }
  }
  
  Wavelet::~Wavelet()
  {
  }
  
  void Wavelet::Init()
  {
  }
}