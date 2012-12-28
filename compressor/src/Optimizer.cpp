/*=========================================================================*/
/*                                                                         */
/* @file            Optimizer.cpp                                          */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/14                                             */
/* @version         0.3                                                    */
/* @brief           Optimizer for generating bright and dark images        */
/*                                                                         */
/*=========================================================================*/

#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <Eigen/SVD>

#include "../inc/Optimizer.h"

#define ROW(n)	1*n,	2*n,	3*n,	4*n,	3*n,	2*n,	1*n

namespace pvrtex {
// Optimization window information
const int Optimizer::window_width_ = 11;
const int Optimizer::window_height_ = 11;
const int Optimizer::matrix_rows_ = 121;
const int Optimizer::matrix_cols_ = 8;
const int Optimizer::offset_x_ = 4;
const int Optimizer::offset_y_ = 4;
  
// Weight matrices
const float Optimizer::kTopLeft[] =
{
  ROW(0.0625f), 0, 0, 0, 0,	// 1/16
  ROW(0.125f), 0, 0, 0, 0,	// 2/16
  ROW(0.1875f), 0, 0, 0, 0,	// 3/16
  ROW(0.25f), 0, 0, 0, 0,		// 4/16
  ROW(0.1875f), 0, 0, 0, 0,	// 3/16
  ROW(0.125f), 0, 0, 0, 0,	// 2/16
  ROW(0.0625f), 0, 0, 0, 0,	// 1/16
  ROW(0), 0, 0, 0, 0,
  ROW(0), 0, 0, 0, 0,
  ROW(0), 0, 0, 0, 0,
  ROW(0), 0, 0, 0, 0
};
const float Optimizer::kTopRight[] =
{
  0, 0, 0, 0, ROW(0.0625f),	// 1/16
  0, 0, 0, 0, ROW(0.125f),	// 2/16
  0, 0, 0, 0, ROW(0.1875f),	// 3/16
  0, 0, 0, 0, ROW(0.25f),		// 4/16
  0, 0, 0, 0, ROW(0.1875f),	// 3/16
  0, 0, 0, 0, ROW(0.125f),	// 2/16
  0, 0, 0, 0, ROW(0.0625f),	// 1/16
  ROW(0), 0, 0, 0, 0,
  ROW(0), 0, 0, 0, 0,
  ROW(0), 0, 0, 0, 0,
  ROW(0), 0, 0, 0, 0
};
const float Optimizer::kBottomLeft[] =
{
  ROW(0), 0, 0, 0, 0,
  ROW(0), 0, 0, 0, 0,
  ROW(0), 0, 0, 0, 0,
  ROW(0), 0, 0, 0, 0,
  ROW(0.0625f), 0, 0, 0, 0,	// 1/16
  ROW(0.125f), 0, 0, 0, 0,	// 2/16
  ROW(0.1875f), 0, 0, 0, 0,	// 3/16
  ROW(0.25f), 0, 0, 0, 0,		// 4/16
  ROW(0.1875f), 0, 0, 0, 0,	// 3/16
  ROW(0.125f), 0, 0, 0, 0,	// 2/16
  ROW(0.0625f), 0, 0, 0, 0,	// 1/16
};
const float Optimizer::kBottomRight[] =
{
  ROW(0), 0, 0, 0, 0,
  ROW(0), 0, 0, 0, 0,
  ROW(0), 0, 0, 0, 0,
  ROW(0), 0, 0, 0, 0,
  0, 0, 0, 0, ROW(0.0625f),	// 1/16
  0, 0, 0, 0, ROW(0.125f),	// 2/16
  0, 0, 0, 0, ROW(0.1875f),	// 3/16
  0, 0, 0, 0, ROW(0.25f),		// 4/16
  0, 0, 0, 0, ROW(0.1875f),	// 3/16
  0, 0, 0, 0, ROW(0.125f),	// 2/16
  0, 0, 0, 0, ROW(0.0625f)	// 1/16
};
  
Optimizer::Optimizer(const Eigen::MatrixXi &o, Eigen::MatrixXi &d,
                     Eigen::MatrixXi &b, SOLVER s, util::DATA_FORMAT f) :
  
  dark_(d),
  bright_(b),
  orig_(o),
  solv_(s),
  format_(f)
{
  red_ = Eigen::MatrixXi(orig_.rows(), orig_.cols());
  green_ = Eigen::MatrixXi(orig_.rows(), orig_.cols());
  blue_ = Eigen::MatrixXi(orig_.rows(), orig_.cols());
}
  
Optimizer::~Optimizer() {
}
  
void Optimizer::ComputeUpdateVector() {
  Eigen::MatrixXi comp = util::ModulateImage(util::Upscale4x4(dark_, format_),
                                             util::Upscale4x4(bright_, format_),
                                             mod_);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int y = 0; y < orig_.rows(); ++y) {
    for (int x = 0; x < orig_.cols(); ++x) {
      Eigen::Vector3i diff;
      diff = (util::MakeColorVector(orig_(y,x), util::PVR888) -
              util::MakeColorVector(comp(y,x), util::PVR888));
      //        red_(y,x) = util::MakeRed(orig_(y,x));
      //        green_(y,x) = util::MakeGreen(orig_(y,x));
      //        blue_(y,x) = util::MakeBlue(orig_(y,x));
      red_(y,x) = diff(0);
      green_(y,x) = diff(1);
      blue_(y,x) = diff(2);
    }
  }
}
  
void Optimizer::OptimizeWindow(int j, int i) {
  Eigen::MatrixXf a(matrix_rows_, matrix_cols_);
  Eigen::MatrixXf w(matrix_rows_, matrix_cols_ / 2);
  Eigen::VectorXf red(matrix_rows_);
  Eigen::VectorXf green(matrix_rows_);
  Eigen::VectorXf blue(matrix_rows_);
  int idx, pixel_x, pixel_y;
  float m, distance;
  float r, g, b;
    
  /* Construct the optimization window */
  for (int y = 0; y < window_height_; ++y) {
    for (int x = 0; x < window_width_; ++x) {
      /* Get the position of the pixel we want to fetch */
      idx = y*window_width_ + x;
      pixel_x = util::Clamp(offset_x_*i-1 + x, 0, mod_.cols()-1);
      pixel_y = util::Clamp(offset_y_*j-1 + y, 0, mod_.rows()-1);
        
      /* Fetch the modulation value and the original color*/
      m = mod_(pixel_y, pixel_x);
      r = static_cast<float>(red_(pixel_y, pixel_x));
      g = static_cast<float>(green_(pixel_y, pixel_x));
      b = static_cast<float>(blue_(pixel_y, pixel_x));
        
      /* Fetch the distance weights and construct the matrix */
      distance = kTopLeft[idx];
      a(idx, 0) = distance * (1.0f - m);
      a(idx, 1) = distance * m;
      w(idx, 0) = distance;
      red(idx) = distance * r;
      green(idx) = distance * g;
      blue(idx) = distance * b;
        
      /* Top right pxel */
      distance = kTopRight[idx];
      a(idx, 2) = distance * (1.0f - m);
      a(idx, 3) = distance * m;
      w(idx, 1) = distance;
      red(idx) += distance * r;
      green(idx) += distance * g;
      blue(idx) += distance * b;
        
      /* Bottom left pixel */
      distance = kBottomLeft[idx];
      a(idx, 4) = distance * (1.0f - m);
      a(idx, 5) = distance * m;
      w(idx, 2) = distance;
      red(idx) += distance * r;
      green(idx) += distance * g;
      blue(idx) += distance * b;
        
      /* Bottom right pixel */
      distance = kBottomRight[idx];
      a(idx, 6) = distance * (1.0f - m);
      a(idx, 7) = distance * m;
      w(idx, 3) = distance;
      red(idx) += distance * r;
      green(idx) += distance * g;
      blue(idx) += distance * b;
        
    }
  }
    
  /* Solve for the best colors */
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(a, Eigen::ComputeThinU |
                                        Eigen::ComputeThinV);
  Eigen::VectorXi optimal_red = svd.solve(red).cast<int>();
  Eigen::VectorXi optimal_green;
  Eigen::VectorXi optimal_blue;
  if (format_ == util::PVR444) {
    Eigen::JacobiSVD<Eigen::MatrixXf> svd_w(w, Eigen::ComputeThinU |
                                            Eigen::ComputeThinV);
    Eigen::Vector3i update;
    optimal_green = svd_w.solve(green).cast<int>();
    optimal_blue = svd_w.solve(blue).cast<int>();
    
    /* Update the dark and bright images */
    for (int x = 0; x < 2; ++x) {
      for (int y = 0; y < 2; ++y) {
        idx = 4*y + 2*x;
        update = Eigen::Vector3i(util::Clamp(optimal_red(idx), -32, 32),
                                 util::Clamp(optimal_green(idx/2), -32, 32),
                                 util::Clamp(optimal_blue(idx/2), -32, 32));
        dark_(j+y, i+x) = util::MakeRGB(
                              util::MakeColorVector(dark_(j+y, i+x), format_) +
                              update, format_);
        update(0) = util::Clamp(optimal_red(idx+1), -32, 32);
        bright_(j+y, i+x) = util::MakeRGB(
                                util::MakeColorVector(bright_(j+y, i+x), format_) +
                                    update, format_);
        
      }
    }
  } else {
    optimal_green = svd.solve(green).cast<int>();
    optimal_blue = svd.solve(blue).cast<int>();
    
    /* Update the dark and bright images */
    for (int x = 0; x < 2; ++x) {
      for (int y = 0; y < 2; ++y) {
        idx = 4*y + 2*x;
        dark_(j+y, i+x) = util::MakeRGB(
                              util::MakeColorVector(dark_(j+y, i+x), format_) +
                              Eigen::Vector3i(
                                  util::Clamp(optimal_red(idx), -32, 32),
                                  util::Clamp(optimal_green(idx), -32, 32),
                                  util::Clamp(optimal_blue(idx), -32, 32)),
                                        format_);
        bright_(j+y, i+x) = util::MakeRGB(
                                util::MakeColorVector(bright_(j+y, i+x), format_) +
                                Eigen::Vector3i(
                                    util::Clamp(optimal_red(idx+1), -32, 32),
                                    util::Clamp(optimal_green(idx+1), -32, 32),
                                    util::Clamp(optimal_blue(idx+1), -32, 32)),
                                          format_);
        //      dark_(j+y, i+x) = util::Make565RGB(Eigen::Vector3i(optimal_red(idx),
        //                                                         optimal_green(idx),
        //                                                         optimal_blue(idx)));
        //      bright_(j+y, i+x) = util::Make565RGB(Eigen::Vector3i(optimal_red(idx+1),
        //                                                           optimal_green(idx+1),
        //                                                           optimal_blue(idx+1)));
        
      }
    }

  }
    
}
  
void Optimizer::Optimize(const Eigen::MatrixXf &m) {
  mod_ = m;
  ComputeUpdateVector();
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int j = 0; j < dark_.rows(); j+=2) {
    for (int i = 0; i < dark_.cols(); i+=2) {
      OptimizeWindow(j, i);
    }
  }
}

void Optimizer::WriteToFile(const char *filename) {
    
}

} /* namespace pvrtex */

