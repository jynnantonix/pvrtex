/*=========================================================================*/
/*                                                                         */
/* @file            Optimizer.cpp                                          */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/14                                             */
/* @version         0.1                                                    */
/* @brief           Optimizer for generating bright and dark images        */
/*                                                                         */
/*=========================================================================*/

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
  
  Optimizer::Optimizer(const Eigen::MatrixXi &o, const Eigen::MatrixXf &m,
                       Eigen::MatrixXi &d, Eigen::MatrixXi &b) :
  dark_(d),
  bright_(b),
  orig_(o),
  mod_(m),
  solv_(SVD)
  {
    red_ = Eigen::MatrixXi(orig_.rows(), orig_.cols());
    green_ = Eigen::MatrixXi(orig_.rows(), orig_.cols());
    blue_ = Eigen::MatrixXi(orig_.rows(), orig_.cols());
  }
  
  Optimizer::Optimizer(const Eigen::MatrixXi &o, const Eigen::MatrixXf &m,
                       Eigen::MatrixXi &d, Eigen::MatrixXi &b, SOLVER s):
  dark_(d),
  bright_(b),
  orig_(o),
  mod_(m),
  solv_(s)
  {
    red_ = Eigen::MatrixXi(orig_.rows(), orig_.cols());
    green_ = Eigen::MatrixXi(orig_.rows(), orig_.cols());
    blue_ = Eigen::MatrixXi(orig_.rows(), orig_.cols());
  }
  
  Optimizer::~Optimizer() {
  }
  
  void Optimizer::ComputeUpdateVector() {
    Eigen::Vector4i diff;
    Eigen::MatrixXi comp = util::ModulateImage(util::Upscale4x4(dark_),
                                               util::Upscale4x4(bright_),
                                               mod_);
    for (int y = 0; y < orig_.rows(); ++y) {
      for (int x = 0; x < orig_.cols(); ++x) {
        diff = (util::MakeColorVector(orig_(y,x)) -
                util::MakeColorVector(comp(y,x)));
//        red(y,x) = util::MakeRed(orig(y,x));
//        green(y,x) = util::MakeGreen(orig(y,x));
//        blue(y,x) = util::MakeBlue(orig(y,x));
        red_(y,x) = diff(1);
        green_(y,x) = diff(2);
        blue_(y,x) = diff(3);
      }
    }
  }
  
  void Optimizer::OptimizeWindow(int j, int i) {
    Eigen::MatrixXf a(matrix_rows_, matrix_cols_);
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
        red(idx) = distance * r;
        green(idx) = distance * g;
        blue(idx) = distance * b;
        
        /* Top right pxel */
        distance = kTopRight[idx];
        a(idx, 2) = distance * (1.0f - m);
        a(idx, 3) = distance * m;
        red(idx) += distance * r;
        green(idx) += distance * g;
        blue(idx) += distance * b;
        
        /* Bottom left pixel */
        distance = kBottomLeft[idx];
        a(idx, 4) = distance * (1.0f - m);
        a(idx, 5) = distance * m;
        red(idx) += distance * r;
        green(idx) += distance * g;
        blue(idx) += distance * b;
        
        /* Bottom right pixel */
        distance = kBottomRight[idx];
        a(idx, 6) = distance * (1.0f - m);
        a(idx, 7) = distance * m;
        red(idx) += distance * r;
        green(idx) += distance * g;
        blue(idx) += distance * b;
        
      }
    }
    
    /* Solve for the best colors */
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(a, Eigen::ComputeThinU |
                                          Eigen::ComputeThinV);
    Eigen::VectorXi optimal_red = svd.solve(red).cast<int>();
    Eigen::VectorXi optimal_green = svd.solve(green).cast<int>();
    Eigen::VectorXi optimal_blue = svd.solve(blue).cast<int>();
    
    /* Update the dark and bright images */
    dark_(j, i) = util::MakeRGBA(util::MakeColorVector(dark_(j, i)) +
                                 Eigen::Vector4i(0xFFFFFFFF,
                                                 optimal_red(0),
                                                 optimal_green(0),
                                                 optimal_blue(0)));
    bright_(j, i) = util::MakeRGBA(util::MakeColorVector(bright_(j, i)) +
                                   Eigen::Vector4i(0xFFFFFFFF,
                                                   optimal_red(1),
                                                   optimal_green(1),
                                                   optimal_blue(1)));
    dark_(j, i+1) = util::MakeRGBA(util::MakeColorVector(dark_(j, i+1)) +
                                   Eigen::Vector4i(0xFFFFFFFF,
                                                   optimal_red(2),
                                                   optimal_green(2),
                                                   optimal_blue(2)));
    bright_(j, i+1) = util::MakeRGBA(util::MakeColorVector(bright_(j, i+1)) +
                                     Eigen::Vector4i(0xFFFFFFFF,
                                                     optimal_red(3),
                                                     optimal_green(3),
                                                     optimal_blue(3)));
    dark_(j+1, i) = util::MakeRGBA(util::MakeColorVector(dark_(j+1, i)) +
                                   Eigen::Vector4i(0xFFFFFFFF,
                                                   optimal_red(4),
                                                   optimal_green(4),
                                                   optimal_blue(4)));
    bright_(j+1, i) = util::MakeRGBA(util::MakeColorVector(bright_(j+1, i)) +
                                     Eigen::Vector4i(0xFFFFFFFF,
                                                     optimal_red(5),
                                                     optimal_green(5),
                                                     optimal_blue(5)));
    dark_(j+1, i+1) = util::MakeRGBA(util::MakeColorVector(dark_(j+1, i+1)) +
                                     Eigen::Vector4i(0xFFFFFFFF,
                                                     optimal_red(6),
                                                     optimal_green(6),
                                                     optimal_blue(6)));
    bright_(j+1, i+1) = util::MakeRGBA(util::MakeColorVector(bright_(j+1, i+1)) +
                                       Eigen::Vector4i(0xFFFFFFFF,
                                                       optimal_red(7),
                                                       optimal_green(7),
                                                       optimal_blue(7)));
  }
  
  void Optimizer::Optimize() {
    ComputeUpdateVector();
    for (int j = 0; j < dark_.rows(); j+=2) {
      for (int i = 0; i < dark_.cols(); i+=2) {
        OptimizeWindow(j, i);
      }
    }
  }
}