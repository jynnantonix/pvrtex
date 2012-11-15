//
//  LeastSquaresSolver.cpp
//  pvrtex
//
//  Created by Chirantan Ekbote on 11/14/12.
//
//

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
  
  Optimizer::Optimizer(Eigen::MatrixXi o, Eigen::MatrixXf m) :
  orig_(o),
  mod_(m),
  solv_(SVD)
  {
    dark_ = Eigen::MatrixXi(o.rows()>>2, o.cols()>>2);
    bright_ = Eigen::MatrixXi(o.rows()>>2, o.cols()>>2);
  }
  
  Optimizer::Optimizer(Eigen::MatrixXi o, Eigen::MatrixXf m, SOLVER s) :
  orig_(o),
  mod_(m),
  solv_(s)
  {
    dark_(o.rows()>>2, o.cols()>>2);
    bright_(o.rows()>>2, o.cols()>>2);
  }
  
  Optimizer::~Optimizer() {
  }
  
  Eigen::VectorXi Optimizer::OptimizeWindow(int j, int i) {
    Eigen::MatrixXf a(matrix_rows_, matrix_cols_);
    Eigen::VectorXf red(matrix_rows_);
    Eigen::VectorXf green(matrix_rows_);
    Eigen::VectorXf blue(matrix_rows_);
    Eigen::VectorXi result(matrix_cols_);
    int idx, pixel_x, pixel_y;
    float m, distance;
    int o;
    
    /* Construct the optimization window */
    for (int y = 0; y < window_height_; ++y) {
      for (int x = 0; x < window_width_; ++x) {
        /* Get the position of the pixel we want to fetch */
        idx = y*window_width_ + x;
        pixel_x = util::Clamp(offset_x_*i-1 + x, 0, orig_.cols()-1);
        pixel_y = util::Clamp(offset_y_*j-1 + y, 0, orig_.rows()-1);
        
        /* Fetch the modulation value and the original color*/
        m = mod_(pixel_y, pixel_x);
        o = orig_(pixel_y, pixel_x);
        
        /* Fetch the distance weights and construct the matrix */
        distance = kTopLeft[idx];
        a(idx, 0) = distance * (1.0f - m);
        a(idx, 1) = distance * m;
        red(idx) = distance * util::MakeRed(o);
        green(idx) = distance * util::MakeGreen(o);
        blue(idx) = distance * util::MakeBlue(o);
        
        /* Top right pxel */
        distance = kTopRight[idx];
        a(idx, 2) = distance * (1.0f - m);
        a(idx, 3) = distance * m;
        red(idx) += distance * util::MakeRed(o);
        green(idx) += distance * util::MakeGreen(o);
        blue(idx) += distance * util::MakeBlue(o);
        
        /* Bottom left pixel */
        distance = kBottomLeft[idx];
        a(idx, 4) = distance * (1.0f - m);
        a(idx, 5) = distance * m;
        red(idx) += distance * util::MakeRed(o);
        green(idx) += distance * util::MakeGreen(o);
        blue(idx) += distance * util::MakeBlue(o);
        
        /* Bottom right pixel */
        distance = kBottomRight[idx];
        a(idx, 6) = distance * (1.0f - m);
        a(idx, 7) = distance * m;
        red(idx) += distance * util::MakeRed(o);
        green(idx) += distance * util::MakeGreen(o);
        blue(idx) += distance * util::MakeBlue(o);
        
      }
    }
    
    /* Solve for the best colors */
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(a, Eigen::ComputeThinU |
                                          Eigen::ComputeThinV);
    Eigen::VectorXf optimal_red = svd.solve(red);
    Eigen::VectorXf optimal_green = svd.solve(green);
    Eigen::VectorXf optimal_blue = svd.solve(blue);
    
    /* Construct the result vector */
    for (int k = 0; k < result.size(); ++k) {
      result(k) = util::MakeRGBA(static_cast<int>(0xFFFFFFFF),
                                 static_cast<int>(optimal_red(k)),
                                 static_cast<int>(optimal_green(k)),
                                 static_cast<int>(optimal_blue(k)));
    }
    
    return result;
  }
  
  void Optimizer::Optimize() {
    for (int j = 0; j < dark_.rows(); j+=2) {
      for (int i = 0; i < dark_.cols(); i+=2) {
        Eigen::VectorXi optimal = OptimizeWindow(j, i);
        
        /* Set the dark and bright colors */
        dark_(j, i) = optimal(0);
        bright_(j, i) = optimal(1);
        
        dark_(j, i+1) = optimal(2);
        bright_(j, i+1) = optimal(3);
        
        dark_(j+1, i) = optimal(4);
        bright_(j+1, i) = optimal(5);
        
        dark_(j+1, i+1) = optimal(6);
        bright_(j+1, i+1) = optimal(7);
      }
    }
  }
}