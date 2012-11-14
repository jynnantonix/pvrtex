/*=========================================================================*/
/*                                                                         */
/* @file            Optimizer.h                                            */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/14                                             */
/* @version         0.1                                                    */
/* @brief           Optimizer for generating bright and dark images        */
/*                                                                         */
/*=========================================================================*/

#ifndef __pvrtex__Optimizer__
#define __pvrtex__Optimizer__

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "Util.h"

namespace pvrtex {
  class Optimizer {
  public:
    enum SOLVER { SVD, QR, CHOLESKY };
    
    Optimizer(Eigen::MatrixXi o, Eigen::MatrixXf m);
    Optimizer(Eigen::MatrixXi o, Eigen::MatrixXf m, SOLVER s);
    ~Optimizer();
    
    inline Eigen::MatrixXi dark() { return dark_; }
    inline Eigen::MatrixXi bright() { return bright_; }
    
    void Optimize();
  private:
    Optimizer();
    
    Eigen::VectorXi OptimizeWindow(int j, int i);
    
    Eigen::MatrixXi dark_;
    Eigen::MatrixXi bright_;
    Eigen::MatrixXi orig_;
    Eigen::MatrixXf mod_;
    SOLVER solv_;
    
    static const int window_width_;
    static const int window_height_;
    static const int matrix_rows_;
    static const int matrix_cols_;
    static const int offset_x_;
    static const int offset_y_;
    static const float kTopLeft[];
    static const float kTopRight[];
    static const float kBottomLeft[];
    static const float kBottomRight[];
  };
}

#endif /* defined(__pvrtex__Optimizer__) */
