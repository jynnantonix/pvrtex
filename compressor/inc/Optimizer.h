/*=========================================================================*/
/*                                                                         */
/* @file            Optimizer.h                                            */
/* @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             */
/* @date            2012/11/14                                             */
/* @version         0.2                                                    */
/* @brief           Optimizer for generating bright and dark images        */
/*                                                                         */
/*=========================================================================*/

#ifndef __pvrtex__Optimizer__
#define __pvrtex__Optimizer__

#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "Util.h"

namespace pvrtex {
  class Optimizer {
  public:
    enum SOLVER { SVD, QR, CHOLESKY };
    
    Optimizer(const Eigen::MatrixXi &o,
              Eigen::MatrixXi &d,
              Eigen::MatrixXi &b);
    Optimizer(const Eigen::MatrixXi &o,
              Eigen::MatrixXi &d,
              Eigen::MatrixXi &b,
              SOLVER s);
    ~Optimizer();
    
    inline Eigen::MatrixXi dark() { return dark_; }
    inline Eigen::MatrixXi bright() { return bright_; }
    inline Eigen::MatrixXf mod() { return mod_; }
    
    void Optimize(const Eigen::MatrixXf &m);
  private:
    Optimizer();
    
    void OptimizeWindow(int j, int i);
    void ComputeUpdateVector();
    
    Eigen::MatrixXi red_;
    Eigen::MatrixXi green_;
    Eigen::MatrixXi blue_;
    Eigen::MatrixXi dark_;
    Eigen::MatrixXi bright_;
    Eigen::MatrixXf mod_;
    const Eigen::MatrixXi orig_;
    const SOLVER solv_;
    
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
