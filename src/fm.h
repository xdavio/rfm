// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#ifndef FM_H
#define FM_H

#include <Eigen/Sparse>
#include <random>
#include <math.h>

using namespace Eigen;

#define ln2inv 1.44269504089

typedef Triplet<double> T;
typedef SparseMatrix<double, RowMajor> SMat;
typedef SparseVector<double> SVec;

// declare structs
struct Params {
    float * beta0;
    VectorXd * beta;
    MatrixXd * v;  // v.rows() == beta.size()!!
};

struct OptParams {
    OptParams(): optimizer(0), minibatch(0), n_outer(0), eta(0.0), \
                 lambda(0.0), eps(0.0), beta1(0.0), beta2(0.0){}
    int optimizer; // 0: adagrad, 1: adam
    int minibatch; // minibatch count
    int n_outer;   // maxiter
    float eta;     // learning rate
    float lambda;  // penalty on v
    float eps;     // epsilon term for adam and adagrad
    float beta1;   // adagrad-only
    float beta2;   // adagrad-only
    int response_type; // 0 is linear, 1 is binary
};

class SparseFM
{
 private:
  SMat m;
  int response_type;
 public:
  SparseFM(const VectorXd & values,
           const VectorXi & rows,
           const VectorXi & cols,
           int & nrow,
           int & ncol,
           int & response_type_int)
    {
      m = SMat(nrow, ncol);
      response_type = response_type_int;

      make_matrix(values, rows, cols);
    }
    
    SparseFM(const VectorXd & values,
             const VectorXi & rows,
             const VectorXi & cols,
             int & nrow,
             int & ncol)
    {
        m = SMat(nrow, ncol);
        
        make_matrix(values, rows, cols);
    }

    inline void make_matrix(const VectorXd & values,
                            const VectorXi & rows,
                            const VectorXi & cols) {
        std::vector<T> triplets;
        triplets.reserve(values.size());
        for (int i=0; i<values.size(); ++i)
            {
                triplets.push_back(T(rows(i), cols(i), values(i)));
            }
        m.setFromTriplets(triplets.begin(), triplets.end());
    }

  inline const SMat & matrix() const {
    // returns a reference to the SMat m
    return m;
  }

  inline float predict(const int row,
		const float beta0,
		const VectorXd & beta,
		const MatrixXd & v)
  {
    // compute fm model for row `row`
    // eval fm likelihood.
    
    float out = 0.0;
    out += beta0;
    
    for (SMat::InnerIterator it(m, row); it; ++it) {
      out += beta(it.index()) * it.value();
      
      for (SMat::InnerIterator subit(m, row); subit; ++subit) {
        // this condition is needed for the i < j part of the FM's interaction terms
        if (it.index() < subit.index()) {
          out += v.row(it.index()).dot(v.row(subit.index())) * \
              it.value() * subit.value();
        }
      }
    }
    return out;
  }

  inline float derm(int row,
                    float beta0,
                    VectorXd & beta,
                    MatrixXd & v,
                    VectorXd & Y,
                    VectorXd & w)
    {
        if (response_type == 0) {
            // computes the derivative of squared loss with respect to the model
            return -2 * w(row) * (Y(row) - predict(row, beta0, beta, v));
        } else {
            // computes the derivative of squared loss with respect to the model
            float pred = predict(row, beta0, beta, v);
            float t = exp(-Y(row) * pred);
            return w(row) * ln2inv * (-Y(row) * t) / (1 + t);
        }
    }
};

// class LogisticSparseFM : public SparseFM
// {
//  public:
//   LogisticSparseFM(const VectorXd & values,
//                    const VectorXi & rows,
//                    const VectorXi & cols,
//                    int & nrow,
//                    int & ncol)
//     : SparseFM(values, rows, cols, nrow, ncol)
//     {}

//   inline float derm(int row,
//                     float beta0,
//                     VectorXd & beta,
//                     MatrixXd & v,
//                     VectorXd & Y,
//                     VectorXd & w)
//     {
//       // computes the derivative of squared loss with respect to the model
//       float pred = predict(row, beta0, beta, v);
//       float t = exp(-Y(row) * pred);
//       return w(row) * ln2inv * (-Y(row) * t) / (1 + t);
//     }
// };

// make sparse vector
template <typename T_val, typename T_ind>
void msv(SVec& v,
         T_val & values,
         T_ind & ind)
{
    // make sparse vector

    // fills sparse vector `v` from values `values`
    // and indices `ind`
    
    for (int i=0; i<ind.size(); ++i) {
        v.coeffRef(ind(i)) = values(i);
    }
}

// fit the factorization machine
Params fit_fm(Params params,
              OptParams opt_params,
              const Eigen::VectorXd & values,
              const Eigen::VectorXi & rows,
              const Eigen::VectorXi & cols,
              Eigen::VectorXd & Y,
              int nrow,
              int ncol,
              VectorXd & w);

#endif // FM_H
