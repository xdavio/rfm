// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#ifndef FM_H
#define FM_H

#include <Eigen/Sparse>
#include <random>

using namespace Eigen;

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
    int minibatch;
    int n_outer;
    float eta;
    float lambda;
};

class SparseFM
{
 private:
  SMat m;
 public:
  SparseFM(const VectorXd & values,
		 const VectorXi & rows,
		 const VectorXi & cols,
		 int & nrow,
		 int & ncol)
    {
      m = SMat(nrow, ncol);
      
      std::vector<T> triplets;
      triplets.reserve(values.size());
      for (int i=0; i<values.size(); ++i)
        {
	  triplets.push_back(T(rows(i), cols(i), values(i)));
        }
      m.setFromTriplets(triplets.begin(), triplets.end());
    }

  const SMat & matrix() const {
    // returns a reference to the SMat m
    return m;
  }

  float predict(const int row,
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
	out += v.row(it.index()).dot(v.row(subit.index())) *	\
	  it.value() * subit.value();
      }
    }
    return out;
  }

  float derm(const int row,
	     const float beta0,
	     const VectorXd & beta,
	     const MatrixXd & v,
	     SVec & Y)
  {
    // computes the derivative of squared loss with respect to the model
    return -2 * (Y.coeffRef(row) - predict(row, beta0, beta, v));
  }
};

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

// randomly sample an int from 0,...,rowmax - 1
template <typename T_val>
T_val rand_ind(T_val & rowmax)
{
    //  generates a random integer with max `rowmax`
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, rowmax-1);

    return dis(gen);
}

// fit the factorization machine
Params fit_fm(Params params,
              OptParams opt_params,
              const Eigen::VectorXd & values,
              const Eigen::VectorXi & rows,
              const Eigen::VectorXi & cols,
              const Eigen::VectorXd & y_values,
              const Eigen::VectorXi & y_ind,
              int nrow,
              int ncol);

#endif // FM_H
