// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <random>

using namespace Eigen;

typedef Triplet<double> T;
typedef SparseMatrix<double, RowMajor> SMat;
typedef SparseVector<double> SVec;

void msp(SMat& m,
         const Eigen::VectorXd & values,
         const Eigen::VectorXi & rows,
         const Eigen::VectorXi & cols)
{
    // make sparse matrix

    // fills `m` from triplets given by
    // `values`: actual values
    // `rows`:   row indices
    // `cols`:   col indices
    
    std::vector<T> triplets;
    triplets.reserve(values.size());
    for (int i=0; i<values.size(); ++i)
        {
            triplets.push_back(T(rows(i), cols(i), values(i)));
        }
    m.setFromTriplets(triplets.begin(), triplets.end());
}

void msv(SVec& v,
         const Eigen::VectorXd & values,
         const Eigen::VectorXi & ind)
{
    // make sparse vector

    // fills sparse vector `v` from values `values`
    // and indices `ind`
    
    for (int i=0; i<ind.size(); ++i) {
        v.coeffRef(ind(i)) = values(i);
    }
}

int rand_ind(int& rowmax)
{
    //  generates a random integer with max `rowmax`

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, rowmax-1);

    return dis(gen);
}

float fm(SMat& m, int row, float beta0,
         const VectorXd& beta,
         const MatrixXd& v)
{
    // compute fm model for row `row`
    // eval fm likelihood.

    float out = 0.0;
    out += beta0;

    for (SMat::InnerIterator it(m, row); it; ++it) {
        out += beta(it.index()) * it.value();

        for (SMat::InnerIterator subit(m, row); it; ++it) {
            out += v.row(it.index()).dot(v.row(subit.index())) *\
                it.value() * subit.value();
        }
    }
    return out;
}

float derm(SMat& m, int row, float beta0,
           const VectorXd& beta,
           const MatrixXd& v,
           SVec& Y)
{
    // computes the derivative of squared loss with respect to the model
    
    return 2 * (Y.coeffRef(row) - fm(m, row, beta0, beta, v));
}

// [[Rcpp::export]]
float sp(float beta0,
         Eigen::VectorXd & beta,
         Eigen::MatrixXd & v,
         const Eigen::VectorXd & values,
         const Eigen::VectorXi & rows,
         const Eigen::VectorXi & cols,
         const Eigen::VectorXd & y_values,
         const Eigen::VectorXi & y_ind,
         int nrow,
         int ncol)
{
    // make sparse matrix X
    SMat X(nrow, ncol);
    msp(X, values, rows, cols);

    // make sparse response Y
    SVec Y(nrow);
    msv(Y, y_values, y_ind);

    int minibatch = 30;

    
    VectorXi ind(minibatch);
    for (int i=0; i<minibatch; ++i) {
        ind(i) = rand_ind(nrow);
    }

    float beta0_cache = 0;
    VectorXd beta_cache(beta.size());
    beta_cache.setZero();
    MatrixXd v_cache(v.rows(), v.cols());
    v_cache.setZero();
    VectorXd v_precompute(v.cols());
    
    float cache;
    int rand;
    for (int i=0; i<minibatch; ++i) {
        rand = rand_ind(nrow);
        cache = derm(X, rand, beta0, beta, v, Y);

        v_precompute.setZero();
        for (SMat::InnerIterator it(X, rand); it; ++it) {
            v_precompute += v.row(it.index()) * it.value();
        }
        
        beta0_cache += cache;
        for (SMat::InnerIterator it(X, rand); it; ++it) {
            beta_cache(it.index()) += cache * it.value();
            v_cache.row(it.index()) += cache * (it.value() * v_precompute - it.value() * it.value() * v.row(it.index()));
        }
    }
    
    float eta = .1;
    beta0 -= eta / minibatch * beta0_cache;
    beta -= eta / minibatch * beta_cache;
    v -= eta / minibatch * v_cache;

    std::cout << "beta0_new: " << beta0 << std::endl;
    std::cout << "beta_new: " << beta << std::endl;
    std::cout << "v_new: " << v << std::endl;
    
    return eta;
}


