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
    
    return -2 * (Y.coeffRef(row) - fm(m, row, beta0, beta, v));
}

// [[Rcpp::export]]
Rcpp::List sp(float beta0,
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
    // hardcoding these for now
    int minibatch = 50;
    int n_outer = 2000;
    float eta = .3;

    // ADAM parameters
    float beta1 = .9;
    float beta2 = .999;
    float eps = .00000001;
    float a_m_beta0 = 0;
    VectorXd a_m_beta(ncol);
    a_m_beta.setZero();
    MatrixXd a_m_v(v.rows(), v.cols());
    a_m_v.setZero();
    float a_v_beta0 = 0;
    VectorXd a_v_beta(ncol);
    a_v_beta.setZero();
    MatrixXd a_v_v(v.rows(), v.cols());
    a_v_v.setZero();
    
    
    // make sparse matrix X
    SMat X(nrow, ncol);
    msp(X, values, rows, cols);

    // make sparse response Y
    SVec Y(nrow);
    msv(Y, y_values, y_ind);

    for (int outer_it=0; outer_it<n_outer; ++outer_it) {
        // set all caches to zero
        float beta0_cache = 0;
        VectorXd beta_cache(beta.size());
        beta_cache.setZero();
        MatrixXd v_cache(v.rows(), v.cols());
        v_cache.setZero();
        VectorXd v_precompute(v.cols());

        // estimate the gradients into the cache variables
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

        beta0_cache /= minibatch;
        beta_cache /= minibatch;
        v_cache /= minibatch;

        // ADAM update
        // first moment
        a_m_beta0 = beta1 * a_m_beta0 + (1-beta1) * beta0_cache;
        a_m_beta = beta1 * a_m_beta + (1-beta1) * beta_cache;
        a_m_v = beta1 * a_m_v + (1-beta1) * v_cache;

        // first moment bias correction
        a_m_beta0 /= (1 - pow(beta1, outer_it + 1));
        a_m_beta /= (1 - pow(beta1, outer_it + 1));
        a_m_v /= (1 - pow(beta1, outer_it + 1));

        // second moment
        a_v_beta0 = beta2 * a_v_beta0 + (1-beta2) * pow(beta0_cache, 2);
        a_v_beta = beta2 * a_v_beta + (1-beta2) * beta_cache.array().square().matrix();
        a_v_v = beta2 * a_v_v + (1-beta2) * v_cache.array().square().matrix();

        // second moment bias correction
        a_v_beta0 /= (1 - pow(beta2, outer_it + 1));
        a_v_beta /= (1 - pow(beta2, outer_it + 1));
        a_v_v /= (1 - pow(beta2, outer_it + 1));

        // update parameters
        beta0 -= eta * a_m_beta0 / (pow(a_v_beta0, .5) + eps);
        //beta = beta - eta * a_m_beta / (a_v_beta.array().sqrt() + eps);
        beta -= eta * (a_m_beta.array() / (a_v_beta.array().sqrt() + eps)).matrix();
        //v = v - eta * a_m_v / (a_v_v.array().sqrt() + eps);
        v -= eta * (a_m_v.array() / (a_v_v.array().sqrt() + eps)).matrix();
        
        // update parameters
        // beta0 -= eta * beta0_cache;
        // beta -= eta * beta_cache;
        // v -= eta * v_cache;
    }

    // std::cout << "beta0_new: " << beta0 << std::endl;
    // std::cout << "beta_new: " << beta << std::endl;
    // std::cout << "v_new: " << v << std::endl;

    return Rcpp::List::create(
                              Rcpp::Named("beta0") = beta0,
                              Rcpp::Named("beta") = beta,
                              Rcpp::Named("v") = v
                        );
}
