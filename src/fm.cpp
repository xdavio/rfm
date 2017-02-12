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

        for (SMat::InnerIterator subit(m, row); subit; ++subit) {
            out += v.row(it.index()).dot(v.row(subit.index())) *\
                it.value() * subit.value();
        }
    }
    return out;
}

float derm(SMat& m, int row, float beta0,
           const VectorXd & beta,
           const MatrixXd & v,
           SVec & Y)
{
    // computes the derivative of squared loss with respect to the model
    
    return -2 * (Y.coeffRef(row) - fm(m, row, beta0, beta, v));
}

struct Params {
    float * beta0;
    VectorXd * beta;
    MatrixXd * v;  // v.rows() == beta.size()!!
};

struct OptParams {
    int minibatch;
    int n_outer;
    float eta;
};

Params fit_fm(Params params,
              OptParams opt_params,
              const Eigen::VectorXd & values,
              const Eigen::VectorXi & rows,
              const Eigen::VectorXi & cols,
              const Eigen::VectorXd & y_values,
              const Eigen::VectorXi & y_ind,
              int nrow,
              int ncol)
{
    float & beta0 = *(params.beta0);
    VectorXd & beta = *(params.beta);
    MatrixXd & v = *(params.v);

    // hardcoding these for now
    int & minibatch = opt_params.minibatch;
    int & n_outer = opt_params.n_outer;
    float & eta = opt_params.eta;

    // penalty parameters
    float lambda = 1;

    // ADAM parameters
    /*
    float beta1 = .9;
    float beta2 = .999;
    float eps = .0000001;
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
    */

    // Adagrad parameters
    float eps = 1.0e-8;
    float G_beta0 = 0;
    VectorXd G_beta = VectorXd::Zero(ncol);
    MatrixXd G_v = MatrixXd::Zero(v.rows(), v.cols());
    
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

            // derivative of loss w.r.t. model
            cache = derm(X, rand, beta0, beta, v, Y);

            v_precompute.setZero();
            for (SMat::InnerIterator it(X, rand); it; ++it) {
                v_precompute += v.row(it.index()) * it.value();
            }
        
            beta0_cache += cache;
            for (SMat::InnerIterator it(X, rand); it; ++it) {
                beta_cache(it.index()) += cache * it.value();
                // + 2 * lambda * beta(it.index())
                v_cache.row(it.index()) += cache * (it.value() * v_precompute - it.value() * it.value() * v.row(it.index())) + 2 * lambda * v.row(it.index());
                // + 2 * lambda * v.row(it.index());
            }
        }

        beta0_cache /= minibatch;
        beta_cache /= minibatch;
        v_cache /= minibatch;

        // ADAM update
        /*
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
        beta -= eta * (a_m_beta.array() / (a_v_beta.array().sqrt() + eps)).matrix();
        v -= eta * (a_m_v.array() / (a_v_v.array().sqrt() + eps)).matrix();
        */

        // Adagrad update
        G_beta0 += pow(beta0_cache, 2);
        G_beta += beta_cache.array().square().matrix();
        G_v += v_cache.array().square().matrix();
        
        beta0_cache /= pow(G_beta0 + eps, .5);
        beta_cache = (beta_cache.array()/(G_beta.array() + eps).sqrt()).matrix();
        v_cache = (v_cache.array()/(G_v.array() + eps).sqrt()).matrix();
        
        // update parameters
        beta0 -= eta * beta0_cache;
        beta -= eta * beta_cache;
        v -= eta * v_cache;
    }

    return params;
}

// [[Rcpp::export]]
Rcpp::List sp(float & beta0,
              Eigen::VectorXd & beta,
              Eigen::MatrixXd & v,
              const Rcpp::List & opt_params_l,
              const Eigen::VectorXd & values,
              const Eigen::VectorXi & rows,
              const Eigen::VectorXi & cols,
              const Eigen::VectorXd & y_values,
              const Eigen::VectorXi & y_ind,
              int nrow,
              int ncol)
{

    // process params
    Params params;
    params.beta0 = &beta0;
    params.beta = &beta;
    params.v = &v;

    // process optimizer params
    OptParams opt_params;
    opt_params.minibatch = Rcpp::as<int>(opt_params_l["minibatch"]);
    opt_params.n_outer = Rcpp::as<int>(opt_params_l["n_outer"]);
    opt_params.eta = Rcpp::as<float>(opt_params_l["eta"]);
    

    params = fit_fm(params,
                    opt_params,
                    values,
                    rows,
                    cols,
                    y_values,
                    y_ind,
                    nrow,
                    ncol);
    
    return Rcpp::List::create(
                              Rcpp::Named("beta0") = *(params.beta0),
                              Rcpp::Named("beta") = *(params.beta),
                              Rcpp::Named("v") = *(params.v)
                              );
}

// [[Rcpp::export]]
Eigen::VectorXd predictfm(float beta0,
                          Eigen::VectorXd & beta,
                          Eigen::MatrixXd & v,
                          const Eigen::VectorXd & values,
                          const Eigen::VectorXi & rows,
                          const Eigen::VectorXi & cols,
                          int nrow,
                          int ncol)
{
    // make sparse matrix X
    SMat X(nrow, ncol);
    msp(X, values, rows, cols);

    VectorXd yhat(nrow);
    for (int i; i<nrow; ++i) {
        yhat(i) = fm(X, i, beta0, beta, v);
    }

    return yhat;
}
