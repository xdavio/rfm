// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

#include "fm.h"

using namespace Eigen;

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

    Params params = {&beta0, &beta, &v};

    // process optimizer params
    OptParams opt_params = {Rcpp::as<int>(opt_params_l["minibatch"]),
                            Rcpp::as<int>(opt_params_l["n_outer"]),
                            Rcpp::as<float>(opt_params_l["eta"]),
                            Rcpp::as<float>(opt_params_l["lambda"])};

    params = fit_fm(params,
                    opt_params,
                    values,
                    rows,
                    cols,
                    y_values,
                    y_ind,
                    nrow,
                    ncol);
    
    return Rcpp::List::create(Rcpp::Named("beta0") = *params.beta0,
                              Rcpp::Named("beta") = *params.beta,
                              Rcpp::Named("v") = *params.v);
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
    SparseFM X(values, rows, cols, nrow, ncol);

    VectorXd yhat(nrow);
    for (int i=0; i<nrow; ++i) {
        yhat(i) = X.predict(i, beta0, beta, v);
    }

    return yhat;
}
