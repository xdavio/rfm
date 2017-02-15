// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

//#include <pybind11/pybind11.h>
//#include <pybind11/pytypes.h>
//#include <pybind11/stl.h>

#include "fm_py.h"

using namespace Eigen;

void c_fit_fm(float & beta0,
              Eigen::VectorXd & beta,
              Eigen::MatrixXd & v,
              std::map<std::string, float> & opt_params_l,
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
  OptParams opt_params = {static_cast<int>(opt_params_l["minibatch"]),
			  static_cast<int>(opt_params_l["n_outer"]),
			  static_cast<float>(opt_params_l["eta"]),
			  static_cast<float>(opt_params_l["lambda"])};

  params = fit_fm(params,
  		  opt_params,
  		  values,
  		  rows,
  		  cols,
  		  y_values,
  		  y_ind,
  		  nrow,
  		  ncol);

  beta0 = *params.beta0;
  beta = *params.beta;
  v = *params.v;
}

VectorXd c_predictfm(float beta0,
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
