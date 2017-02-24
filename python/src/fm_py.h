#ifndef FM_PY
#define FM_PY

/* #include <pybind11/pybind11.h> */
/* #include <pybind11/pytypes.h> */
/* #include <pybind11/stl.h> */

#include "fm.h"

using namespace Eigen;
//const pybind11::dict

std::tuple<float, Eigen::VectorXd, Eigen::MatrixXd> c_fit_fm(float & beta0,
							     Eigen::VectorXd & beta,
							     Eigen::MatrixXd & v,
							     std::map<std::string, float> & opt_params_l,
							     Eigen::VectorXd & values,
							     Eigen::VectorXi & rows,
							     Eigen::VectorXi & cols,
							     Eigen::VectorXd & y_values,
							     Eigen::VectorXi & y_ind,
							     int nrow,
							     int ncol,
							     Eigen::VectorXd w);

VectorXd c_predictfm(float beta0,
		     Eigen::VectorXd & beta,
		     Eigen::MatrixXd & v,
		     Eigen::VectorXd & values,
		     Eigen::VectorXi & rows,
		     Eigen::VectorXi & cols,
		     int nrow,
		     int ncol);
  
#endif // FM_PY
