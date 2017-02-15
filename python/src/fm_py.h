#ifndef FM_PY
#define FM_PY

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "fm.h"

using namespace Eigen;
//const pybind11::dict

pybind11::dict c_fit_fm(float & beta0,
			Eigen::VectorXd & beta,
			Eigen::MatrixXd & v,
			const std::map<std::string, float> & opt_params_l,
			const Eigen::VectorXd & values,
			const Eigen::VectorXi & rows,
			const Eigen::VectorXi & cols,
			const Eigen::VectorXd & y_values,
			const Eigen::VectorXi & y_ind,
			int nrow,
			int ncol);

VectorXd c_predictfm(float beta0,
		     Eigen::VectorXd & beta,
		     Eigen::MatrixXd & v,
		     const Eigen::VectorXd & values,
		     const Eigen::VectorXi & rows,
		     const Eigen::VectorXi & cols,
		     int nrow,
		     int ncol);
  
#endif // FM_PY
