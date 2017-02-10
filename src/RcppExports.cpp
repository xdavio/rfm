// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// sp
Rcpp::List sp(float beta0, Eigen::VectorXd& beta, Eigen::MatrixXd& v, const Eigen::VectorXd& values, const Eigen::VectorXi& rows, const Eigen::VectorXi& cols, const Eigen::VectorXd& y_values, const Eigen::VectorXi& y_ind, int nrow, int ncol);
RcppExport SEXP fm_sp(SEXP beta0SEXP, SEXP betaSEXP, SEXP vSEXP, SEXP valuesSEXP, SEXP rowsSEXP, SEXP colsSEXP, SEXP y_valuesSEXP, SEXP y_indSEXP, SEXP nrowSEXP, SEXP ncolSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< float >::type beta0(beta0SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type v(vSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type values(valuesSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXi& >::type rows(rowsSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXi& >::type cols(colsSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y_values(y_valuesSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXi& >::type y_ind(y_indSEXP);
    Rcpp::traits::input_parameter< int >::type nrow(nrowSEXP);
    Rcpp::traits::input_parameter< int >::type ncol(ncolSEXP);
    rcpp_result_gen = Rcpp::wrap(sp(beta0, beta, v, values, rows, cols, y_values, y_ind, nrow, ncol));
    return rcpp_result_gen;
END_RCPP
}
