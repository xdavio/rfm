// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

#include <Eigen/Core>
#include <random>

using namespace Eigen;

// [[Rcpp::export]]
Rcpp::List sgd(const Eigen::MatrixXd & X,
	       const Eigen::VectorXd & y,
	       int K,
	       int minibatch_size)
{
  int N = X.rows();    // sample size
  int M = X.cols();    // n_features
  
  VectorXd beta(M);  // main effects
  MatrixXd v(M, K);  // latent space
  VectorXd intercept(1);  // intercept

  VectorXd betaold(M);  // main effects old
  MatrixXd vold(M, K);  // latent space old
  VectorXd interceptold(1);   // intercept old

  // permutation matrix
  PermutationMatrix<Dynamic, Dynamic> perm(N);
  perm.setIdentity();

  // minibatch data
  MatrixXd mdata(minibatch_size, M);
  VectorXd mresponse(minibatch_size);
  
  // random guesses for the parameters
  betaold.setRandom();
  vold.setRandom();
  interceptold.setRandom();

  beta.setZero();
  v.setZero();
  intercept.setZero();

  int iter = 0;
  while (((beta-betaold).squaredNorm() > .00001 &&
	  (v - vold).squaredNorm() > .00001) &&
	 iter < 100)
    {
      iter += 1;
      std::cout << "iter: " << iter << std::endl;

      float out = 0;
      int r;
      for (int j=0; j<minibatch_size; ++j)
	{
            //r = rand_ind(minibatch_size);
	  //out += y(r) - interceptold - betaold * X.row(r) - 
	}
	
      // std::random_shuffle(perm.indices().data(),
      // 			  perm.indices().data()+perm.indices().size());

      // mdata = (perm * X).topRows(minibatch_size);
      // std::cout << mdata << std::endl;

      // //mresponse = (perm * y).head(minibatch_size);
      // std::cout << (perm * y).head(minibatch_size) << std::endl;

      
    }

  

  return Rcpp::List::create(Rcpp::Named("")=1,
  			    Rcpp::Named("inner")=1);
  
}

// int rand_ind(int & rowmax)
// {
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_int_distribution<> dis(0, rowmax-1);

//   return dis(gen);
// }


float model(const VectorXd & intercept,
	    const VectorXd & beta,
	    const MatrixXd & v,
	    const MatrixXd & data,
	    const int & r)
{
    //data.row(r)
    return 1.0;
}
