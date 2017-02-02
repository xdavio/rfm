// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <random>

using namespace Eigen;

template <typename Derived>
void write(const MatrixBase<Derived>& out)
{
  std::cout << out << std::endl;
}

// [[Rcpp::export]]
int main()
{
  int N = 5;

  // matrix
  MatrixXd m(N, N);
  m.setRandom();
  write(m);

  // array
  ArrayXi m1(6);
  m1.setRandom();
  write(m1.matrix());

  std::cout << "sum: " << m.sum() << std::endl;

  Map<Matrix<double, Dynamic, 1>> m2(m.data(), m.size());
  std::cout << "map: " << m2 << std::endl;
  
  return 1;
}

typedef Triplet<double> T;
typedef SparseMatrix<double, RowMajor> SMat;

void msp(SMat& m,
         const Eigen::VectorXd & values,
         const Eigen::VectorXi & rows,
         const Eigen::VectorXi & cols)
{
    // make sparse vector
    
    std::vector<T> triplets;
    triplets.reserve(values.size());
    for (int i=0; i<values.size(); ++i)
        {
            triplets.push_back(T(rows(i), cols(i), values(i)));
        }
    m.setFromTriplets(triplets.begin(), triplets.end());
}

int rand_ind(int & rowmax)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, rowmax-1);

  return dis(gen);
}

float fm(SMat& m, int row, float beta0,
         const VectorXd& beta,
         const MatrixXd& v)
{
    // compute fm model
    // eval fm likelihood
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
           const VectorXd& y) {
    return 2 * (y(row) - fm(m, row, beta0, beta, v));
}

// [[Rcpp::export]]
float sp3(float beta0,
          Eigen::VectorXd & beta,
          Eigen::MatrixXd & v,
          const Eigen::VectorXd & values,
          const Eigen::VectorXi & rows,
          const Eigen::VectorXi & cols,
          const Eigen::VectorXd & y,
          int nrow,
          int ncol)
{
    // make sparse matrix X
    SMat X(nrow, ncol);
    msp(X, values, rows, cols); 

    int minibatch = 30;
    float out = 0;
    int ind;
    for (int i=0; i<minibatch; ++i) {
        ind = rand_ind(nrow);
        out += derm(X, ind, beta0, beta, v, y);
    }
    out = out / minibatch;

    beta0 -= .1 * out * 1;
    for (int k=0; k<X.outerStride(), ++k) {
        for (SMat::InnerIterator it(m, k); it; ++it) {
            beta(it.index()) -= .1 * out * it.value();
        }
    }
    
    return out;
}


// [[Rcpp::export]]
int sp2(int row,
        float beta0,
        const Eigen::VectorXd & beta,
        const Eigen::MatrixXd & v,
        const Eigen::VectorXd & values,
        const Eigen::VectorXi & rows,
        const Eigen::VectorXi & cols,
        int nrow,
        int ncol)
{
    SMat m(nrow, ncol);
    msp(m, values, rows, cols);
    
    float out = fm(m, row, beta0, beta, v);
    std::cout << out << std::endl;

    return 0;
}

// [[Rcpp::export]]
int sp1(const Eigen::VectorXd & values,
        const Eigen::VectorXi & rows,
        const Eigen::VectorXi & cols,
        int nrow,
        int ncol)
{
    SMat m(nrow, ncol);
    msp(m, values, rows, cols);
    
    for (int k=0; k<m.outerSize(); ++k) {
        std::cout << "k: " << k << std::endl;
        
        for (SMat::InnerIterator it(m, k); it; ++it) {
            std::cout << "value:" << std::endl;
            std::cout << it.value() << std::endl;

            std::cout << "row:" << std::endl;            
            std::cout << it.row() << std::endl;

            std::cout << "col:" << std::endl;            
            std::cout << it.col() << std::endl;

            std::cout << "index:" << std::endl;            
            std::cout << it.index() << std::endl;
        }
    }
    
    return 0;
}
