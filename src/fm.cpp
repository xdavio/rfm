// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include "fm.h"

using namespace Eigen;

Params fit_fm(Params params,
              OptParams opt_params,
              const Eigen::VectorXd & values,
              const Eigen::VectorXi & rows,
              const Eigen::VectorXi & cols,
              Eigen::VectorXd & Y,
              int nrow,
              int ncol,
              Eigen::VectorXd & w)
{
    float & beta0 = *params.beta0;
    VectorXd & beta = *params.beta;
    MatrixXd & v = *params.v;

    // set universal opt params
    int & minibatch = opt_params.minibatch;
    int & n_outer = opt_params.n_outer;
    float & eta = opt_params.eta;
    float & eps = opt_params.eps;
    float & beta1 = opt_params.beta1; // ADAM parameters
    float & beta2 = opt_params.beta2;

    int & loss = opt_params.loss;
        
    // penalty parameter
    const float & lambda = opt_params.lambda;

    // get optimizer
    int & optimizer = opt_params.optimizer;

    // declare all possible variables
    float G_beta0;
    VectorXd G_beta;
    MatrixXd G_v;
    float a_m_beta0;
    VectorXd a_m_beta;
    MatrixXd a_m_v;
    float a_v_beta0;
    VectorXd a_v_beta;
    MatrixXd a_v_v;
    if (optimizer == 0) {
        // Adagrad parameters
        // float eps = 1.0e-8;
        G_beta0 = 0;
        G_beta = VectorXd::Zero(ncol);
        G_v = MatrixXd::Zero(v.rows(), v.cols());
    } else if (optimizer == 1) {
        // ADAM parameters
        a_m_beta0 = 0;
        a_m_beta = VectorXd::Zero(ncol);
        a_m_v = MatrixXd::Zero(v.rows(), v.cols());

        a_v_beta0 = 0;
        a_v_beta = VectorXd::Zero(ncol);
        a_v_v = MatrixXd::Zero(v.rows(), v.cols());
    }
    
    // make sparse matrix X
    SparseFM X(values, rows, cols, nrow, ncol, loss);
    SMat Xval = X.matrix(); // ref the SMat

    float beta0_derlik;
    VectorXd beta_derlik(beta.size());
    MatrixXd v_derlik(v.rows(), v.cols());
    VectorXd v_precompute(v.cols());

    // set up RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    gen.seed(43824);
    std::uniform_int_distribution<> dis(0, nrow-1);
    
    for (int outer_it=0; outer_it<n_outer; ++outer_it) {
        // set all derliks to zero
        beta0_derlik = 0;
        beta_derlik.setZero();
        v_derlik.setZero();

        // estimate the gradients into the derlik variables
        float derlik;
        int rand;
        for (int i=0; i<minibatch; ++i) {
            rand = dis(gen);  // get random ind

            // derivative of loss w.r.t. model
            derlik = X.derm(rand, beta0, beta, v, Y, w);

            v_precompute.setZero();
            for (SMat::InnerIterator it(Xval, rand); it; ++it) {
                v_precompute += v.row(it.index()) * it.value();
            }
        
            beta0_derlik += derlik;
            for (SMat::InnerIterator it(Xval, rand); it; ++it) {
                beta_derlik(it.index()) += derlik * it.value() \
                    + 2 * lambda * beta(it.index());
                // + 2 * lambda * beta(it.index())
                v_derlik.row(it.index()) += derlik * \
                    (it.value() * v_precompute - \
                     it.value() * it.value() * v.row(it.index())) + \
                    2 * lambda * v.row(it.index());
                // 2 * lambda * v.row(it.index());
            }
        }

        beta0_derlik /= minibatch;
        beta_derlik /= minibatch;
        v_derlik /= minibatch;

        if (optimizer == 0) {        
            // Adagrad update
            G_beta0 += pow(beta0_derlik, 2);
            G_beta += beta_derlik.array().square().matrix();
            G_v += v_derlik.array().square().matrix();
        
            beta0_derlik /= pow(G_beta0 + eps, .5);
            beta_derlik = (beta_derlik.array()/(G_beta.array() + eps).sqrt()).matrix();
            v_derlik = (v_derlik.array()/(G_v.array() + eps).sqrt()).matrix();
        
            // update parameters
            beta0 -= eta * beta0_derlik;
            beta -= eta * beta_derlik;
            v -= eta * v_derlik;
        } else if (optimizer == 1) {
            // ADAM update
            // first moment
            a_m_beta0 = beta1 * a_m_beta0 + (1-beta1) * beta0_derlik;
            a_m_beta = beta1 * a_m_beta + (1-beta1) * beta_derlik;
            a_m_v = beta1 * a_m_v + (1-beta1) * v_derlik;

            // first moment bias correction
            a_m_beta0 /= (1 - pow(beta1, outer_it + 1));
            a_m_beta /= (1 - pow(beta1, outer_it + 1));
            a_m_v /= (1 - pow(beta1, outer_it + 1));

            // second moment
            a_v_beta0 = beta2 * a_v_beta0 + (1-beta2) * pow(beta0_derlik, 2);
            a_v_beta = beta2 * a_v_beta + (1-beta2) * beta_derlik.array().square().matrix();
            a_v_v = beta2 * a_v_v + (1-beta2) * v_derlik.array().square().matrix();

            // second moment bias correction
            a_v_beta0 /= (1 - pow(beta2, outer_it + 1));
            a_v_beta /= (1 - pow(beta2, outer_it + 1));
            a_v_v /= (1 - pow(beta2, outer_it + 1));

            // update parameters
            beta0 -= eta * a_m_beta0 / (pow(a_v_beta0, .5) + eps);
            beta -= eta * (a_m_beta.array() / (a_v_beta.array().sqrt() + eps)).matrix();
            v -= eta * (a_m_v.array() / (a_v_v.array().sqrt() + eps)).matrix();
        } else if (optimizer == 2) {
            // SGD update
            beta0 -= eta * beta0_derlik;
            beta -= eta * beta_derlik;
            v -= eta * v_derlik;
        }
    }

    return params;
}
