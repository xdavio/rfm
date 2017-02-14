library(devtools)
library(Matrix)
Sys.setenv("PKG_CXXFLAGS"="-std=c++11")
load_all()

n = 10000
p = 100
m = p
K = 10

X = matrix(rnorm(n*m), n, m)

beta0 = 1
beta = rnorm(p)
v = matrix(rnorm(p*K), p, K)

#y = rnorm(n) + 3 + X %*% beta
#y = .4 + X %*% beta
y = 3 + X %*% beta + .5* X[,1] * X[,2]
y_ind = (1:n) - 1

print(summary(lm(y ~ X)))

X = as(X, 'TsparseMatrix')

rows = as.numeric(X@i)
cols = as.numeric(X@j)
values = as.numeric(X@x)
## print(cols)
## print(values)


opt_params = list(minibatch=128,
                  n_outer=10000,
                  eta=.1,
                  lambda=1)

out = sp(beta0, beta/100*0, v*.01, opt_params,
         values, rows, cols,
         y, y_ind, n, p)

pred = predictfm(out$beta0,
                 out$beta,
                 out$v,
                 values,
                 rows,
                 cols,
                 n,p)

print("compare y with pred")
print(cbind(y, pred))

print("compare betas: real to estimated")
print(cbind(3, out$beta0))
print(cbind(beta, out$beta))

print("squared error")
print(sum((y-pred)^2)/n)

apply(out$v, 1, FUN=function(x) sqrt(sum(x*x)))  

print(sum(v[1, ] * v[2, ]))
