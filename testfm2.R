library(devtools)
library(Matrix)
Sys.setenv("PKG_CXXFLAGS"="-std=c++11")
load_all()

n = 1000
p = 3
m = p
K = 2

X = matrix(rnorm(n*m), n, m)

beta0 = 1
beta = rnorm(p)
v = matrix(rnorm(n*K), n, K)

y = rnorm(n) + 3 + X %*% beta
print(y)

y_ind = (1:n) - 1

X = as(X, 'TsparseMatrix')

rows = as.numeric(X@i)
cols = as.numeric(X@j)
values = as.numeric(X@x)
## print(cols)
## print(values)

out = sp(beta0, beta, v,
         values, rows, cols,
         y, y_ind, n, p)

pred = predictfm(out$beta0,
                 out$beta,
                 out$v,
                 values,
                 rows,
                 cols,
                 n,p)

print(cbind(y, pred))

print(sum((y-pred)^2)/n)
