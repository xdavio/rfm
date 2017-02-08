


library(devtools)
Sys.setenv("PKG_CXXFLAGS"="-std=c++11")

load_all()

n = 1000
m = 100

set.seed(1121)

rows = sample(n, 1000, replace=TRUE) - 1
cols = sample(m, 1000, replace=TRUE) - 1
values = rnorm(1000)

beta0 = 5
beta = rnorm(m) / 10
v = matrix(rnorm(n * 10), n, 10) / 10
y = rnorm(n) + beta0

out = sp(beta0, beta, v, values, rows, cols, y, (1:n) - 1, n, m)
print(out$beta0)
print(sum(is.nan(out$beta)))
print(sum(is.nan(out$v)))



