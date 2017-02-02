library(devtools)
Sys.setenv("PKG_CXXFLAGS"="-std=c++11")

load_all()


main()

X <- rnorm(100)
X <- ifelse(X > 0, 0, 1)
X <- matrix(X, 10, 10)
y <- rnorm(100)



sgd(X, y, 10, 2)

library(devtools)
load_all()

rows = c(1,1,1)
cols = c(1,2,90)
values = c(1, 1, 1)
nrow = 100
ncol = 100
#sp1(values, rows, cols, nrow, ncol)

beta0 = 1.1
beta = rnorm(100)
v = matrix(rnorm(100 * 10), 100, 10)
y = rnorm(100)

sp3(beta0, beta, v, values, rows, cols, y, nrow, ncol)

for (j in 1:100) {
    sp2(j-1, beta0, beta, v, values, rows, cols, nrow, ncol)
}
