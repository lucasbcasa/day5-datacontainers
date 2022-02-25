# Exercises for Day 5
Using SciPy, Scikit-Learn and Pandas

## 1. Scipy

### Linear Algebra
Have a look at the ```scipy.linalg``` [module](https://docs.scipy.org/doc/scipy/reference/linalg.html)

#### a. Define a matrix A
```
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

#### b. Define a vector b
```
[1 2 3]
```

#### c. Solve the linear system of equations A x = b

#### d. Check that your solution is correct by plugging it into the equation

#### e. Repeat steps a-d using a random 3x3 matrix B (instead of the vector b)

#### f. Solve the eigenvalue problem for the matrix A and print the eigenvalues and eigenvectors

#### g. Calculate the inverse, determinant of A

#### h. Calculate the norm of A with different orders


### Statistics
Have a look at the ```scipy.stats``` [module](https://docs.scipy.org/doc/scipy/reference/stats.html)

#### a. Create a discrete random variable with poissonian distribution and plot its probability mass function (PMF), cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable

#### b. Create a continious random variable with normal distribution and plot its probability mass function (PMF), cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable

#### c. Test if two sets of (independent) random data comes from the same distribution
Hint: Have a look at the ```ttest_ind``` function
