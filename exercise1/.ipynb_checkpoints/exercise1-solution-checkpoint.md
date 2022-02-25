# Exercises for Day 5
#Using SciPy, Scikit-Learn and Pandas

## 1. Scipy

### Linear Algebra
#Have a look at the ```scipy.linalg``` [module](https://docs.scipy.org/doc/scipy/reference/linalg.html)

import scipy.linalg as linalg
import numpy as np

#### a. Define a matrix A

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]] )
print("A:\n", A)

print("This matrix is not invertible, which means that linear systems below do not always have solutions for this A. For this reason I now define a new A that is invertible:")

A = np.array([[4, -1, 0], [2, 3, 5], [-1, 7, 2]] )
print("A:\n", A)

#### b. Define a vector b
print()


b = np.array([1, 2, 3])
print("b:\n", b)

#### c. Solve the linear system of equations A x = b
print()


x = linalg.solve(A,b)
print("Solution to A x = b: \n", x)

#### d. Check that your solution is correct by plugging it into the equation
print()


tol = 1E-9
print("Define tolerance: ", tol)
print("Checking solution:")
print((np.abs(A.dot(x) - b) < tol).all)

#### e. Repeat steps a-d using a random 3x3 matrix B (instead of the vector b)
print()


B = np.random.rand(3,3)
print("Define B: \n", B)

X = linalg.solve(A,B)
print("Solution to A X = B: \n", X)

print("Checking solution:")
print((np.abs(A.dot(X) - B) < tol).all)

#### f. Solve the eigenvalue problem for the matrix A and print the eigenvalues and eigenvectors
print()


print("Solving eigenvalue problem for A: ")
eigvals, eigvecs = linalg.eig(A)
print("Eigenvalues: \n", eigvals)
print("Eigenvectors: \n", eigvecs)

#### g. Calculate the inverse, determinant of A
print()


invA = linalg.inv(A)
print("Inverse of A, A^-1: \n", invA)

detA = linalg.det(A)
print("Determinant of A, det(A): ", detA)

#### h. Calculate the norm of A with different orders
print()


norms = dict(Frobenius='fro', Nuclear='nuc', L1=1, L_minus1=-1, Inf=np.inf)

print("Norms of A: ")
for name,norm in norms.items():
    print(name, ": ", linalg.norm(A, norm))

### Statistics
Have a look at the ```scipy.stats``` [module](https://docs.scipy.org/doc/scipy/reference/stats.html)

#### a. Create a discrete random variable with poissonian distribution and plot its probability mass function (PMF), cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable

#### b. Create a continious random variable with normal distribution and plot its probability mass function (PMF), cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable

#### c. Test if two sets of (independent) random data comes from the same distribution
Hint: Have a look at the ```ttest_ind``` function
