import numpy as np


def LU(A):
    U = np.copy(A)
    n, _ = A.shape
    L = np.eye(n, n)

    for k in range(0, n-1):
        for i in range(k+1, n):
            L[i][k] = U[i][k] / U[k][k]
            U[i,k:] = U[i,k:] - L[i,k]*U[k,k:]

    return U, L


if __name__ == "__main__":
    # A = np.random.rand(3, 3)
    A = np.array([ [7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6] ])
    U, L = LU(A)

    print("A:")
    print(A)

    print("\nL:")
    print(L)

    print("\nU:")
    print(U)

    print("\nL*U:")
    print(L.dot(U))