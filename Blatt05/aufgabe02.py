import numpy as np


def LU(A):
    U = np.copy(A)
    n, _ = A.shape
    L = np.eye(n, n)

    for k in range(0, n - 1):
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]

    return U, L


def LUP(A):
    U = np.copy(A)
    n, _ = A.shape
    L = np.eye(n, n)
    P = np.eye(n, n)

    for k in range(0, n - 1):
        max_index = k
        for i in range(k, n):
            if abs(U[i][k]) > abs(U[max_index][k]):
                max_index = i

        U[k], U[max_index] = U[max_index], U[k]
        L[k], L[max_index] = L[max_index], L[k]
        P[k], P[max_index] = P[max_index], P[k]

        for j in range(k + 1, n):
            if U[k][k] != 0:
                L[j][k] = U[j][k] / U[k][k]
            U[j, k:] = U[j, k:] - L[j, k] * U[k, k:]

    return U, L, P


def BackSubstitution(U, y):
    return np.linalg.solve(U, y)


def ForwardSubstitution(L, b):
    return np.linalg.solve(L, b)


def SolveLinearSystemLU(A, b):
    U, L = LU(A)
    y = ForwardSubstitution(L, b)
    return BackSubstitution(U, y)


def SolveLinearSystemLUP(A, b):
    U, L, P = LUP(A)
    y = ForwardSubstitution(L, P.dot(b))
    return BackSubstitution(U, y)

if __name__ == "__main__":
    A = np.random.rand(4, 4)
    #A = np.array([[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]])
    U, L, P = LUP(A)

    print("A:")
    print(A)

    print("\nL:")
    print(L)

    print("\nU:")
    print(U)

    print("\nP:")
    print(P)

    print("\nL*U:")
    print(L.dot(U))

    print("\nP*A")
    print(P.dot(A))

    A = np.array([
        [3, 2, -1],
        [2, -2, 4],
        [-1, 0.5, -1]
    ])
    b = np.array([1, -2, 0])

    print("Erwartetes Ergebnis: 1, -2 , -2")
    print(SolveLinearSystemLUP(A, b))
