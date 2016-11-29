import numpy as np


def LinearSolve(A, b):
    pinv = PseudoInverse(A)
    return pinv.dot(b)


def PseudoInverse(A):
    u, s, v = SingularValueDecomposition(A)
    s = np.linalg.inv(s)

    return v.dot(s.dot(u.T))


def SingularValueDecomposition(A):
    # Berechne Eigenwerte und -vektoren von A^T*A
    values, vectors = np.linalg.eig(A.T.dot(A))

    sigma = [np.sqrt(x) for x in values]
    s = np.diag(sigma)

    u = []
    for i, v in enumerate(vectors.T):
        u.append(A.dot(v)/np.sqrt(values[i]))
    u = np.array(u)
    u = u.T

    return u, s, vectors



# Beispiel
if __name__ == "__main__":

    A = np.array([
        [3, 2, -1],
        [2, -2, 4],
        [-1, 0.5, -1]
    ])
    b = np.array([1, -2, 0])

    # Erwartetes Ergebnis: 1, -2 , -2
    print(LinearSolve(A, b))
