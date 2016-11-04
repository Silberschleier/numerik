import numpy as np


def LinearSolve(A, b):
    pinv = PseudoInverse(A)
    return pinv.dot(b)


def PseudoInverse(A):
    u, s, v = np.linalg.svd(A)
    #u, s, v = SingularValueDecomposition(A)

    # Kehrwerte fuer alle Elemente != 0
    s = [1./x if x != 0 else 0 for x in s]
    s = np.diag(s)

    # V*S^+*U^T
    return np.dot(v, s, u)


def SingularValueDecomposition(A):
    # Berechne Eigenwerte und -vektoren von A^T*A
    values, vectors = np.linalg.eig(A.T.dot(A))

    sigma = [np.sqrt(x) for x in values]

    u = []
    for i, v in enumerate(vectors):
        u.append(A.dot(v)/np.sqrt(values[i]))
    u = np.array(u)

    return u, sigma, vectors



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

