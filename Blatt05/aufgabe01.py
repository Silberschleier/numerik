import numpy as np
from numpy import linalg as la


def Householder(A):
    """Computes a QR-decomposition of the given matrix using the Householder 
       algorithm.
      \return A pair (NormalList,R) where NormalList is a list of Householder 
              normal vectors and R is an upper triangular matrix shaped like A.
      \sa ComputeQ """

    m, n = A.shape
    normal_list = []

    for k in range(0, n):
        x = A[k:, k]

        x[0] += np.sign(x.item(0)) * la.norm(x, 2)
        vk = np.zeros(m)
        vk[k:m] = x
        normal_list.append(vk)

        A[k:, k] = A[k:, k] - (2*vk)*(vk.T.dot(A))
    return normal_list, A


def ComputeQ(NormalList):
    """Given a normal list such as the one returned by householder() this 
       function computes the corresponding orthogonal matrix."""


if __name__ == "__main__":
    A = np.random.rand(4, 3)
    NormalList, R = Householder(A)
    # Q = ComputeQ(NormalList)
    print("The following matrix should be upper triangular:")
    print(R)
    # print("If the solution consitutes a decomposition, the following is near zero:")
    # print(la.norm(A - np.dot(Q, R)))
    # print("If Q is unitary, the following is near zero:")
    # print(la.norm(np.dot(Q.T, Q) - np.eye(Q.shape[0])))
