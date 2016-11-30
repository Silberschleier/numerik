import numpy as np
from numpy import linalg as la

# Funktion für vk * vk^t, da transpose und .T bei eindimensionalen Array nicht Funktioniert haben
# Tipp mit vprod aus dem Internet: http://stackoverflow.com/questions/19238024/transpose-of-a-vector-using-numpy
def vprod(x):
    y = np.atleast_2d(x)
    return np.dot(y.T, y)

def Householder(A):
    """Computes a QR-decomposition of the given matrix using the Householder
           algorithm.
          \return A pair (NormalList,R) where NormalList is a list of Householder
                  normal vectors and R is an upper triangular matrix shaped like A.
          \sa ComputeQ """
    normal_list = [];
    m, n = A.shape
    for k in range(0, n):
        x = np.copy(A[k:, k]);
        vk = x;
        vk[0] = np.sign(x.item(0)) * la.norm(x, 2) + x.item(0);
        print(vk)
        vk = vk/( la.norm(vk,2) );
        normal_list.append(vk)
        A[k:, k:] = A[k:, k:] -2*vprod(vk).dot(A[k:, k:]);
    return normal_list, A;

def ComputeQ(NormalList):
    """Given a normal list such as the one returned by householder() this
       function computes the corresponding orthogonal matrix."""
    Q = np.eye(NormalList[0].__len__()) - 2*vprod(NormalList[0])
    print("Q: \n", Q, "\n")
    for k in range(1,NormalList.__len__()):
        H = np.eye(NormalList[k].__len__()) - 2 * vprod(NormalList[k])
        Q_m_minus_k = np.eye(NormalList[0].__len__())
        Q_m_minus_k[k:,k:] = H
        #print(Q_m_minus_k)
        Q = Q.dot(Q_m_minus_k)

    return Q;

if __name__ == "__main__":
    A = np.random.rand(4, 3);
    print("A: \n ",A,"\n")
    NormalList, R = Householder(A)
    print("R: \n", R, "\n")
    Q = ComputeQ(NormalList)
    print(Q.dot(R))
