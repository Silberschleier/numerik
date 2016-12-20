import numpy as np
from matplotlib import pyplot
from matplotlib import animation
from time import clock
from ChristmasTrees import *


def GetForceMatrix(MassReciprocal, SpringConstant, SpringIndices):
    """Generates the matrix C as explained on the exercise sheet. It maps an 
       excursion vector to a force vector for the specified point mass model.
    \param MassReciprocal is a column vector containing the reciprocal of the 
           mass for each point mass.
    \param SpringConstant is a single scalar spring constant (i.e. "hardness") 
           for all springs.
    \param SpringIndices has to be an (n,2) integer array where n is the number
           of springs. Each line provides the zero-based indices of two point 
           masses which are connected by a spring. There is one line per 
           spring, so if (i,j) is contained, (j,i) is not. Of course the 
           adjacency matrix still has to be symmetric to enforce Actio=Reactio, 
           the third of Newton's laws of motion."""
    n = MassReciprocal.shape[0]

    # Adjazenzmatrix erzeugen
    A = np.zeros((n, n))
    for i, j in SpringIndices:
        print('i: {}, j: {}'.format(i, j))
        A[i][j] = 1
        A[j][i] = 1

    S = np.zeros((n, n))
    for i in range(0, n):
        S[i][i] = np.sum([SpringConstant * a for a in A[i]])

    M = np.diag(MassReciprocal)

    return M.dot(SpringConstant * A - S)


def GetModeOfVibration(ForceMatrix, iMode, KernelDimension):
    """This function analyzes vibrations of the point mass model with the given 
       force matrix.
    \param iMode is the zero-based index of the mode of vibration. The lowest
           frequency (i.e. largest non-zero Eigenvalue) corresponds to index 
           zero.
    \param KernelDimension is provided for convenience. It is the multiplicity 
           of the Eigenvalue zero, resp. the dimension of the kernel.
    \return A tuple (AngularFrequencyList,AngularFrequency,Offset). 
       AngularFrequencyList provides a complete list of the (non-zero) angular 
       frequencies of the modes of vibration.
       AngularFrequency provides the angular frequency of the mode with the 
       given zero-based index.
       Offset provides the corresponding displacement vector for the point 
       mass coordinates. It is normalized such that the largest magnitude of 
       its entries is 1."""


def ApplyOffset(CurrentTime, Y, AngularFrequency, Offset, MaxExcursion):
    """This function returns the vector of y-coordinates for the given time
       assuming that the point mass model vibrates in the specified mode.
       Y corresponds to the vector of y-coordinates at time zero, CurrentTime 
       corresponds to t, AngularFrequency is sqrt(-lambda(i)), Offset is the 
       Eigenvector, MaxExcursion is L."""


def Update2DVibration(Frame, InitialTime, X, Y, SpringIndices, AngularFrequency, Offset, MaxExcursion, Points, Springs):
    """Updates the animation started by Animate2DVibration() without redoing 
       the entire plot."""
    CurrentTime = clock() - InitialTime
    NewY = ApplyOffset(CurrentTime, Y, AngularFrequency, Offset, MaxExcursion)
    Points.set_data(X, NewY)
    for i, Spring in enumerate(Springs):
        Spring.set_data(X[SpringIndices[i, :]], NewY[SpringIndices[i, :]])


def Animate2DVibration(X, Y, SpringIndices, AngularFrequency, Offset, MaxExcursion):
    """This function animates the given 2D point mass model using the vibration 
       defined by the given offset and angular frequency and scaled by the 
       given maximal excursion.
    \return An animation object. It has to be stored."""
    # Prepare a figure with appropriate title
    FigureName = "Vibration at " + str(AngularFrequency / (2.0 * np.pi)) + "hz"
    Figure = pyplot.figure("Name")
    Figure.suptitle(FigureName)
    Axes = pyplot.axes(aspect=1.0)
    # Do an initial plot
    InitialTime = clock()
    Springs = pyplot.plot(X[SpringIndices.T], Y[SpringIndices.T], "-g", linewidth=5.0)
    Points = pyplot.plot(X, Y, "or", markersize=12.0)[0]
    # Run an animation
    return animation.FuncAnimation(Figure, Update2DVibration, 100000, fargs=(
    InitialTime, X, Y, SpringIndices, AngularFrequency, Offset, MaxExcursion, Points, Springs), repeat=False,
                                   interval=60)


if __name__ == "__main__":
    # Choose a mode
    iMode = 11
    # Get the model
    X, Y, SpringIndices, MassReciprocal = GetChristmasTree()
    # Construct the force matrix
    ForceMatrix = GetForceMatrix(MassReciprocal, 1.0, SpringIndices)
    # Compute the modes of vibration
    AngularFrequencyList, AngularFrequency, Offset = GetModeOfVibration(ForceMatrix, iMode,
                                                                        np.sum(MassReciprocal == 0))
    print("Modes of vibration have the following frequencies:")
    print(AngularFrequencyList / (2.0 * np.pi))
    # Display an animation
    Animation = Animate2DVibration(X, Y, SpringIndices, AngularFrequency, Offset, 0.2)
    pyplot.show()
