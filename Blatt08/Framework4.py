import numpy as np
from matplotlib import pyplot
from matplotlib import animation
from mpl_toolkits import mplot3d
from scipy import sparse
import scipy.sparse.linalg 
from time import clock
from ChristmasTrees import *

def GetSparseForceMatrix(MassReciprocal,SpringConstant,SpringIndices):
    """Generates the sparse matrix C as explained on the exercise sheet. It 
       maps an excursion vector to a force vector for the specified point mass 
       model.
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
    

	
def GetSparseModeOfVibration(ForceMatrix,iMode):
    """This function analyzes vibrations of the point mass model with the given 
       force matrix.
    \param iMode is the zero-based index of the mode of vibration. The lowest
           frequency (i.e. largest non-zero Eigenvalue) corresponds to index 
           zero.
    \return A tuple (AngularFrequency,Offset). 
       AngularFrequency provides the angular frequency of the mode with the 
       given zero-based index.
       Offset provides the corresponding displacement vector for the point 
       mass coordinates. It is normalized such that the largest magnitude of 
       its entries is 1."""
    

	
def ApplyOffset(CurrentTime,Y,AngularFrequency,Offset,MaxExcursion):
    """This function returns the vector of y-coordinates for the given time
       assuming that the point mass model vibrates in the specified mode.
       Y corresponds to the vector of y-coordinates at time zero, CurrentTime 
       corresponds to t, AngularFrequency is sqrt(-lambda(i)), Offset is the 
       Eigenvector, MaxExcursion is L."""
    

	
def Update3DVibration(Frame,InitialTime,X,Y,Z,TriangleIndices,AngularFrequency,Offset,MaxExcursion,Mesh):
    """Updates the animation started by Animate3DVibration() without redoing 
       the entire plot."""
    CurrentTime=clock()-InitialTime;
    NewY=ApplyOffset(CurrentTime,Y,AngularFrequency,Offset,MaxExcursion);
    Vertices=np.dstack([X[TriangleIndices],Z[TriangleIndices],NewY[TriangleIndices]]);
    Mesh.set_verts(Vertices);


def Animate3DVibration(X,Y,Z,TriangleIndices,AngularFrequency,Offset,MaxExcursion):
    """This function animates the given 3D point mass model using the vibration 
       defined by the given offset and angular frequency and scaled by the 
       given maximal excursion.
    \return An animation object. It has to be stored."""
    # Prepare a figure with appropriate title
    FigureName="Vibration at "+str(AngularFrequency/(2.0*np.pi))+"hz";
    Figure=pyplot.figure("Name");
    Figure.suptitle(FigureName);
    Axes=mplot3d.axes3d.Axes3D(Figure,aspect=1.0);
    # Do an initial plot
    InitialTime=clock();
    Mesh=Axes.plot_trisurf(X,Z,Y,triangles=TriangleIndices,color="g");
    # Run an animation
    return animation.FuncAnimation(Figure,Update3DVibration,100000,fargs=(InitialTime,X,Y,Z,TriangleIndices,AngularFrequency,Offset,MaxExcursion,Mesh),repeat=False,interval=60);


if(__name__=="__main__"):
    # Choose a mode
    iMode=3;
    # Get the model
    X,Y,Z,SpringIndices,TriangleIndices,MassReciprocal=Get3DChristmasTree();
    # Construct the force matrix
    ForceMatrix=GetSparseForceMatrix(MassReciprocal,1.0,SpringIndices);
    # Compute a single mode of vibration
    print("Solving the Eigenproblem...");
    AngularFrequency,Offset=GetSparseModeOfVibration(ForceMatrix,iMode);
    # Display an animation
    print("Showing the animation...");
    Animation=Animate3DVibration(X,Y,Z,TriangleIndices,AngularFrequency,Offset,0.3);
    pyplot.show();
