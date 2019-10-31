import numpy as np
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import axes3d

def getMesh( minVal, maxVal, numTicsX, numTicsY ):
    x = np.linspace( minVal, maxVal, numTicsX )
    y = np.linspace( minVal, maxVal, numTicsY )
    return np.meshgrid( x, y )

# Plot the 2D surface given by the parametric form (x, y, f(x,y))
def getSurfaceCoords( f, minVal, maxVal, numBins ):
    xx, yy = getMesh( minVal, maxVal, numBins, numBins )
    return (xx, yy, f( xx, yy ))

def addFuncWeighted( f1, f2, w1, w2, w3 = 0 ):
    return lambda xx, yy: w1 * f1( xx, yy ) + w2 * f2( xx, yy ) + w3
	
def composeFunc( f1, f2 ):
	return lambda xx, yy: f1( f2( xx, yy ) )

def sigmoid( t ):
    return 1 / (1 + np.exp( - t ))

w = np.array( [1, 1] )
wp = np.array( [1, -1] )

minVal = -5
maxVal = 5
numBins = 50

# Python supports functional programming. Lambda notation helps
# us define unnamed functions that can be passed around and returned
# as variables. This helps avoid clutter while writing code

n1 = lambda xx, yy: sigmoid( w[0] * xx + w[1] * yy + 2 )
n2 = lambda xx, yy: sigmoid( w[0] * xx + w[1] * yy - 2 )
n3 = composeFunc( sigmoid, addFuncWeighted( n1, n2, 5, -5, -5 ) )

n4 = lambda xx, yy: sigmoid( wp[0] * xx + wp[1] * yy + 2 )
n5 = lambda xx, yy: sigmoid( wp[0] * xx + wp[1] * yy - 2 )
n6 = composeFunc( sigmoid, addFuncWeighted( n4, n5, 5, -5, -5 ) )

# Increasing the magnitude of the weights and biases will make the peaks get steeper
# Using just a few neurons, we were able to place a Dirac delta at our location of
# choice. This is the key to the universality of neural networks. They can use these
# to memorize data. This also causes NN to be prone to overfitting.
n7 = composeFunc( sigmoid, addFuncWeighted( n3, n6, 10, 10, -5 ) )
# n7 = composeFunc( sigmoid, addFuncWeighted( n3, n6, 50, 50, -20 ) )

fig = plt.figure( figsize = (7, 7) )
ax = fig.gca( projection = '3d' )

# xx, yy, zz = getSurfaceCoords( n1, minVal, maxVal, numBins )
# ax.plot_surface( xx, yy, zz, alpha = 0.1, cmap = cm.gist_rainbow )
# ax.plot_wireframe( xx, yy, zz, rstride = 2, cstride = 2, color = 'k', alpha = 0.1, linewidths = 0.5 )

# xx, yy, zz = getSurfaceCoords( n2, minVal, maxVal, numBins )
# ax.plot_surface( xx, yy, zz, alpha = 0.1, cmap = cm.gist_rainbow )
# ax.plot_wireframe( xx, yy, zz, rstride = 2, cstride = 2, color = 'k', alpha = 0.1, linewidths = 0.5 )

xx, yy, zz = getSurfaceCoords( n7, minVal, maxVal, numBins )
ax.plot_surface( xx, yy, zz, alpha = 0.3, cmap = cm.gist_rainbow )
ax.plot_wireframe( xx, yy, zz, rstride = 2, cstride = 2, color = 'k', alpha = 0.3, linewidths = 0.5 )

ax.set_xlabel( 'X' )
ax.set_xlim( minVal, maxVal )
ax.set_ylabel( 'Y' )
ax.set_ylim( minVal, maxVal )
ax.set_zlabel( 'Z' )

plt.show()