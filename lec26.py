import numpy as np
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import axes3d

def getMesh( minVal, maxVal, numTicsX, numTicsY ):
    x = np.linspace( minVal, maxVal, numTicsX )
    y = np.linspace( minVal, maxVal, numTicsY )
    return np.meshgrid( x, y )

# Given a set of 2D points, map them onto the 2D surface
# given by the parametric form (x, y^2, x^2)
def applyMap( xx, yy ):
	return xx, np.square( yy ), np.square( xx )

# Plot the 2D surface given by the parametric form (x, y^2, x^2)
def getSurfaceCoords( minVal, maxVal, numBins ):
    xx, yy = getMesh( minVal, maxVal, numBins, numBins )
    return applyMap( xx, yy )

# Plot the 2D hyperplane given by the parametric form (x, y, wx * x + wy * y)
def getHyperplaneCoords( minVal, maxVal, numBins, wx, wy ):
    xx, yy = getMesh( minVal, maxVal, numBins, numBins )
    zz = wx * xx + wy * yy
    return xx, yy, zz

fig = plt.figure( figsize = (7, 7) )
ax = fig.gca( projection = '3d' )

minVal = -3
maxVal = 3
numBins = 50

xx, yy, zz = getSurfaceCoords( minVal, maxVal, numBins )

ax.plot_surface( xx, yy, zz, alpha = 0.1, cmap = cm.gray_r )
ax.plot_wireframe( xx, yy, zz, rstride = 2, cstride = 2, color = 'k', alpha = 0.1, linewidths = 0.5 )

ax.set_xlabel( 'X' )
ax.set_xlim( minVal, maxVal )
ax.set_ylabel( 'Y' )
ax.set_ylim( minVal, maxVal )
ax.set_zlabel( 'Z' )

# The red points
xp = np.array( [-0.5,2.5,1,1] )
yp = np.array( [0,0,1.5,-1.5] )

# The green points
xn = np.array( [0.5,0.5,1.5,1.5] )
yn = np.array( [0.5,-0.5,0.5,-0.5] )

# Plot the original points a bit lower for sake of clarity
z = -8 * np.ones( (4,) )

# Plot the original points
ax.scatter( xp, yp, z, marker = 'o', s = 50, color = "red", alpha = 0.3 )
ax.scatter( xn, yn, z, marker = 'o', s = 50, color = "green", alpha = 0.3 )

# Plot the mapped points
xx, yy, zz =  applyMap( xp, yp )
ax.scatter( xx, yy, zz, marker = 'o', s = 50, color = "red" )
xx, yy, zz =  applyMap( xn, yn )
ax.scatter( xx, yy, zz, marker = 'o', s = 50, color = "green" )

# The mapped points are linearly separable even though the original points were not so 
# Plot a such a separating hyperplane for the mapped points
xx, yy, zz = getHyperplaneCoords( minVal, maxVal, numBins, 2, -1 )
ax.plot_surface( xx, yy, zz, cmap = cm.gist_heat, alpha = 0.2 )

plt.show()