import numpy as np
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import axes3d

def getMesh( minVal, maxVal, numTicsX, numTicsY ):
    x = np.linspace( minVal, maxVal, numTicsX )
    y = np.linspace( minVal, maxVal, numTicsY )
    return np.meshgrid( x, y )

def getCoords( minVal, maxVal, numBins ):
    xx, yy = getMesh( minVal, maxVal, numBins, numBins )
    zz = np.sin( xx ) + np.sin( yy )
    return xx, yy, zz

fig = plt.figure( figsize = (7, 7) )
ax = fig.gca( projection = '3d' )

minVal = -6
maxVal = 6
numBins = 50

xx, yy, zz = getCoords( minVal, maxVal, numBins )

ax.plot_surface( xx, yy, zz, alpha = 0.5, cmap = cm.gist_rainbow )
ax.plot_wireframe( xx, yy, zz, rstride = 2, cstride = 2, color = 'k', linewidths = 0.5 )

ax.set_xlabel( 'X' )
ax.set_xlim( minVal, maxVal )
ax.set_ylabel( 'Y' )
ax.set_ylim( minVal, maxVal )
ax.set_zlabel( 'Z' )

# cset = ax.contour( xx, yy, zz, zdir = 'z' )

plt.show()
