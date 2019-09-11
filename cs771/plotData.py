'''
    Package: cs771
    Module: plotData
    Author: Puru
    Institution: CSE, IIT Kanpur
    License: GNU GPL v3.0
    
    Plot 2D data in various ways to visualize classifiers and other things
'''

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
import numpy as np

# A light red and light green binary colormap
binaryColors = [ (1, 0.85, 0.85), (0.85, 1, 0.85) ]
nBins = 2
lrlg = lsc.from_list( 'lrlg', binaryColors, nBins )

# A light red and light green binary colormap with shades
probColors = [ (1, 0.75, 0.75), (1, 1, 1), (0.75, 1, 0.75) ]
nBinsShade = 200
lrlgShade = lsc.from_list( 'lrlg', probColors, nBinsShade )

# A more gradual transition with more room for white shades
probColorsGradual = [ (1, 0.75, 0.75), (1, 1, 1), (1, 1, 1), (0.75, 1, 0.75) ]
nBinsShade = 200
lrlgShadeGradual = lsc.from_list( 'lrlg', probColorsGradual, nBinsShade )

def getFigure( sizex = 7, sizey = 7 ):
    fig = plt.figure( figsize = (sizex, sizey) )
    return fig

def getFigList( nrows = 1, ncols = 1, sizex = 3, sizey = 3 ):
    fig, axs = plt.subplots( nrows = nrows, ncols = ncols, figsize = (sizex * ncols, sizey * nrows), squeeze = 0 )
    axs = axs.reshape( -1 )
    return (fig, axs)

def showImagesNoAxes( axes, imageList, numRows, numCols, resize = False, imShape = [], cmap = plt.cm.gray_r, labelList = [] ):
    for i in range( numRows * numCols ):
        currAxis = axes[i]
        im = imageList[i]
        if resize:
            im = im.reshape( imShape )
        currAxis.imshow( im, cmap = cmap, interpolation = 'nearest' )
        currAxis.tick_params( axis = 'x', which = "both", bottom = False, labelbottom = False )
        currAxis.tick_params( axis = 'y', which = "both", left = False, labelleft = False )
        if labelList:
            currAxis.set_title( labelList[i] )

def plotCurve( responseGenerator, fig, mode = "point", color = 'b', linestyle = "-", xlimL = 0, xlimR = 10, nBins = 500, label = "" ):
    X = np.linspace( xlimL, xlimR, nBins, endpoint = True )
    if mode == "point":
	    y = np.zeros( X.shape )
	    for i in range( X.size ):
		    y[i] = responseGenerator( X[i] )
    elif mode == "batch":
	    y = responseGenerator( X )
    plt.figure( fig.number )
    plt.plot( X, y, color = color, linestyle = linestyle, label = label )
    if label:
        plt.legend()

def plot2D( X, fig, color = 'r', marker = '+', size = 100 ):
    plt.figure( fig.number )
    plt.scatter( X[:,0], X[:,1], s = size, c = color, marker = marker )

def plot2DPoint( X, fig, color = 'r', marker = '+', size = 100 ):
    plt.figure( fig.number )
    plt.scatter( X[0], X[1], s = size, c = color, marker = marker )
	
def plotLine( w, b, fig, color = 'k', linestyle = "-", xlimL = -10, xlimR = 10, nBins = 500, label = "" ):
    plt.figure( fig.number )
    if np.abs( w[1] ) < 1e-6:
        y = np.linspace( xlimL, xlimR, nBins )
        x = -b/w[0]*np.ones( y.shape )
    else:
        x = np.linspace( xlimL, xlimR, nBins )
        y = (-w[0] * x - b)/w[1]
    plt.plot( x, y, color = color, linestyle = linestyle, label = label )
    if label:
        plt.legend()
		
def plotVerticalLine( x, fig, color = 'k', linestyle = '-', yLimB = -10, yLimT = 10, nBins = 500, label = ""):
    plt.figure( fig.number )
    y = np.linspace( yLimB, yLimT, nBins )
    plt.plot( x*np.ones( y.shape ), y, color = color, linestyle = linestyle, label = label )
    if label:
        plt.legend()
	
def shade2D( labelGenerator, fig, mode = "point", colorMap = lrlg, xlim = 10, ylim = 10, nBins = 500 ):
    xi, yi = np.mgrid[ -xlim:xlim:nBins*1j, -ylim:ylim:nBins*1j ]
    zi = np.zeros( xi.shape )
    if mode == "point":
        for i in range( zi.shape[0] ):
            for j in range( zi.shape[1] ):
                zi[i,j] = labelGenerator( xi[i,j], yi[i,j] )
    elif mode == "batch":
        zi = labelGenerator( np.hstack( (xi.reshape((xi.size, 1)), yi.reshape((yi.size, 1))) ) )
        zi = zi.reshape( xi.shape )
    zi[zi < 1/nBins] = 0
    zi[zi > 0] = 1
    plt.figure( fig.number )
    plt.pcolormesh( xi, yi, zi, cmap = colorMap )

def shade2DProb( scoreGenerator, fig, mode = "point", colorMap = lrlgShadeGradual, xlim = 10, ylim = 10, nBins = 500 ):
    xi, yi = np.mgrid[ -xlim:xlim:nBins*1j, -ylim:ylim:nBins*1j ]
    zi = np.zeros( xi.shape )
    if mode == "point":
        for i in range( zi.shape[0] ):
            for j in range( zi.shape[1] ):
                zi[i,j] = scoreGenerator( xi[i,j], yi[i,j] )
    elif mode == "batch":
        zi = scoreGenerator( np.hstack( (xi.reshape((xi.size, 1)), yi.reshape((yi.size, 1))) ) )
        zi = zi.reshape( xi.shape )
    plt.figure( fig.number )
    plt.pcolormesh( xi, yi, zi, cmap = colorMap )
