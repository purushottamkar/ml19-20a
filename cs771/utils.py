'''
    Package: cs771
    Module: utils
    Author: Puru
    Institution: CSE, IIT Kanpur
    License: GNU GPL v3.0
    
    A few handy utilities
'''

import numpy as np
import numpy.linalg as lin

# A fast way to compute all pairs distances using Python's broadcasting techniques
def getAllPairsDistances( A, B ):
    squaredNormsA = np.square( lin.norm( A, axis = 1 ) )
    squaredNormsB = np.square( lin.norm( B, axis = 1 ) )
    return squaredNormsA[:, np.newaxis] + squaredNormsB - 2 * A.dot( B.T )