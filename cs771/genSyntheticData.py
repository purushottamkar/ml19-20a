'''
    Package: cs771
    Module: helloWorld
    Author: Puru
    Institution: CSE, IIT Kanpur
    License: GNU GPL v3.0
    
    Generate synthetic data of various sorts
'''

import numpy as np
import numpy.linalg as lin
import numpy.random as rnd

# Return n data points (as an n x d array) sampled from N(mu, sigma^2 . I)
def genSphericalNormalData( d, n, mu, sigma ):
    X = rnd.normal( 0, sigma, (n, d) ) + mu
    return X

# Return n data points (as an n x d array) sampled from N(mu, cov)
def genNormalData( d, n, mu, cov ):
    X = rnd.multivariate_normal( mu, cov, n )
    return X

# Return n data points (as an n x d array) sampled from the surface of sphere of radius r centered at mu
def genSphericalData( d, n, mu, r ):
    X = rnd.normal( 0, 1, (n, d) )
    norms = lin.norm( X, axis = 1 )
    X = X / norms[:, np.newaxis]
    X = (X * r) + mu
    return X

# Return n data points (as an n x d array) sampled from the surface of ellipse of covariance cov centered at mu
def genEllipticallData( d, n, mu, cov ):
    X = genSphericalData( d, n, np.zeros((d,)), 1 )
    L = lin.cholesky( cov )
    X = np.matmul( X, L ) + mu
    return X