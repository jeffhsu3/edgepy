""" Fitting functions that use numba for optimization.
"""

from numba import double, jit, int_, void 
from numba.decorators import autojit
from numpy.linalg import lstsq
import numpy as np

@jit
class GLM_Levenberg(object):
    """ At the moment the design array needs to be an int
    """

    @void(int_, int_, double[:], int_)
    def __init__(self, nlibs, ncoefs, design, maxit):
        # Hmmm the cpp is treating design as one-dimensions
        self.nlibs = nlibs
        self.ncoefs = ncoefs
        self.maxit = maxit
        self.len_ = nlibs * ncoefs
        self.design = design
        self.d = np.zeros(self.len_, dtype=np.double)
        #self.wx = np.zeros(len_, dtype=np.double)
        self.xwx = np.zeros(ncoefs*ncoefs, dtype=np.double)
        self.xwx_copy = np.zeros(ncoefs*ncoefs, dtype=np.double)
        self.dl = np.zeros(ncoefs, dtype=np.double)
        self.dbeta = np.zeros(ncoefs, dtype=np.double)
        self.mu_new = np.zeros(nlibs, dtype=np.double)
        self.beta_new = np.zeros(ncoefs, dtype=np.double)


    '''
    @void(double[:], double[:], double[:], double[:], double[:])
    def fit(self, offset, y, disp, mu, beta):
        """Fit the nbinom glm. 
        """
        low_value = 10e-6
        ymax = np.max(y)
        if ymax < low_value:
            mu = np.zeros(self.nlibs)
            for coef in range(self.ncoefs): beta[coef] = np.nan
        wx = np.zeros(self.len_, dtype=np.double)

        dev = self.nb_deviance(y, mu, disp)
        max_info = -1
        lambda_ = 0
        for iter_ in range(self.maxit):
            # Reset DL
            for i in range(self.ncoefs): self.dl[i]=0
            for row in range(self.nlibs):
                cur_mu = mu[row]
                denom = 1+cur_mu * disp
                weight = cur_mu/denom 
                deriv = (y[row] - cur_mu)/denom
                for col in xrange(self.ncoefs):
                    # cur_idx = the current index of flattened design array
                    cur_idx = col*self.nlibs + row
                    self.wx[cur_idx] = self.design[cur_idx] * weight
                    self.dl[col] = self.design[cur_idx] * deriv
                xwx = dgemm(alpha=1.0, a=self.design, b=wx)

            for i in range(self.ncoefs):
                cur_val = xwx[i * self.ncoefs + i]
                #if cur_val > max_info: max_info = cur_val
            

            if iter == 1:
                #lambda_ = max_info * 1e-6
                #if lambda_ < 1e-13: lambda_ = 1e-13
                pass

            lev = 0
            low_dev = False

            # Create an iterator

            while True:
                lev += 1
                # Copy dl to dlbeta
                break

    '''

    @double(double[:], double[:], double[:])
    def nb_deviance(self, y, mu, phi):
        """ Calculate negative binomial deviance.
        """
        low_value = np.power(10, -8)
        one_millionth = np.power(10, -6)
        one_million = np.power(10, 6)
        dev = 0.0
        for i in xrange(self.nlibs):
            cur_y = y[i] if y[i] > low_value else low_value
            cur_mu = mu[i] if mu[i] > low_value else low_value
            product = cur_mu * cur_y
            # Poisson distribution for small values, gamma for very large
            # values and nbinom for everything else
            if product < one_millionth:
                dev += cur_y * np.log(cur_y/cur_mu) - (cur_y - cur_mu)
            elif product > one_million:
                dev += (cur_y - cur_mu)/cur_mu - np.log(cur_y/mu[i])
            else:
                dev += cur_y * np.log(cur_y/cur_mu) + (cur_y + 1/phi[i]) *\
                        np.log((cur_mu+1/phi[i])/(cur_y+1/phi[i]))
        return dev * 2


@autojit()    
def mglm_Levenberg_numba(counts, dispersion, offset, beta, 
        fitted, tol, maxit):
    """Numba implementation of Levenberg algorithm
    """
    ntags = counts.shape[0]
    nsamples = counts.shape[1]
    out = np.zeros((ntags,nsamples))
    for tag in range(ntags):
        pass


@autojit()
def mlstsqs(design_matrix, my):
    """ Fits multiple ordinary linear models to the same endogenous variable.

    Parameters
    ---------
    design_matrix: the design matrix
    my: a matrix containing 

    Returns 
    A matrix containing only the coefficients.
    """
    ny = my.shape[0]
    out = np.zeros((my.shape[0], design_matrix.shape[1]), dtype=np.double)

    for i in range(ny):
        
        out[i, :] = lstsq(design_matrix, my[i,:])[0]

    return out
