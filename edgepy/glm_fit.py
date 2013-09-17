"""
This module implements fitting multiple generalized linear models.

Robinson M, McCarthy DJ, Smyth GK. edgeR: a Bioconductor package for
differential expression anaylsis of digital expression data. Bioinformatics 26,
139-140. 2010

Jeffrey Hsu
"""

from numpy import asarray, exp
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from numba import double, jit, int_, void 
from numba.decorators import autojit
#from scipy.linalg.fblas import dgemm
from numpy.linalg import lstsq
from numpy.linalg import matrix_rank
from statsmodels.genmod.families.links import log


def glm_fit(y, design, dispersion, offset=None, weights=None, lib_size=None,
        method='auto'):
    """ A general glm fit function.

    Parameters
    ----------
    y : dataframe of count matrix
    design: dataframe of the experimental design
    offset: offset values with coefficient constrained to 1 that is added 
    weights: weights
    lib_size: 
    method: {'auto', 'levenburg', 'simple'}


    Returns
    -------

    See Also
    --------
    mglm_simple, mglm_levenberg
    """
    rank = matrix_rank(design)
    isna = any(np.isnan(y))

    if rank < min(design.shape):
        # :TODO raise error
        print(rank)
        print(design.shape)

    #out = mglm_Levenberg(y.as_matrix(), design, dispersion, offeset=offset)
    if method == 'auto':
        if isna:
            method == 'simple'
        else:
            pass
    elif method == 'levenberg':
        pass
    else:
        pass

    fit = mglm_simple(y.as_matrix(), design, dispersion, offset=offset,
            weights=None)


    return fit


def mglm_simple(y, design, dispersion=0, offset=0, weights=None):
    """ 
    Fit negative binomial glm with calls to statsmodels glm.fit().

    Parameters
    ----------
    y: digital count matrix
    design: design dataframe or matrix
    dispersion: dispersion estimate
    offset: offset array 
    weights: weights array

    Returns
    -------

    Notes
    -----
    """
    ngenes = y.shape[0]
    # Initialize the output
    coefficients = np.zeros((ngenes, design.shape[1]), dtype=np.double)
    fitted_values = np.zeros((ngenes, y.shape[1]), dtype=np.double)
    df_residual = np.zeros(ngenes, dtype=np.double)
    dev = np.zeros(ngenes, dtype=np.double)
    error = np.repeat(False, ngenes)
    converged = np.repeat(False, ngenes)
    pvalues = np.zeros((ngenes, design.shape[1]), dtype=np.double)

    # Create masks for weights 
    if getattr(weights, 'size', None):
        weights[weights <= 0] = np.nan 
        weights = expand_as_matrix(weights, y.shape)
    else:
        weights = np.zeros(y.shape, dtype=np.double)
        weights[:] = 1

    offset = expand_as_matrix(offset, y.shape)
        

    # Setting the glm family
    if len(dispersion) > 1:
        common_family = False
    else:
        common_family = True
        if dispersion > 1e-10:
            f = sm.families.NegativeBinomial(link=log, alpha=dispersion)
        else:
            f = sm.families.Poisson(link=log)

    for i in xrange(ngenes):
        if not common_family:
            if dispersion[i] > 1e-10:
                f = sm.families.NegativeBinomial(link=log,
                        alpha=dispersion[i])
            else:
                f = sm.families.Poisson(link=log)

        z = np.asarray(y[i,:])
        obs = np.isfinite(z)
        if sum(obs) > 0:
            X = design.ix[obs,:]
            z = z[obs]
            w = weights[i, obs]
            output = sm.GLM(z, X, family=f, offset=np.log(offset[i, obs])).fit()
            coefficients[i, :] = output.params
            fitted_values[i, :] = output.fittedvalues
            df_residual[i] = output.df_resid
            dev[i] = output.deviance
            pvalues[i, :] = output.pvalues
        else: pass


    return(coefficients, fitted_values, df_residual, dev, pvalues)





def mglm_Levenberg(y, design, dispersion=0, offset=0, coef_start=None,
        start_method='null'):
    """ Fit genewise negative binomial glms with log-link using Levenberg
    dampening for convergence.  

    Parameters
    ----------
    y : matrix 
    design : dataframe

    Adapted from Gordon Smyth's and Yunshun Chen's algorithm in R.
    """
    design = add_constant(design)
    if not coef_start:
        start_method = [i for i in ['null', 'y'] if i == start_method][0]
        if start_method == 'null': N = exp(offset)
    else: 
        coef_start = asarray(coef_start)

    if not coef_start:
        if start_method == 'y':
            delta = np.min(np.max(y), 1/6)
            y1 = np.maximum(y, delta)
            #Need to find something similiar to pmax
            fit = lstsq(design, np.log(y1 - offset)).fit()
            beta = fit.params
            mu = np.exp(beta + offset)
        else:
            beta_mean = np.log(np.average(y,axis=1, weights=offset))
    else:
        beta = coef_start.T

    pass


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


def mismatch_addition(array, b):
    """ Add an array to a matrix row-wise or column wise dep
    """
    raise NotImplemented


def expand_as_matrix(x, dim):
    """ Expands an array to a matrix matching the matrix dimension.
    """
    x_shape = getattr(x, "shape", None)
    if len(x_shape) == 1 and x_shape:
        if x_shape[0] == dim[0]:
            return np.repeat(np.asarray(x), dim[1]).reshape((x.shape[0], dim[1]))
        if x_shape[0] == dim[1]:
            return np.repeat(np.asarray(x), dim[0]).reshape((x.shape[0], dim[0])).T
    elif len(x_shape) == 2:
        pass
    else: 
        return np.empty(dim).fill(x)
        




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
        dev = 0
        for i in xrange(self.nlibs):
            cur_y = y[i] if y[i] > low_value else low_value
            cur_mu = mu[i] if mu[i] > low_value else low_value
            product = cur_mu * cur_y
            # Poisson distribution for small values, gamma for very large
            # values and nbinom for everything else
            if product < one_millionth:
                dev += cur_y * np.log(cur_y/cur_mu) - (cur_y - cur_mu)
            elif product > one_million:
                dev += (cur_y - cur_mu)/cur_mu - np.log(cur_y/mu)
            else:
                dev += cur_y * np.log(cur_y/cur_mu) + (cur_y + 1/phi) *\
                        np.log((cur_mu+1/phi)/(cur_y+1/phi))
        return dev * 2



def glm_lrt(glm_fit, coef, contrast=None, test='chisq'):
    """ Liklehood Ratio Test

    Parameters
    ----------

    Returns
    -------
    """
    pass
