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
from numpy.linalg import qr
from statsmodels.genmod.families.links import log
from .glm_fit_numba import *


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
    mglm_simple, mglm_levenberg, mrlm
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


def mrlm(M, design, ndups=1):
    """ Robustly fit a linear model for each gene to a series of data.

    Parameters:
    ----------
    M : Observation matrix with different tests as rows
    design : design dataframe
    ndups : number of duplicate tests

    Returns:
    -------

    Citation
    """

    M = M.as_matrix()
    n_tests = M.shape[0]
    n_samples = M.shape[1]
    n_col = design.shape[1]

    stdev_unscaled = np.empty((n_tests, n_col), dtype=np.double)
    beta = np.empty((n_tests, n_col), dtype=np.double)
    sigma = np.empty(n_tests)
    df_residual = np.zeros(n_tests, dtype=np.double)

    design = design.as_matrix()
    out = {}

    for i in xrange(n_tests):
        y = M[i, :]
        obs = np.isfinite(y)
        X = design[obs, :]
        y = y[obs]
        if len(y) > n_col:
            out = sm.RLM(y, X).fit()
            beta[i, :] = out.params
            df_residual[i] = out.df_resid
            #stdev_unscaled[i, :] = np.sqrt(out.exog_q.qr.diagonal)
            if df_residual[i] > 0:
                #sigma[i] = out.sigma
                pass
            else: pass
    q, r = np.qr(design)

    out['coefs'] = beta
    out['stdev_unscaled'] = stdev_unscaled
    out['df_residual'] = df_residual
    return(out)




def mglm_simple(y, design, dispersion=0, offset=0, weights=None):
    """ 
    Fit negative binomial glm with calls to statsmodels glm.fit().

    Parameters
    ----------
    y : digital count matrix
    design : design dataframe or matrix
    dispersion : dispersion estimate
    offset : offset array 
    weights : weights array

    Returns:
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


def mglm_line_search(y, design, dispersion=0, offset=0, coef_start=None):
    """ Multiple 
    """
    X = design.as_matrix()
    #ncoef = X.shape[1]
    #ntags = y.shape[0]
    #nsamps = y.shape[1]
    offset = expand_as_matrix(offset, y.shape)
    # Orthnormal transform of design matrix
    q, r = qr(X)
    X = qr.Q(q, r)



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
        







def glm_lrt(glm_fit, coef, contrast=None, test='chisq'):
    """ Liklehood Ratio Test

    Parameters
    ----------

    Returns
    -------
    """
    pass



def voom(DGEList, design=None, lib_size=None):
    """ limma Voom

    Parameters
    ----------
    counts : DGEList or count matrix
    design : if counts is not a DGEList or object with design attribute. 
    lib_size:  if counts is not a DGEList or object with lib_size

    Returns
    -------

    Citation
    --------

    """
    out = {}

    if not lib_size or DGEList.samples['lib_size']:
        try:
            lib_size = DGEList.counts.sum(axis=0)
        except AttributeError:
            pass

    #y = normalize_between_samples(y)
    y = (np.log2((DGEList.counts + 0.5).T/(lib_size + 1) * 1e6)).T
    fit = lm_fit(y, DGEList.design)
    sx = fit.Amean  
    sy = np.sqrt(fit.sigma)




    return out


def matrix_eQTL():
    """ Specialize function for doing someithing similar to what matrix eQTL
    does

    Parameters
    ----------

    Returns
    -------

    Citation
    --------
    """

    pass
