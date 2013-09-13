import numpy as np
from numpy import multiply, logical_or, empty, logical_not
from scipy.stats import norm, gamma


def q2qnbinom(counts, input_mean, output_mean, dispersion):
    """ Quantile to Quantile for a negative binomial
    """
    zero = logical_or(input_mean < 1e-14, output_mean < 1e-14)
    input_mean[zero] = input_mean[zero] + 0.25
    output_mean[zero] = output_mean[zero] + 0.25
    ri = 1 + multiply(np.matrix(dispersion).T, input_mean)
    vi = multiply(input_mean , ri)
    rO = 1 + multiply(np.matrix(dispersion).T, output_mean)
    vO = multiply(output_mean, rO)
    i = counts >= input_mean
    low = logical_not(i)
    p1 = empty(counts.shape, dtype=np.float64)
    p2 = p1.copy()
    q1, q2 = p1.copy(), p1.copy()
    if i.any():
        p1[i] = norm.logsf(counts[i], loc=input_mean[i],
                scale=np.sqrt(vi[i]))[0,:] 
        p2[i] = gamma.logsf(counts[i], (input_mean/ri)[i], scale=ri[i])[0, :]
        q1[i] = norm.ppf(1-np.exp(p1[i]), output_mean[i], np.sqrt(vO[i]))[0, :] 
        q2[i] = gamma.ppf(1-np.exp(p2[i]), np.divide(output_mean[i],rO[i]),
                scale=rO[i])[0, :]

    if low.any():
        p1[low] = norm.logcdf(counts[low], 
                loc=input_mean[low], 
                scale=np.sqrt(vi[low]))[0, :] 
        p2[low] = gamma.logcdf(counts[low], 
                input_mean[low]/ri[low], 
                scale = ri[low])[0,:]
        q1[low] = norm.ppf(np.exp(p1[low]), 
                loc=output_mean[low],
                scale=np.sqrt(vO[low]))[0,:] 
        q2[low] = gamma.ppf(np.exp(p2[low]), 
                output_mean[low]/rO[low], 
                scale = rO[low])[0,:]
    return((q1+q2)/2)
