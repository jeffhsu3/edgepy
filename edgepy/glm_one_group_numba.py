from numba import double
from numba.decorators import autojit
import numpy as np
import pdb


@autojit(locals={'beta': double})
def glm_one_group_numba(x, dispersion, 
       offset):
    """ Returns the estimated betas using newton-raphson

    Params
    x : count matrix
    dispersion : negative binomial dispersion value
    """
    maxit = 50
    ntags = x.shape[0]
    nsamples = x.shape[1]
    out = np.zeros(ntags)
    low_level = 1e-10
    for i in range(ntags):
        beta = 0
        nonzero = False 
        for j in range(nsamples):
            cur_val = x[i, j]
            if cur_val > low_level:
                beta += cur_val/np.exp(offset[j]) 
                nonzero = True
            else: pass
        if not nonzero: 
            beta = -np.inf
        else:
            beta = np.log(beta/double(nsamples))
            for it in range(maxit):
                dl = 0
                info = 0
                for k in range(nsamples):
                    mu = np.exp(beta + offset[k])
                    denominator = 1 + mu * dispersion[i]
                    dl += (x[i, k] - mu)/denominator
                    info += mu/denominator
                step = dl/info
                beta += step
                if abs(step) < 1e-6:
                    break
        out[i] = beta
    return out

