#cython: boundscheck=False
#cython: wraparound=False
from __future__ import division
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

ctypedef np.double_t DTYPE
ctypedef np.int64_t ITYPE

cdef extern from "math.h":
        double c_sqrt "sqrt"(double)
        double c_exp "exp"(double)
        

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double glm_one_group(int[:] x, double[:] offset, 
        double dispersion,
        int maxit=50, double tol=1e-10):
    
    cdef double cur_beta = 0
    cdef double low_level = 1e-10
    cdef double info = 0
    cdef float step, dl, mu, denom
    cdef np.intp_t j, nsamp 
    cdef unsigned int i 
    nsamp = x.shape[0]
    #cur_beta = (x[x>low_level]/np.exp(offset[x>low_level])).sum()
    for j in range(nsamp):
        cur_beta += x[j]/c_exp(offset[j])
    if not cur_beta > 0: return(np.inf)
    # Newton-Raphson iterations
    else:
        #has_converged = False
        cur_beta = np.log(cur_beta/x.shape[0])
        for i in xrange(0, maxit):
            dl = 0
            info = 0
            for j in range(nsamp):
                mu = c_exp(cur_beta + offset[j])
                denom = 1 + mu * dispersion
                dl += ((x[j] - mu)/denom)
                info += (mu/denom)
            step = dl/info
            cur_beta += step
            if abs(step) < tol:
                #has_converged = True
                break
            else: pass
        return(cur_beta)



@cython.wraparound(False)
@cython.boundscheck(False)
def mglgmOneGroup(int[:, ::1] counts, 
        double[:] dispersion, 
        double[:] norm_offset, 
        int maxit=50, double tol=1e-10):
    """ Estimate the beta given a design. All dispersions must be > 0.
    """
    cdef np.intp_t i, ntags
    cdef unsigned int nsamp
    cdef double disp
    cdef float m
    #cdef np.ndarray[DTYPE, ndim=1] tag
    cdef int[:] tag
    ntags = counts.shape[0]
    cdef double[:] betas = np.empty(ntags)
    #cdef np.ndarray[DTYPE, ndim=1] N
    #cdef np.ndarray betas = np.zeros([ntags,], dtype=np.float)
    #if any(dispersion < 0): raise ValueError("Dispersion cannot be negative")
    #N = np.exp(norm_offset)
    # Poisson special case
    '''
    if all(dispersion == 0):
        m = N.mean(axis=0)
        #return(np.log((counts/m).mean(axis=0)))
        return(0)
    '''
    for i in xrange(ntags): 
        #tag = np.asarray(<np.int32_t[:nsamp]> &counts[i, :]) 
        tag = counts[i, :]
        disp = dispersion[i]
        betas[i] = glm_one_group(tag, norm_offset, disp)
    return betas 


'''
cpdef double glm_one_group(x, dispersion, np.ndarray[double, ndim=1] offset, int maxit=50, double tol=1e-10):
    """ Returns an updated beta
    """
    cdef double cur_beta = 0
    cdef double low_level = 1e-10
    cdef double info = 0
    cdef double step, dl
    cdef unsigned int i
    cur_beta = (x[x>low_level]/np.exp(offset[x>low_level])).sum()
    if not cur_beta > 0: return(np.inf)
    
    # Newton-Raphson iterations
    else:
        #has_converged = False
        cur_beta = np.log(cur_beta/x.shape[0])
        for i in xrange(0, maxit):
            mu = np.exp(cur_beta + offset)
            denom = 1 + mu * dispersion[x.name]
            dl = ((x - mu)/denom).sum()
            info = (mu/denom).sum()
            step = dl/info
            cur_beta += step
            if abs(step) < tol:
                #has_converged = True
                break
            else: pass
        return(cur_beta)
'''
