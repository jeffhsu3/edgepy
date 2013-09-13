import numpy as np
cimport numpy as np
cimport cython
ctypedef np.double_t DTYPE
ctypedef np.int64_t ITYPE
cdef extern from "math.h":
        double c_sqrt "sqrt"(double)
        double c_exp "exp"(double)
        double c_log "log"(double)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(False)
def glm_one_group(int[:, ::1] x not None, double[:] dispersion, 
      double[:] offset, int maxit=50):
    """ Returns the estimated betas using newton raphson
    """
    cdef:
        np.intp_t ntags, nsamples, i, j, k
        double tol, beta, dl, info, mu
        double denominator, step
        double cur_val
    ntags = x.shape[0]
    cdef double[:] out = np.empty(ntags)
    nsamples = x.shape[1]
    tol = 1e-10
    for i in range(ntags):
        beta = 0
        for j in range(nsamples):
            cur_val = x[i, j]
            if cur_val > tol:
                beta += cur_val/c_exp(offset[j]) 
            else: pass
        if beta <= 0 : 
            beta = -np.inf
        else:
            beta = c_log(beta/nsamples)
            for it in range(maxit):
                dl = 0
                info = 0
                for k in range(nsamples):
                    mu = c_exp(beta + offset[k])
                    denominator = 1 + mu * dispersion[i]
                    dl += (x[i, k] - mu)/denominator
                    info += mu/denominator
                step = dl/info
                beta += step
                if abs(step) < 1e-10:
                    break
        out[i] = beta
    return out


cpdef glm_one_group_double(double[:, :] x, double[:] dispersion,
        double[:] offset, int maxit=50):
    """ Returns the estimated betas using newton raphson
    """
    cdef:
        np.intp_t ntags, nsamples, i, j, k
        cdef double tol, beta, dl, info, mu
        cdef double denominator, step
        cdef double cur_val
    ntags = x.shape[0]
    cdef double[:] out = np.empty(ntags)
    nsamples = x.shape[1]
    tol = 1e-10
    for i in range(ntags):
        beta = 0
        for j in range(nsamples):
            cur_val = x[i, j]
            if cur_val > tol:
                beta += cur_val/c_exp(offset[j]) 
            else: pass
        if beta <= 0 : 
            beta = -np.inf
        else:
            beta = c_log(beta/nsamples)
            for it in range(maxit):
                dl = 0
                info = 0
                for k in range(nsamples):
                    mu = c_exp(beta + offset[k])
                    denominator = 1 + mu * dispersion[i]
                    dl += (x[i, k] - mu)/denominator
                    info += mu/denominator
                step = dl/info
                beta += step
                if abs(step) < 1e-10:
                    break
        out[i] = beta
    return out
