""" Conditinal log liklihood functions
"""
from numpy import sum, logical_not, power
#from numpy import power as pow
from scipy.special import gammaln, digamma
#digamma, polygamma 
from numpy import nansum


def common_log_lik_delta(delta, y, groups, sign=1.0):
    r = (1/delta) - 1
    lO = 0
    for _, group in groups.iteritems():
        n = y[:, group].shape[1]
        t = nansum(y[:,group], axis=1)
        lO += nansum(gammaln(y[:,group] + r), axis=1) + gammaln(n*r)\
                - gammaln(t + n * r) - n * gammaln(r)
    return(nansum(lO * sign))


def common_log_lik_der(delta, y, groups, sign=1.0):
    r = (1/delta) - 1
    lO = 0
    for _, group in groups.iteritems():
        n = y[:, group].shape[1]
        t = sum(y[:,group], axis=1)
        lO += (sum(digamma(y + r), axis=1) + (n * digamma(n * r))-\
                n * digamma(t + n * r) - n * digamma(r)) * (-1 *\
                power(delta, 2))
    return(sum(lO * sign))
