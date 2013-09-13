'''A python port of Gordon Smyth's cutWithMinN. 
'''

from collections import defaultdict 
from numpy import (isnan, repeat, nan, hstack,
        logical_not, digitize, linspace, zeros, int32)
from numpy.random import uniform
from scipy.stats.mstats import mquantiles
import numpy as np


def cut_with_min_N(x, intervals=2, min_n=1):
    """ 
    """
    x = hstack(x)
    na = isnan(x)
    out = {}
    if any(na):
        group = repeat(nan, len(x))
        out = cut_with_min_N(x=x[logical_not(na)], intervals=intervals,
                min_n=min_n)
        group[logical_not(na)] = out['group']
        out['group'] = group
        return(out)

    intervals = int(intervals)
    min_n = int(min_n)
    nx = len(x)
    if nx < intervals * min_n: raise TypeError
    if intervals == 1: return({'group':repeat(1, nx), 'breaks':nan})
    # Add jittering to ensure no x's are the same
    x = x + 1e-6*(uniform(size=nx) - 0.5)
    # Breaks equally spaced by x

    breaks_equal_x = np.linspace(min(x), max(x), num=intervals+1)
    breaks_equal_x[0] = breaks_equal_x[0] - 1
    breaks_equal_x[intervals] = breaks_equal_x[intervals] + 1

    # Breaks equally spaced by quantiles
    breaks_quantile_x = mquantiles(x, prob=np.linspace(0,1,num=intervals+1), 
            alphap=1, betap=1)
    breaks_quantile_x[0] = breaks_quantile_x[0] - 1
    breaks_quantile_x[-1] = breaks_quantile_x[-1] + 1
    
    z = cut(x, intervals, True)
    n = tabulate(z)
    good = [i[1] >= min_n for i in n.items()]
    
    if all(good): 
        out['group'] = z
        out['breaks'] = breaks_equal_x
        return out
    # Stepping down gradually

    for i in range(9):
        breaks = (i*breaks_quantile_x+(10-i)*breaks_equal_x)/10
        z = cut(x, breaks)
        n = tabulate(z)
        good = [i[1] >= min_n for i in n.items()]
        if all(good): 
            out['group'] = z
            out['breaks'] = breaks_equal_x
            return out

    # Try equally spaced by quantiles 

    z = cut(x, breaks)
    n = tabulate(z)
    good = [i[1] >= min_n for i in n.items()]
    if all(good): 
        out['group'] = z
        out['breaks'] = breaks_equal_x
        return out

    # If all else fails order by x
    o = np.argsort(x)
    n = np.floor(nx/intervals)
    nresid = nx - intervals * n
    n = np.asarray(np.repeat(n, intervals), dtype=np.int64)
    n[0] = n[0] + nresid
    z = np.repeat(np.arange(0, intervals, dtype=np.int64) , n)
    z = z[o]
    #z = z.reshape
    out['group'] = z
    out['breaks'] = breaks_quantile_x
    return(out)


def tabulate(array):
    """
    """
    cnt = defaultdict(int)
    for i in array:
        cnt[i] += 1
    return cnt


def cut(x, bins, right=True):
    """ Not in numpy release yet, but in the repo.
    Skipper Seabold
    License: BSD
    """
    if not np.iterable(bins):
        if np.isscalar(bins) and bins < 1:
            raise ValueError("'bins' should be a positive integer")
        if x.size == 0:
            trange = (0, 1)
        else:
            trange = (x.min(), x.max())
        mn, mx = [mi+0.0 for mi in trange]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        bins = np.linspace(mn, mx, bins+1, endpoint=True)
        bins[0] -= 0.5
        bins[-1] += 0.1*mx
    else:
        bins = np.asarray(bins)
        if (np.diff(bins) < 0).any():
            raise AttributeError(
                    'bins must increase monotonically'
                    )
    return np.digitize(x, bins, right)


