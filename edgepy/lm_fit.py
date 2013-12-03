
"""
"""

from numpy import np

def lm_fit(y):
    """ Fit linear model
    """
    pass


def lm_series(M, design, ndups=2, spacing=1, block=None, correlation=None,
        weights=None):
    """ Fit linear model for each gene to a series of microarrays.
    Fit is by generalized least squares allowing for correlation between
    duplicate spots.

    """

    n_arrays = M.shape[1]
    n_beta = design.shape[1]
    coef_names = design.columns

    if getattr(weights, 'size', False):
        weights[np.insnan[weights]] = 0
        if weights.shape[0] == M.shape[0]:
            weights = np.tile(weights, (1, M.shape[1]))
        else:
            weights = np.tile(weights, (M.shape[0], 1))
        # :TODO need to refactor this
        M[weights < 1e-15] = np.nan
        weights[weights < 1e-15] = np.nan


    if ndups >= 1:
        M = unwrap_duplicates(M, ndups=ndups, spacing=spacing)
        design = np.mul(design, np.repeat(1, ndups))
        if getattr(weights, 'size', False):
            weights = unwrap_duplicates(weights, ndups=ndups, spacing=spacing)

    n_genes = M.shape[0]

    stdev_unscaled = np.repeat(np.nan, (n_genes, n_beta))
    sigma = np.repeat(np.nan, n_genes)
    df_resid = np.zeros(n_genes)
    no_probe_wts = not any(np.isnan(M)) and\
            (getattr(weights, 'size', False) or\
             getattr(weights, array_weights, False))
    if no_probe_wts:
        if getattr(weights, 'size', False):
            # :TODO find best lm fit function
            fit = lm_fit(design, M.T)
        else:
            fit = lm_wfit(design, M.T, weights[1,:])
            fit.weights = None
        if fit.df_residual > 0:
            fit.sigma = np.sqrt()

        df_resid = fit.resid
        sigma = fit.sigma
        weights = fit





    else: pass


    if not correlation:
        correlation = duplicate_correlation(M, design=design, 
                                            ndups=ndups)


    if not block:
        if ndups < 2:
            ndups = 1
            correlation = 0
        corr_matrix = np.diag(np.repeat(correlation, )).dot(())
        


    corr_matrix = Z.dot(correlation * Z.T)

    """
    if NoProbeWts:
        V = corr_matrix
    """


def unwrap_duplicates(M, ndups=1, spacing=0):
    """ Unwrap duplicate spots in an array
    """
    pass


def duplicate_correlation(M, design=None, ndups=1, spacing=1):
    """
    """
    pass

def as_matrix_weights():
    """ Convert probe-weights or array-weights to a matrix of weights
    """
