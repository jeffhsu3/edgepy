"""
Container for digital expression data.

Robinson M, McCarthy DJ, Smyth GK. edgeR: a Bioconductor package for
differential expression anaylsis of digital expression data. Bioinformatics 26,
139-140. 2010

Jeffrey Hsu
"""

import pandas as pd
import numpy as np
from numpy.linalg import lstsq

from scipy.optimize import minimize_scalar
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp1d
from glm_one_group_numba import glm_one_group_numba
from q2qnbinom import q2qnbinom
import norm_functions as nf
import cond_log_lik as ccl
from cut_with_min_N import cut_with_min_N
#from numpy.ma import masked_array
#from bspline import bsplvander
from statsmodels.tools.tools import add_constant
# Try import from both biopython and statsmodels
from statsmodels.nonparametric.smoothers_lowess import lowess
from .glm_fit import glm_fit


def norm_factor(matrix):
    """ Calculates a normalization factor
    """
    size = matrix.sum()
    return(1/(size / float(size[0])))


class DGEList(object):
    """ Class for holding RNAsequencing read counts for overdispersed count
    data.

    Parameters:

    """
    def __init__(self, counts, design_matrix=None, genes=None):
        """ 
        DGEList object simliar to Gordon Smyth's DGEList. Transcripts that have
        no tags across all samples are removed.

        Parameters
        ----------
        counts: counts of digital expression data as a pandas dataframe
        design_matrix: experimental design matrix
        genes: An annotation dataframe for the count data.
        """
        # Remove transcripts that are
        good_tags = np.logical_not(counts.sum(axis=1) == 0)
        self.counts = counts.ix[good_tags, :]
        self.samples = pd.DataFrame({"id": self.counts.columns,  
            "lib_sizes": self.counts.sum()}, index = self.counts.columns)
        self.genes = genes
        if design_matrix is not None:
            self.design = design_matrix
        else:
            self.design = pd.DataFrame({
                'group': np.repeat(1, self.counts.shape[1]),
                })
        self.groups = self.design.groupby(list(self.design.columns)).groups
        self.groups_loc = {}
        for key, value in self.groups.iteritems():
            self.groups_loc[key] = [self.counts.columns.get_loc(i) for i in
                    value]
    
    
    def calc_norm_factors(self, method = 'TMM', ref_lib=0):
        """ Calculates normalization factor between the libraries
        """
        y = self.counts.ix[self.counts.sum(axis=1) > 0, :]
        if method == 'TMM':
            f = nf.TMM(y)
        elif method == 'RLE':
            f = nf._RLE(y)/self.samples.lib_sizes
        f = f/np.exp(np.log(f).mean())
        self.samples['norm_factor'] = f

    
    def _equalize_lib_sizes(self, dispersion=0, common_size=None):
        lib_size = self.get_lib_size() 
        return(equalize_lib_sizes(self.counts, self.groups_loc, lib_size))

    
    def estimate_dispersion(self, tol=1e-06, rowsum_filter = 5):
        """ Estimates the common dispersion across all tags.
        """
        lib_size = self.get_lib_size()
        out = estimate_dispersion(self.counts, self.groups_loc, lib_size)
        self.common_dispersion = out[0]
        self.pseudo_counts = out[1]
        self.common_size = out[1]
        self.avg_log_cpm = self.avg_cpm()


    def estimate_trended_dispersion(self, method='bin_spline', df=5, 
            frac=2/3.0):
        """ 
        Calculates the trended dispersion across tags based on the average
        log counts per million.  Adds trended_dispersion attribute.  Note this
        is implemented differently for the spline method than it is done in
        edgeR.  The location of the knots is actually not specified.  Rather it
        is calculated from the fitting algorithm.

        Parameters
        ----------
        method: either 'bin_spline' or 'bin_lowess'
        df: number of knots to fit a spline to, default is 5
        frac: proportion of the data to use in estimating y, 
        analogous to R's loess fit's span, defaulte is 2/3.

        Returns
        ------
        """
        nbins = 50
        ntags = self.counts.shape[0]
        bins = cut_with_min_N(self.avg_log_cpm, intervals=nbins,
                min_n=ntags/nbins)
        disp_bins, log_cpm_bins = np.repeat(np.nan, nbins), np.repeat(np.nan,
                nbins)
        lib_size = self.get_lib_size()
        if nbins > ntags:
            raise TypeError
        for i in range(nbins):
            tags_in_bin = bins['group'] == i
            '''
            print(tags_in_bin)
            print(self.counts.ix[tags_in_bin,:])
            '''
            if not any(tags_in_bin):
                raise TypeError
            disp_bins[i] =\
            estimate_dispersion(self.counts.ix[tags_in_bin,:],
                    self.groups_loc,
                    lib_size)[0]
            log_cpm_bins[i] = self.avg_log_cpm[tags_in_bin].mean()

        if method == 'bin_spline':
            p1 = np.arange(1.0, (df))/df
            knots1 = mquantiles(log_cpm_bins, prob = p1, alphap=1,
                    betap=1)
            r = (np.min(log_cpm_bins), np.max(log_cpm_bins))
            knots2 = r[0]+p1*(r[1]-r[0])
            knots = 0.3*knots1 + 0.7*knots2
            ind = np.zeros(df + 1)
            ind[0] = np.argmin(log_cpm_bins)
            ind[df] = np.argmax(log_cpm_bins)
            for i in range(1, df):
                # Still missing i = 4 for ind
                temp = np.abs(knots[i-1]-log_cpm_bins)
                ind[i] = np.argmin(temp)
            #fit = lstsq(disp_bins, log_cpm_bins)
            #f = interp1d(log_cpm_bins[ind], fit, s=0)
            #dispersion = f(self.avg_log_cpmn)
            raise NotImplemented
            #test = bsplvander(log_cpm_bins,  knots , df-1)

        elif method == 'bin_loess':
            # Need to be done else interpertation range is mucked up
            log_cpm_bins[0] = min(self.avg_log_cpm)
            log_cpm_bins[-1] = max(self.avg_log_cpm)
            fit = lowess(disp_bins, log_cpm_bins, frac=frac)
            f = interp1d(fit[:,0], fit[:,1])
        dispersion = f(self.avg_log_cpm)
        self.trended_dispersion = dispersion


    def estimate_tagwise_dispersion():
        """ 
        """

        raise NotImplemented


    def principle_component():
        """
        """
        raise NotImplemented


    def avg_cpm(self, normalized_lib_sizes=True, prior_count=2, dispersion=0.05):
        if normalized_lib_sizes:
            lib_size = self.samples['norm_factor'] * self.samples['lib_sizes']
            return(average_cpm(self.counts, lib_size = lib_size, prior_count =
                    prior_count, dispersion=dispersion))
                
                
    def get_lib_size(self):
        """ Returns library size or caculates norm_factor if it hasn't been
        calculated yet.
        """
        offset = getattr(self, 'offset', None)
        if getattr(offset, 'size', None):
            pass
        else:
            try:
                offset = self.samples['lib_sizes'] * self.samples['norm_factor']
            except KeyError:
                self.calc_norm_factors(method='RLE')
                offset = self.samples['lib_sizes'] * self.samples['norm_factor']
            self.offset = offset
        return offset


    def glm_fit(self):
        """ Fit the generalized linear model

        Parameters
        ----------

        Returns
        -------
        """
        offset = self.get_lib_size()
        dispersion = getattr(self, 'trended_dispersion', None)
        if not getattr(dispersion, 'shape', None):
            raise AttributeError('Need to estimate the dispersion\
                    parameter first')
        #avg_log_cpm = self.avg_log_cpm
        temp_design = add_constant(self.design)
        fit = glm_fit(self.counts, temp_design, dispersion, offset)
        self.pvalues = fit[4]
        print(fit[4][0:10, 0:10])



    # Plotting functions
    def histogram(self, **kwargs):
        # :TODO fix this
        import matplotlib.pyplot as plt
        hist_data = [np.histogram(self.counts[x], **kwargs) for x in\
                self.counts]
        for i, j in hist_data:
            j = 0.5*(j[1:]+j[:-1])
            plt.plot(j, i, '-')


    def plot_dispersion(self):
        """ Plots the dispersion as a fucntion of the average log counts per
        million.
        """
        raise NotImplemented


def average_cpm(y, lib_size = None, prior_count=2, dispersion=0.05):
    """ 

    Parameters
    ---------
    y: matrix of counts
    lib_size: 
    prior_count:
    dispersion:

    """
    y = np.ascontiguousarray(y)
    #if lib_size == None: np.sum(y, axis=0)
    prior_counts_scaled = np.asarray(lib_size/np.mean(lib_size) *
            prior_count, dtype=np.double).reshape(len(lib_size), 1)
    offset = np.log(np.asarray(lib_size,
        dtype=np.double).reshape(len(lib_size),1) + 2 * prior_counts_scaled)
    dispersion = np.repeat(dispersion, y.shape[0])
    abundence = glm_one_group_numba(np.ascontiguousarray((y.T +
        prior_counts_scaled).T), 
            dispersion, offset.T[0])
    return((np.asarray(abundence) + np.log(1e6))/np.log(2))


def equalize_lib_sizes(counts, groups, lib_size, dispersion=0, common_size=None):
    """ Equalize the library sizes
    """
    if not common_size:
        common_size = np.exp(np.mean(np.log(lib_size)))
    try: len(dispersion)
    except TypeError: dispersion = np.repeat(dispersion, 
            counts.shape[0]) 
    input_mean = np.empty(counts.shape, dtype=np.double)
    output_mean = input_mean.copy()
    for key, group in groups.iteritems():
        beta = glm_one_group_numba(
                np.ascontiguousarray(counts.ix[:,group].as_matrix(),
                    dtype=np.int32),
                dispersion,
                np.ascontiguousarray(np.log(lib_size[group]),
                    dtype=np.double)
                )
        beta = np.asarray(beta)
        bn_lambda = np.exp(beta).T.reshape(len(beta),1)
        temp_lib_size = np.array(lib_size[group]).reshape(1,
                len(lib_size[group]))
        out_size = np.repeat(common_size, len(group)).reshape(1,
                len(group))
        input_mean[:, group] = np.dot(bn_lambda, temp_lib_size)
        output_mean[:, group] = np.dot(bn_lambda, 
                out_size)
    pseudo = q2qnbinom(np.asarray(counts.as_matrix()), 
            input_mean, output_mean, dispersion)
    pseudo[pseudo < 0] = 0
    return pseudo, common_size


def estimate_dispersion(counts, groups, lib_size, rowsum_filter=5, tol=1e-06):
    """  Estimate the dispersion in a negative binoamial

    Parameters
    ----------
    counts:
    groups:
    lib_size:
    """
        
    tags = np.array(np.sum(counts.as_matrix(), axis=1) >= rowsum_filter)
    disp = 0.01
    for i in [0,1]:
        pseudo_counts, common_size = equalize_lib_sizes(counts, groups,
                lib_size,
                dispersion=disp)
        delta = minimize_scalar(ccl.common_log_lik_delta,
                bounds = (1e-4, 100.0/(100.0+1)),
                method = 'Bounded',
                args=(pseudo_counts[tags,:], groups, -1), 
                tol=1e-06,
                options = {}).x
        disp = delta/(1-delta)
    return disp, pseudo_counts, common_size


