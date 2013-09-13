"""
Special subclass of DGEList that specifically deals with eQTL studies
"""

from edgepy.DGEList import DGEList



class DGE_eQTL(DGEList):
    """
    An extension of DGEList. Where the dispersion is calculated form the
    covariates. 


    """


    def __init__(self, counts, design_matrix=None, genes=None,
            genotypes=None):
        """

        Parameters
        ----------
        y: 
        design:
        """
        if genotypes:
            self.genotypes = genotypes

        self.super(DGE_eQTL, self).__init__(counts, design_matrix, genes)


    def add_genotypes(self, genotypes):
        """

        Parameters
        ----------
        genotypes: a matrix of the genotypes or a genda.genotype object

        Subsets the data to only include samples that have both genotype 
        and expression data.  

        """
        common = [i for i in self.counts.columns if i in genotypes.columns]
        self.genotypes = genotypes.ix[:, common]

        self.

        # recalculate 





class aei_eQTL(DGEList):
    """ 
    Adding aellic expression imblance to the eQTLs.  P-values are fischer
    combined p-values with the eQTL data.
    """

    def __init__(self,):
        """

        Parameters
        ----------
        genotype
        """
