
import os, sys, glob
#import pysam

name = 'edgepy'
version = '0.1'

from distutils.core import setup
from distutils.extension import Extension

def make_ext(modname, pyxfilename):
    #import pysam, os
    dirname = os.path.dirname( pysam.__file__)[:-len("pysam")]
    return Extension(name = modname, sources = [pyxfilename],
                    extra_link_args = [os.pth.join(dirname, "csamtools.so")],
                    include_dirs = ['../samtools',
                                    '../pysam',
                                    ])

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
    print("Cython not found")
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    print('using cython')
    ext_modules = cythonize('edgepy/glm_levenberg.pyx', 
            sources=['src/glm_levenberg.cpp'], 
            language='c++')
    cmdclass.update({'build_ext': build_ext})
else:
    pass


metadata = {'name':name,
            'version': version,
            'cmdclass': cmdclass,
            'ext_modules': ext_modules,
            'scripts': glob.glob('scripts/*.py'),
            'description':'A python port the R package edgeR by Gordon Smyth\
                    and DESeq by Simon Anders',
            'author':'Jeffrey Hsu',
            'packages':['edgepy'],
}


if __name__ == '__main__':
    dist = setup(**metadata)
    """
        Extension("pySeq.pysam_callbacks.gene_counter",
            ["pySeq/pysam_callbacks/gene_counter.pyx"],
            include_dirs=pysam.get_include()),
    """
