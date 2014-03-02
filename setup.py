import ez_setup
ez_setup.use_setuptools()

import os
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

from setuptools import setup, find_packages
setup(
    name = "IMCoalHMM",
    version = "0.1",
    packages = find_packages(where='src'),
    package_dir = {'': 'src'},
    
    scripts = ['scripts/prepare-alignments.py',
               'scripts/isolation-model.py',
              ],

    install_requires = ['numpy', 
                        'scipy',
                        #'pyZipHMM', -- FIXME: there isn't a package for this!
                        ],

    # metadata for upload to PyPI
    author = "Thomas Mailund",
    author_email = "mailund@birc.au.dk",
    license = "GPLv2",
    keywords = "demography coalescence bioinformatics genetics",
    url = "https://github.com/mailund/IMCoalHMM",
    description = "Package for building and analysing pair-wise CoalHMMs.",
    long_description = read('README.md'),

)