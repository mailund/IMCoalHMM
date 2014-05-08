import ez_setup
ez_setup.use_setuptools()

import os


def read(fname):
    """For reading in the README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

from setuptools import setup, find_packages
setup(
    name = "IMCoalHMM",
    version = "0.4.0",
    packages = find_packages(where='src'),
    package_dir = {'': 'src'},
    
    scripts = ['scripts/prepare-alignments.py',
               'scripts/isolation-model.py',
               'scripts/isolation-model-mcmc.py',
               'scripts/initial-migration-model.py',
               'scripts/initial-migration-model-isolation-profile-likelihood.py',
               'scripts/initial-migration-model-migration-profile-likelihood.py',
               'scripts/initial-migration-model-theta-profile-likelihood.py',
               'scripts/initial-migration-model-rho-profile-likelihood.py',
               'scripts/initial-migration-model-migration-rate-profile-likelihood.py',
               'scripts/psmc-isolation-model.py',
               'scripts/variable-migration-model.py',
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