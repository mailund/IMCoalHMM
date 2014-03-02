Isolation with migration CoalHMMs
=================================

Cleaner re-implementation of CoalHMMs specialised for pairs of genomes in isolation and isolation-with-migration models.

Without having to deal with the general cases with an arbitrary number of samples the algorithms can be made much simpler, so this library should be both cleaner and faster than the general library, but at the cost of only modeling isolation-with-migration models.

The code contains a python package for building demographic models and translating them into hidden Markov models and scripts for doing inference in pre-specified models.


Scripts
-------

The script _prepare-alignments.py_ is used to translate a pairwise alignment into the formath ZipHMM uses to compute the likelihood of a model. This is a preprocessing script that will always be needed before an analysis.

The script _isolation-model.py_ implements the isolation model from Mailund _et al._ (2011): [Estimating Divergence Time and Ancestral Effective Population Size of Bornean and Sumatran Orangutan Subspecies Using a Coalescent Hidden Markov Model](http://www.plosgenetics.org/article/info%3Adoi%2F10.1371%2Fjournal.pgen.1001319). The script will estimate the split time, the effective population size and the recombination rate, all measured in number of substitutions, in a model assuming a clean split between two species.


Requirements
------------

The code for building hidden Markov models requires [numpy](http://www.numpy.org) and [scipy](http://www.scipy.org) to be installed and the HMM code uses [ZipHMM](https://code.google.com/p/ziphmm/).

The _prepare-alignments.py_ script also requires [BioPython](http://biopython.org) to be installed in order to read in alignment files in different formats.