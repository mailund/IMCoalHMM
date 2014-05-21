#!/usr/bin/env python

"""Script for estimating parameters in an isolation model.
"""

from argparse import ArgumentParser

from IMCoalHMM.isolation_model import IsolationModel
from IMCoalHMM.likelihood import Likelihood
from pyZipHMM import Forwarder

from IMCoalHMM.mcmc import MCMC, LogNormPrior
from math import log


def main():
    """
    Run the main script.
    """
    usage = """%(prog)s [options] <forwarder dirs>

This program samples the posterior parameters of an isolation model with two species
and uniform coalescence and recombination rates."""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.0")

    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/dev/stdout",
                        help="Output file for the estimate (/dev/stdout)")

    parser.add_argument("--states",
                        type=int,
                        default=10,
                        help="Number of intervals used to discretize the time (10)")

    parser.add_argument("-n", "--samples",
                        type=int,
                        default=500,
                        help="Number of samples to draw (500)")

    parser.add_argument("-k", "--thinning",
                        type=int,
                        default=100,
                        help="Number of MCMC steps between samples (100)")

    meta_params = [
        ('split', 'split time in substitutions', 1e6 / 1e9),
        ('theta', 'effective population size in 4Ne substitutions', 1e6 / 1e9),
        ('rho', 'recombination rate in substitutions', 0.4),
    ]

    for parameter_name, description, default in meta_params:
        parser.add_argument("--%s" % parameter_name,
                            type=float,
                            default=default,
                            help="Meta-parameter mean of the %s (%g)" % (description, default))

    parser.add_argument('alignments', nargs='+', help='Alignments in ZipHMM format')

    options = parser.parse_args()
    if len(options.alignments) < 1:
        parser.error("Input alignment not provided!")

    # Specify priors and proposal distributions... 
    # I am sampling in log-space to make it easier to make a random walk
    split_prior = LogNormPrior(log(options.split))
    coal_prior = LogNormPrior(log(1/(options.theta/2)))
    rho_prior = LogNormPrior(log(options.rho))
    priors = [split_prior, coal_prior, rho_prior]

    # Draw initial parameters from the priors
    init_params = [pi.sample() for pi in priors]

    # Read data and provide likelihood function
    forwarders = [Forwarder.fromDirectory(arg) for arg in options.alignments]
    log_likelihood = Likelihood(IsolationModel(options.states), forwarders)

    mcmc = MCMC(priors, log_likelihood)

    def transform(params):
        split_time, coal_rate, recomb_rate = params
        return split_time, 2 / coal_rate, recomb_rate

    with open(options.outfile, 'w') as outfile:
        print >> outfile, '\t'.join(['split.time', 'theta', 'rho', 'posterior'])
        
        for _ in xrange(options.samples):
            params, post = mcmc.sample(thinning=options.thinning)
            print >> outfile, '\t'.join(map(str, transform(params) + (post,)))
            outfile.flush()


if __name__ == '__main__':
    main()
