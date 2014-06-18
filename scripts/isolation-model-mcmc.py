#!/usr/bin/env python

"""Script for estimating parameters in an isolation model.
"""

from argparse import ArgumentParser

from IMCoalHMM.isolation_model import IsolationModel
from IMCoalHMM.likelihood import Likelihood
from pyZipHMM import Forwarder

from IMCoalHMM.mcmc import MCMC, MC3, LogNormPrior
from math import log


def main():
    """
    Run the main script.
    """
    usage = """%(prog)s [options] <forwarder dirs>

This program samples the posterior parameters of an isolation model with two species
and uniform coalescence and recombination rates."""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.4")

    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/dev/stdout",
                        help="Output file for the estimate (/dev/stdout)")

    parser.add_argument("--logfile",
                        type=str,
                        default=None,
                        help="Log for sampled points in all chains for the MCMCMC during the run." \
                             "This parameter is only valid when running --mc3.")

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

    parser.add_argument("--mc3", help="Run a Metropolis-Coupled MCMC", action="store_true")
    parser.add_argument("--mc3-chains", type=int, default=3, help="Number of MCMCMC chains")
    parser.add_argument("--temperature-scale", type=float, default=1.0,
                        help="The scale by which higher chains will have added temperature." \
                             "Chain i will have temperature scale*i.")

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

    if options.logfile and not options.mc3:
        parser.error("the --logfile option is only valid together with the --mc3 option.")

    # Specify priors and proposal distributions... 
    # I am sampling in log-space to make it easier to make a random walk
    split_prior = LogNormPrior(log(options.split))
    coal_prior = LogNormPrior(log(1/(options.theta/2)))
    rho_prior = LogNormPrior(log(options.rho))
    priors = [split_prior, coal_prior, rho_prior]

    # Draw initial parameters from the priors
    init_params = [pi.sample() for pi in priors]

    # Read data and provide likelihood function


    if options.mc3:
        mcmc = MC3(priors, input_files=options.alignments,
                   model=IsolationModel(options.states),
                   thinning=options.thinning, no_chains=options.mc3_chains,
                   switching=options.thinning/10,
                   temperature_scale=options.temperature_scale)
    else:
        forwarders = [Forwarder.fromDirectory(arg) for arg in options.alignments]
        log_likelihood = Likelihood(IsolationModel(options.states), forwarders)
        mcmc = MCMC(priors, log_likelihood, thinning=options.thinning)

    def transform(params):
        split_time, coal_rate, recomb_rate = params
        return split_time, 2 / coal_rate, recomb_rate

    with open(options.outfile, 'w') as outfile:
        print >> outfile, '\t'.join(['split.time', 'theta', 'rho', 'posterior'])

        if options.logfile:
            with open(options.logfile, 'w') as logfile:
                print >> logfile, '\t'.join(['chain', 'split.time', 'theta', 'rho', 'posterior'])

                for _ in xrange(options.samples):
                    params, post = mcmc.sample()

                    # Main chain written to output
                    print >> outfile, '\t'.join(map(str, transform(params) + (post,)))
                    outfile.flush()

                    # All chains written to the log
                    for chain_no, chain in enumerate(mcmc.chains):
                        params = chain.current_theta
                        post = chain.current_posterior
                        print >> logfile, '\t'.join(map(str, (chain_no,) + transform(params) + (post,)))
                    logfile.flush()
        else:
            for _ in xrange(options.samples):
                params, post = mcmc.sample()
                print >> outfile, '\t'.join(map(str, transform(params) + (post,)))
                outfile.flush()

    if options.mc3:
        mcmc.terminate()

if __name__ == '__main__':
    main()
