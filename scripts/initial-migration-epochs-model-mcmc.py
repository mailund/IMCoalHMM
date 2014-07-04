#!/usr/bin/env python

"""Script for estimating parameters in an initial migration model with multiple epochs.
"""

from argparse import ArgumentParser

from IMCoalHMM.likelihood import Likelihood
from IMCoalHMM.isolation_with_migration_model_epochs import IsolationMigrationEpochsModel
from pyZipHMM import Forwarder


from IMCoalHMM.mcmc import MCMC, MC3, LogNormPrior, ExpLogNormPrior
from math import log

import sys


def transform(no_epochs, parameters):
    """
    Translate the parameters to the input and output parameter space.
    """

    isolation_time, migration_time, recomb_rate = parameters[:3]
    coal_rates = tuple(parameters[3:2 * no_epochs + 1 + 3])
    mig_rates = tuple(parameters[2 * no_epochs + 1 + 3:])
    thetas = tuple(2 / coal_rate for coal_rate in coal_rates)

    transformed = (isolation_time, migration_time) + thetas + (recomb_rate,) + mig_rates
    return transformed


def main():
    """
    Run the main script.
    """
    usage = """%(prog)s [options] alignments...

This program estimates the parameters of an isolation model with an initial migration period with two species
and uniform coalescence and recombination rates and with variable effective population size."""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.1")

    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/dev/stdout",
                        help="Output file for the estimate (/dev/stdout)")

    parser.add_argument("--logfile",
                        type=str,
                        default=None,
                        help="Log for sampled points in all chains for the MCMCMC during the run." \
                             "This parameter is only valid when running --mc3.")

    parser.add_argument("--epochs",
                        type=int,
                        default=2,
                        help="Number of epochs (variation in population size and migration rate) used " \
                             "in the migration period and in the ancestral population (2)")
    parser.add_argument("--ancestral-states",
                        type=int,
                        default=5,
                        help="Number of intervals per epoch used to discretize the time in the " \
                             "ancestral population (5)")
    parser.add_argument("--migration-states",
                        type=int,
                        default=5,
                        help="Number of intervals per epoch used to discretize the time in the migration period (5)")

    parser.add_argument("-n", "--samples",
                        type=int,
                        default=500,
                        help="Number of samples to draw (500)")


    parser.add_argument("--mc3", help="Run a Metropolis-Coupled MCMC", action="store_true")
    parser.add_argument("--mc3-chains", type=int, default=3, help="Number of MCMCMC chains")
    parser.add_argument("--temperature-scale", type=float, default=1.0,
                        help="The scale by which higher chains will have added temperature." \
                             "Chain i will have temperature scale*i.")
    parser.add_argument("-k", "--thinning",
                        type=int,
                        default=100,
                        help="Number of MCMC steps between samples (100)")


    parser.add_argument("--sample-priors", help="Sample independently from the priors", action="store_true")
    parser.add_argument("--mcmc-priors", help="Run the MCMC but use the prior as the posterior", action="store_true")

    meta_params = [
        ('isolation-period', 'time where the populations have been isolated', 1e6 / 1e9),
        ('migration-period', 'time period where the populations exchanged genes', 1e6 / 1e9),
        ('theta', 'effective population size in 4Ne substitutions', 1e6 / 1e9),
        ('rho', 'recombination rate in substitutions', 0.4),
        ('migration-rate', 'migration rate in number of migrations per substitution', 200.0)
    ]

    for parameter_name, description, default in meta_params:
        parser.add_argument("--%s" % parameter_name,
                            type=float,
                            default=default,
                            help="Meta-parameter mean of the %s (%g)" % (description, default))

    parser.add_argument('alignments', nargs='*', help='Alignments in ZipHMM format')

    options = parser.parse_args()
    if len(options.alignments) < 1 and not (options.sample_priors or options.mcmc_priors):
        parser.error("Input alignment not provided!")

    if len(options.alignments) > 0 and options.mcmc_priors:
        parser.error("You should not provide alignments to the command line when sampling from the prior")

    if options.logfile and not options.mc3:
        parser.error("the --logfile option is only valid together with the --mc3 option.")

    # Specify priors and proposal distributions... 
    # I am sampling in log-space to make it easier to make a random walk
    isolation_period_prior = LogNormPrior(log(options.isolation_period))
    migration_period_prior = LogNormPrior(log(options.migration_period))
    coal_prior = LogNormPrior(log(1/(options.theta/2)))
    rho_prior = LogNormPrior(log(options.rho))
    migration_rate_prior = ExpLogNormPrior(options.migration_rate)

    # Putting these together with the right number of priors when there are multiple epochs...
    priors = [isolation_period_prior, migration_period_prior, rho_prior] + \
             [coal_prior] * (2 * options.epochs + 1) + \
             [migration_rate_prior] * options.epochs

    out_header = '\t'.join(['isolation.period', 'migration.period'] +
                           ['isolation.theta'] +
                           ['migration.theta.{}'.format(epoch) for epoch in range(options.epochs)] +
                           ['ancestral.theta.{}'.format(epoch) for epoch in range(options.epochs)] +
                           ['rho'] +
                           ['migration.{}'.format(epoch) for epoch in range(options.epochs)] +
                           ['log.prior', 'log.likelihood', 'log.posterior'])
    log_header = "chain\t{}".format(out_header)

    # If we only want to sample from the priors we simply collect random points from these
    if options.sample_priors:
        with open(options.outfile, 'w') as outfile:
            print >> outfile, out_header
            for _ in xrange(options.samples):
                params = [prior.sample() for prior in priors]
                prior = sum(log(prior.pdf(p)) for prior, p in zip(priors, params))
                likelihood = 0.0
                posterior = prior + likelihood
                print >> outfile, '\t'.join(map(str, transform(options.epochs, params) +
                                                (prior, likelihood, posterior)))
                outfile.flush()

        sys.exit(0) # Successful termination

    if options.mc3:
        mcmc = MC3(priors, input_files=options.alignments,
                   model=IsolationMigrationEpochsModel(options.epochs,
                                                       options.migration_states,
                                                       options.ancestral_states),
                   thinning=options.thinning, no_chains=options.mc3_chains,
                   switching=max(1,int(options.thinning/10.0)),
                   temperature_scale=options.temperature_scale)
    else:
        forwarders = [Forwarder.fromDirectory(arg) for arg in options.alignments]
        log_likelihood = Likelihood(IsolationMigrationEpochsModel(options.epochs,
                                                                  options.migration_states,
                                                                  options.ancestral_states),
                                    forwarders)
        mcmc = MCMC(priors, log_likelihood, thinning=options.thinning)

    with open(options.outfile, 'w') as outfile:
        print >> outfile, out_header
        
        if options.logfile:
            with open(options.logfile, 'w') as logfile:
                print >> logfile, log_header

                for _ in xrange(options.samples):
                    # Write main chain to output
                    params, prior, likelihood, posterior = mcmc.sample()
                    print >> outfile, '\t'.join(map(str, transform(options.epochs, params) +
                                                    (prior, likelihood, posterior)))
                    outfile.flush()

                    # All chains written to the log
                    for chain_no, chain in enumerate(mcmc.chains):
                        params = chain.current_theta
                        prior = chain.current_prior
                        likelihood = chain.current_likelihood
                        posterior = chain.current_posterior
                        print >> logfile, '\t'.join(map(str, (chain_no,) + transform(options.epochs, params) +
                                                        (prior, likelihood, posterior)))
                    logfile.flush()

        else:
            for _ in xrange(options.samples):
                params, prior, likelihood, posterior = mcmc.sample()
                print >> outfile, '\t'.join(map(str, transform(options.epochs, params) +
                                                (prior, likelihood, posterior)))
                outfile.flush()

    if options.mc3:
        mcmc.terminate()

if __name__ == '__main__':
    main()
