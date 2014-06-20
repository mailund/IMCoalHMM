#!/usr/bin/env python

"""Script for estimating parameters in an isolation model.
"""

from argparse import ArgumentParser

from IMCoalHMM.variable_migration_model import VariableCoalAndMigrationRateModel
from IMCoalHMM.likelihood import Likelihood, maximum_likelihood_estimate
from IMCoalHMM.mcmc import MC3, LogNormPrior, ExpLogNormPrior
from pyZipHMM import Forwarder


def main():
    """
    Run the main script.
    """
    usage = """%(prog)s [options] <forwarder dirs>

This program estimates the coalescence and migration rates over time together with a constant
recombination rate."""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.0")

    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/dev/stdout",
                        help="Output file for the estimate (/dev/stdout)")

    parser.add_argument("--logfile",
                        type=str,
                        default=None,
                        help="Log for all points estimated in the optimization")

    meta_params = [
        ('theta', 'effective population size in 4Ne substitutions', 1e6 / 1e9),
        ('rho', 'recombination rate in substitutions', 0.4),
        ('migration-rate', 'migration rate in number of migrations per substitution', 100.0),
    ]

    for parameter_name, description, default in meta_params:
        parser.add_argument("--%s" % parameter_name,
                            type=float,
                            default=default,
                            help="Meta-parameter for the %s prior (default %g)" % (description, default))

    parser.add_argument("-n", "--samples",
                        type=int,
                        default=500,
                        help="Number of samples to draw (500)")

    parser.add_argument("-k", "--thinning",
                        type=int,
                        default=100,
                        help="Number of MCMC steps between samples (100)")

    parser.add_argument("--mc3-chains", type=int, default=3, help="Number of MCMCMC chains")
    parser.add_argument("--temperature-scale", type=float, default=1.0,
                        help="The scale by which higher chains will have added temperature.")

    parser.add_argument('-a11', '--alignments11', nargs='+',
                        help='Alignments of two sequences from the first population')
    parser.add_argument('-a12', '--alignments12', nargs='+',
                        help='Alignments of two sequences, one from each population')
    parser.add_argument('-a22', '--alignments22', nargs='+',
                        help='Alignments of two sequences from the second population')

    options = parser.parse_args()

    if len(options.alignments11) < 1:
        parser.error("Input alignment for the 11 system not provided!")
    if len(options.alignments12) < 1:
        parser.error("Input alignment for the 12 system not provided!")
    if len(options.alignments22) < 1:
        parser.error("Input alignment for the 22 system not provided!")

    coal_prior = LogNormPrior(log(1/(options.theta/2)))
    rho_prior = LogNormPrior(log(options.rho))
    migration_rate_prior = ExpLogNormPrior(options.migration_rate)

    # FIXME: I don't know what would be a good choice here...
    # intervals = [4] + [2] * 25 + [4, 6]
    intervals = [5, 5, 5, 5]
    no_epochs = len(intervals)
    priors = (coal_prior,) * 2 * no_epochs + (migration_rate_prior,) * 2 * no_epochs + (rho_prior,)

    def transform(parameters):
        coal_rates_1 = tuple(parameters[0:no_epochs])
        coal_rates_2 = tuple(parameters[no_epochs:(2 * no_epochs)])
        mig_rates_12 = tuple(parameters[(2 * no_epochs):(3 * no_epochs)])
        mig_rates_21 = tuple(parameters[(3 * no_epochs):(4 * no_epochs)])
        recomb_rate = parameters[-1]
        theta_1 = tuple([2 / coal_rate for coal_rate in coal_rates_1])
        theta_2 = tuple([2 / coal_rate for coal_rate in coal_rates_2])
        return theta_1 + theta_2 + mig_rates_12 + mig_rates_21 + (recomb_rate,)

    # load alignments
    forwarders_11 = [Forwarder.fromDirectory(alignment) for alignment in options.alignments11]
    forwarders_12 = [Forwarder.fromDirectory(alignment) for alignment in options.alignments12]
    forwarders_22 = [Forwarder.fromDirectory(alignment) for alignment in options.alignments22]

    model_11 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_11, intervals)
    model_12 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_12, intervals)
    model_22 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_22, intervals)
    log_likelihood_11 = Likelihood(model_11, forwarders_11)
    log_likelihood_12 = Likelihood(model_12, forwarders_12)
    log_likelihood_22 = Likelihood(model_22, forwarders_22)

    def log_likelihood(parameters):
        return log_likelihood_11(parameters) + log_likelihood_12(parameters) + log_likelihood_22(parameters)

    if options.logfile:
        with open(options.logfile, 'w') as logfile:
            mle_parameters = maximum_likelihood_estimate(log_likelihood, initial_parameters,
                                                         log_file=logfile,
                                                         log_param_transform=transform)
    else:
        mle_parameters = maximum_likelihood_estimate(log_likelihood, initial_parameters)

    max_log_likelihood = log_likelihood(mle_parameters)

    with open(options.outfile, 'w') as outfile:
        print >> outfile, '\t'.join(map(str, transform(mle_parameters) + (max_log_likelihood,)))


if __name__ == '__main__':
    main()
