#!/usr/bin/env python

"""Script for estimating parameters in an isolation model.
"""

from optparse import OptionParser

from IMCoalHMM.variable_migration_model import VariableCoalAndMigrationRateModel
from IMCoalHMM.likelihood import Likelihood, maximum_likelihood_estimate
from pyZipHMM import Forwarder


def main():
    """
    Run the main script.
    """
    usage = """%prog [options] <forwarder dirs>

This program estimates the coalescence and migration rates over time together with a constant
recombination rate."""

    parser = OptionParser(usage=usage, version="%prog 1.0")

    parser.add_option("-o", "--out",
                      dest="outfile",
                      type="string",
                      default="/dev/stdout",
                      help="Output file for the estimate (/dev/stdout)")

    parser.add_option("--logfile",
                      dest="logfile",
                      type="string",
                      default=None,
                      help="Log for all points estimated in the optimization")

    optimized_params = [
        ('split', 'split time in substitutions', 1e6 / 1e9),
        ('theta', 'effective population size in 4Ne substitutions', 1e6 / 1e9),
        ('rho', 'recombination rate in substitutions', 0.4),
        ('migration-rate', 'migration rate in number of migrations per substitution', 100.0),
    ]

    for parameter_name, description, default in optimized_params:
        parser.add_option("--%s" % parameter_name,
                          dest=parameter_name.replace('-', '_'),
                          type="float",
                          default=default,
                          help="Initial guess at the %s (%g)" % (description, default))

    options, args = parser.parse_args()
    if len(args) != 3:  # FIXME: How do I specify more alignments and split them in three?
        parser.error("Input alignment not provided!")

    # get options
    theta = options.theta
    rho = options.rho

    init_coal = 1 / (theta / 2)
    init_mig = options.migration_rate
    init_recomb = rho


    # FIXME: I don't know what would be a good choice here...
    # intervals = [4] + [2] * 25 + [4, 6]
    intervals = [5, 5, 5, 5]
    no_epochs = len(intervals)
    initial_parameters = (init_coal,) * 2 * no_epochs + (init_mig,) * 2 * no_epochs + (init_recomb,)


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
    # FIXME: pick the three types of alignments
    forwarders_11 = [Forwarder.fromDirectory(args[0])]
    forwarders_12 = [Forwarder.fromDirectory(args[1])]
    forwarders_22 = [Forwarder.fromDirectory(args[2])]

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
