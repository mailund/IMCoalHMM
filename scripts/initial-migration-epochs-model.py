#!/usr/bin/env python

"""Script for estimating parameters in an initial migration model.
"""

from argparse import ArgumentParser

from IMCoalHMM.likelihood import Likelihood, maximum_likelihood_estimate
from IMCoalHMM.isolation_with_migration_model_epochs import IsolationMigrationEpochsModel
from pyZipHMM import Forwarder


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
    usage = """%(prog)s [options] <forwarder dirs>

This program estimates the parameters of an isolation model with an initial migration period with two species
and uniform coalescence and recombination rates."""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.0")

    parser.add_argument("--header",
                        action="store_true",
                        default=False,
                        help="Include a header on the output")
    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/dev/stdout",
                        help="Output file for the estimate (/dev/stdout)")

    parser.add_argument("--logfile",
                        type=str,
                        default=None,
                        help="Log for all points estimated in the optimization")


    parser.add_argument("--epochs",
                        type=int,
                        default=2,
                        help="Number of epochs (variation in population size and migration rate) used " \
                             "in the migration period and in the ancestral population (2)")

    parser.add_argument("--ancestral-states",
                        type=int,
                        default=10,
                        help="Number of intervals used to discretize the time in the ancestral population (10)")
    parser.add_argument("--migration-states",
                        type=int,
                        default=10,
                        help="Number of intervals used to discretize the time in the migration period (10)")

    parser.add_argument("--optimizer",
                        type=str,
                        default="Nelder-Mead",
                        help="Optimization algorithm to use for maximizing the likelihood (Nealder-Mead)",
                        choices=['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC'])

    optimized_params = [
        ('isolation-period', 'time where the populations have been isolated', 1e6 / 1e9),
        ('migration-period', 'time period where the populations exchanged genes', 1e6 / 1e9),
        ('theta', 'effective population size in 4Ne substitutions', 1e6 / 1e9),
        ('rho', 'recombination rate in substitutions', 0.4),
        ('migration-rate', 'migration rate in number of migrations per substitution', 200.0)
    ]

    for parameter_name, description, default in optimized_params:
        parser.add_argument("--%s" % parameter_name,
                            type=float,
                            default=default,
                            help="Initial guess at the %s (%g)" % (description, default))

    parser.add_argument('alignments', nargs='+', help='Alignments in ZipHMM format')

    options = parser.parse_args()
    if len(options.alignments) < 1:
        parser.error("Input alignment not provided!")

    # get options
    no_migration_states = options.migration_states
    no_ancestral_states = options.ancestral_states
    theta = options.theta
    rho = options.rho

    forwarders = [Forwarder.fromDirectory(arg) for arg in options.alignments]

    init_isolation_time = options.isolation_period
    init_migration_time = options.migration_period
    init_coal = 1 / (theta / 2)
    init_recomb = rho
    init_migration = options.migration_rate

    log_header = '\t'.join(['isolation.period', 'migration.period'] +
                           ['isolation.theta'] +
                           ['migration.theta.{}'.format(epoch) for epoch in range(options.epochs)] +
                           ['ancestral.theta.{}'.format(epoch) for epoch in range(options.epochs)] +
                           ['rho'] +
                           ['migration.{}'.format(epoch) for epoch in range(options.epochs)])
    out_header = "{}\tlog.likelihood".format(log_header)

    log_likelihood = Likelihood(IsolationMigrationEpochsModel(options.epochs, no_migration_states, no_ancestral_states), forwarders)
    initial_parameters = (init_isolation_time, init_migration_time, init_recomb) + \
                         (init_coal,) * (2 * options.epochs + 1) + (init_migration,) * options.epochs

    if options.logfile:
        with open(options.logfile, 'w') as logfile:

            if options.header:
                print >> logfile, log_header

            mle_parameters = \
                maximum_likelihood_estimate(log_likelihood, initial_parameters,
                                            log_file=logfile, optimizer_method=options.optimizer,
                                            log_param_transform=lambda p: transform(options.epochs, p))
    else:
        mle_parameters = \
            maximum_likelihood_estimate(log_likelihood, initial_parameters,
                                        optimizer_method=options.optimizer)

    max_log_likelihood = log_likelihood(mle_parameters)
    with open(options.outfile, 'w') as outfile:
        if options.header:
            print >> outfile, out_header
        print >> outfile, '\t'.join(map(str, transform(options.epochs, mle_parameters) + (max_log_likelihood,)))


if __name__ == '__main__':
    main()
