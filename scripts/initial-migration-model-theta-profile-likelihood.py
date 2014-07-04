#!/usr/bin/env python

"""Script for estimating parameters in an isolation model.
"""

from argparse import ArgumentParser
import scipy

from IMCoalHMM.likelihood import Likelihood
from IMCoalHMM.isolation_with_migration_model import IsolationMigrationModel
from pyZipHMM import Forwarder


def transform(params):
    """
    Translate the parameters to the input and output parameter space.
    """
    isolation_time, migration_time, coal_rate, recomb_rate, mig_rate = params
    return isolation_time, migration_time, 2 / coal_rate, recomb_rate, mig_rate


def main():
    """
    Run the main script.
    """
    usage = """%(prog)s [options] <forwarder dirs>

This program estimates the parameters of an initial-migration model with two species
and uniform coalescence and recombination rates and outputs the profile likelihood for
the theta parameter."""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.0")

    parser.add_argument("--header",
                        action="store_true",
                        default=False,
                        help="Include a header on the output")
    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/dev/stdout",
                        help="Output file for the estimate (/dev/stdout)")

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

    parser.add_argument('--theta-start', type=float, required=True,
                        help="First point in the range to compute likelihood for.")
    parser.add_argument('--theta-end', type=float, required=True,
                        help="Last point in the range to compute likelihood for.")
    parser.add_argument('--number-of-points', type=int, default=50,
                        help="Number of points to compute the profile likelihood in.")

    optimized_params = [
        ('isolation-period', 'time where the populations have been isolated', 1e6 / 1e9),
        ('migration-period', 'time period where the populations exchanged genes', 1e6 / 1e9),
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

    forwarders = [Forwarder.fromDirectory(arg) for arg in options.alignments]

    init_isolation_time = options.isolation_period
    init_migration_time = options.migration_period
    theta_points = scipy.linspace(options.theta_start, options.theta_end, options.number_of_points)

    rho = options.rho
    init_recomb = rho
    init_migration = options.migration_rate

    log_likelihood = Likelihood(IsolationMigrationModel(no_migration_states, no_ancestral_states), forwarders)
    initial_parameters = (init_isolation_time, init_migration_time, init_recomb, init_migration)

    def make_minimize_wrapper(coal_rate):
        def wrapper(params):
            parameters = params[0], params[1], coal_rate, params[2], params[3]
            return - log_likelihood(scipy.array(parameters))
        return wrapper

    with open(options.outfile, 'w') as outfile:
        if options.header:
            print >> outfile, '\t'.join(['isolation.period', 'migration.period',
                                         'theta', 'rho', 'migration', 'log.likelihood'])

        for theta in theta_points:
            coal_rate = 1 / (theta / 2)
            minimize_wrapper = make_minimize_wrapper(coal_rate)
            optimized_results = scipy.optimize.minimize(fun=minimize_wrapper,
                                                        x0=initial_parameters,
                                                        method=options.optimizer)

            isolation_period, migration_period, rho, mig_rate = optimized_results.x
            mle_parameters = [isolation_period, migration_period, coal_rate, rho, mig_rate]
            log_likelihood = -optimized_results.fun
            print >> outfile, '\t'.join(map(str, transform(mle_parameters) + (log_likelihood,)))


if __name__ == '__main__':
    main()
