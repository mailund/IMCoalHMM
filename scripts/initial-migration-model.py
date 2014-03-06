#!/usr/bin/env python

"""Script for estimating parameters in an isolation model.
"""

from optparse import OptionParser

from IMCoalHMM.likelihood import Likelihood, maximum_likelihood_estimate
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
    usage = """%prog [options] <forwarder dirs>

This program estimates the parameters of an isolation model with two species
and uniform coalescence and recombination rates."""

    parser = OptionParser(usage=usage, version="%prog 1.0")

    parser.add_option("--header",
                      dest="include_header",
                      action="store_true",
                      default=False,
                      help="Include a header on the output")
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

    parser.add_option("--ancestral-states",
                      dest="ancestral_states",
                      type="int",
                      default=10,
                      help="Number of intervals used to discretize the time in the ancestral population (10)")
    parser.add_option("--migration-states",
                      dest="migration_states",
                      type="int",
                      default=10,
                      help="Number of intervals used to discretize the time in the migration period (10)")

    optimized_params = [
        ('isolation-period', 'time where the populations have been isolated', 1e6 / 1e9),
        ('migration-period', 'time period where the populations exchanged genes', 1e6 / 1e9),
        ('theta', 'effective population size in 4Ne substitutions', 1e6 / 1e9),
        ('rho', 'recombination rate in substitutions', 0.4),
        ('migration-rate', 'migration rate in number of migrations per substitution', 200.0)
    ]

    for (cname, desc, default) in optimized_params:
        parser.add_option("--%s" % cname,
                          dest=cname.replace('-', '_'),
                          type="float",
                          default=default,
                          help="Initial guess at the %s (%g)" % (desc, default))

    options, args = parser.parse_args()
    if len(args) < 1:
        parser.error("Input alignment not provided!")

    # get options
    no_migration_states = options.migration_states
    no_ancestral_states = options.ancestral_states
    theta = options.theta
    rho = options.rho

    forwarders = [Forwarder.fromDirectory(arg) for arg in args]

    init_isolation_time = options.isolation_period
    init_migration_time = options.migration_period
    init_coal = 1 / (theta / 2)
    init_recomb = rho
    init_migration = options.migration_rate

    log_likelihood = Likelihood(IsolationMigrationModel(no_migration_states, no_ancestral_states), forwarders)
    initial_parameters = (init_isolation_time, init_migration_time, init_coal, init_recomb, init_migration)

    if options.logfile:
        with open(options.logfile, 'w') as logfile:

            if options.include_header:
                print >> logfile, '\t'.join(['isolation.period', 'migration.period',
                                             'theta', 'rho', 'migration'])

            mle_parameters = \
                maximum_likelihood_estimate(log_likelihood, initial_parameters,
                                            log_file=logfile,
                                            log_param_transform=transform)
    else:
        mle_parameters = \
            maximum_likelihood_estimate(log_likelihood, initial_parameters)

    max_log_likelihood = log_likelihood(mle_parameters)
    with open(options.outfile, 'w') as outfile:
        if options.include_header:
            print >> outfile, '\t'.join(['isolation.period', 'migration.period',
                                         'theta', 'rho', 'migration', 'logL'])
        print >> outfile, '\t'.join(map(str, transform(mle_parameters) + (max_log_likelihood,)))


if __name__ == '__main__':
    main()
