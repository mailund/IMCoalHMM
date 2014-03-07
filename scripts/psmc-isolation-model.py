#!/usr/bin/env python

"""Script for estimating parameters in an PSMC like isolation model.


"""

from argparse import ArgumentParser

from IMCoalHMM.variable_coalescence_rate_isolation_model \
    import VariableCoalescenceRateIsolationModel
from IMCoalHMM.likelihood import Likelihood, maximum_likelihood_estimate
from pyZipHMM import Forwarder


def transform_with_split(params):
    split_time = params[0]
    coal_rates = params[1:-1]
    recomb_rate = params[-1]
    return [split_time] + [2 / cr for cr in coal_rates] + [recomb_rate]


def transform_without_split(params):
    coal_rates = params[0:-1]
    recomb_rate = params[-1]
    return [2 / cr for cr in coal_rates] + [recomb_rate]


def main():
    """
    Run the main script.
    """
    usage = """%(prog)s [options] <forwarder dirs>

This program estimates the parameters of an isolation model with two species
and uniform coalescence and recombination rates."""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.1")

    parser.add_argument("--isolation-model",
                        action="store_true",
                        default=False,
                        help="Estimate an isolation period before coalescences")

    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/dev/stdout",
                        help="Output file for the estimate (/dev/stdout)")

    parser.add_argument("--logfile",
                        type=str,
                        default=None,
                        help="Log for all points estimated in the optimization")

    optimized_params = [
        ('split', 'split time in substitutions', 1e6 / 1e9),
        ('theta', 'effective population size in 4Ne substitutions', 1e6 / 1e9),
        ('rho', 'recombination rate in substitutions', 0.4),
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
    split = options.split
    theta = options.theta
    rho = options.rho

    forwarders = [Forwarder.fromDirectory(arg) for arg in options.alignments]

    init_split = split
    init_coal = 1 / (theta / 2)
    init_recomb = rho

    intervals = [4] + [2] * 25 + [4, 6]
    log_likelihood = Likelihood(VariableCoalescenceRateIsolationModel(intervals, options.isolation_model), forwarders)

    if options.isolation_model:
        initial_params = (init_split,) + (init_coal,) * 28 + (init_recomb,)

        if options.logfile:
            with open(options.logfile, 'w') as logfile:

                mle_parameters = \
                    maximum_likelihood_estimate(log_likelihood,
                                                initial_params,
                                                log_file=logfile,
                                                log_param_transform=transform_with_split)
        else:  # no logging
            mle_parameters = \
                maximum_likelihood_estimate(log_likelihood, initial_params)

        with open(options.outfile, 'w') as outfile:
            print >> outfile, '\t'.join(map(str, transform_with_split(mle_parameters)))

    else:  # Not estimating split time
        initial_params = (init_coal,) * 28 + (init_recomb,)

        if options.logfile:
            with open(options.logfile, 'w') as logfile:

                mle_parameters = \
                    maximum_likelihood_estimate(log_likelihood,
                                                initial_params,
                                                log_file=logfile,
                                                log_param_transform=transform_without_split)
        else:  # no logging
            mle_parameters = \
                maximum_likelihood_estimate(log_likelihood, initial_params)

        with open(options.outfile, 'w') as outfile:
            print >> outfile, '\t'.join(map(str, transform_without_split(mle_parameters)))


if __name__ == '__main__':
    main()
