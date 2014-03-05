#!/usr/bin/env python

"""Script for estimating parameters in an PSMC like isolation model.


"""

from optparse import OptionParser

from IMCoalHMM.variable_coalescence_rate_isolation_model \
    import VariableCoalescenceRateIsolationModel, MinimizeWrapper
from IMCoalHMM.likelihood import Likelihood, maximum_likelihood_estimate
from pyZipHMM import Forwarder


def main():
    """
    Run the main script.
    """
    usage = """%prog [options] <forwarder dir>

This program estimates the parameters of an isolation model with two species
and uniform coalescence and recombination rates."""

    parser = OptionParser(usage=usage, version="%prog 1.0")

    parser.add_option("--isolation-model",
                      dest="est_split_time",
                      action="store_true",
                      default=False,
                      help="Estimate an isolation period before coalescences")

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
    ]

    for (cname, desc, default) in optimized_params:
        parser.add_option("--%s" % cname,
                          dest=cname,
                          type="float",
                          default=default,
                          help="Initial guess at the %s (%g)" % (desc, default))

    options, args = parser.parse_args()
    if len(args) != 1:
        parser.error("Input alignment not provided!")

    # get options
    split = options.split
    theta = options.theta
    rho = options.rho

    forwarder = Forwarder.fromDirectory(args[0])

    init_split = split
    init_coal = 1 / (theta / 2)
    init_recomb = rho

    log_likelihood = Likelihood(VariableCoalescenceRateIsolationModel(), forwarder)
    intervals = [4] + [2] * 25 + [4, 6]

    if options.est_split_time:
        minimizer = MinimizeWrapper(log_likelihood, intervals, True)
        initial_params = (init_split,) + (init_coal,) * 28 + (init_recomb,)

        if options.logfile:
            with open(options.logfile, 'w') as logfile:

                def transform(params):
                    split_time = params[0]
                    coal_rates = params[1:-1]
                    recomb_rate = params[-1]
                    return [split_time] + [2 / cr for cr in coal_rates] + [recomb_rate]

                mle_parameters = \
                    maximum_likelihood_estimate(minimizer,
                                                initial_params,
                                                log_file=logfile,
                                                log_param_transform=transform)
        else:  # no logging
            mle_parameters = \
                maximum_likelihood_estimate(minimizer, initial_params)

        with open(options.outfile, 'w') as outfile:
            print >> outfile, '\t'.join(map(str, mle_parameters))



    else:  # Not estimating split time
        minimizer = MinimizeWrapper(log_likelihood, intervals, False)
        initial_params = (init_coal,) * 28 + (init_recomb,)

        if options.logfile:
            with open(options.logfile, 'w') as logfile:

                def transform(params):
                    coal_rates = params[0:-1]
                    recomb_rate = params[-1]
                    return [2 / cr for cr in coal_rates] + [recomb_rate]

                mle_parameters = \
                    maximum_likelihood_estimate(minimizer,
                                                initial_params,
                                                log_file=logfile,
                                                log_param_transform=transform)
        else:  # no logging
            mle_parameters = \
                maximum_likelihood_estimate(minimizer, initial_params)

        with open(options.outfile, 'w') as outfile:
            print >> outfile, '\t'.join(map(str, mle_parameters))


if __name__ == '__main__':
    main()
