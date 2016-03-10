#!/usr/bin/env python

"""Script for estimating parameters in a three species isolation model.
"""

from argparse import ArgumentParser

from IMCoalHMM.ILS import ILSModel
from IMCoalHMM.likelihood import Likelihood, maximum_likelihood_estimate
from pyZipHMM import Forwarder


def transform(params):
    split_time_12, split_time_123, coal_rate_1, coal_rate_2, coal_rate_3, \
        coal_rate_12, coal_rate_123, recomb_rate = params
    return split_time_12, split_time_123, \
           2 / coal_rate_1, 2 / coal_rate_2, 2 / coal_rate_3, \
           2 / coal_rate_12, 2 / coal_rate_123, \
           recomb_rate


def main():
    """
    Run the main script.
    """
    usage = """%(prog)s [options] <forwarder dirs>

This program estimates the parameters of an isolation model with two species
and uniform coalescence and recombination rates."""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.1")

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

    parser.add_argument("--states-12",
                        type=int,
                        default=10,
                        help="Number of intervals used to discretize the time between the first and second speciation (10)")
                        
    parser.add_argument("--states-123",
                        type=int,
                        default=10,
                        help="Number of intervals used to discretize the time after the second speciation (10)")

    parser.add_argument("--optimizer",
                        type=str,
                        default="Nelder-Mead",
                        help="Optimization algorithm to use for maximizing the likelihood (Nealder-Mead)",
                        choices=['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC'])

    parser.add_argument("--has-outgroup",
                        action="store_true",
                        default=None,
                        help="Outgroup is included as fourth sequence in alignment.")

    optimized_params = [
        ('split-12', 'First split time in substitutions', 1e6 / 1e9),
        ('split-123', 'Second split time in substitutions', 1e6 / 1e9),
        ('theta-1', 'effective population size in 4Ne substitutions for species 1', 1e6 / 1e9),
        ('theta-2', 'effective population size in 4Ne substitutions for species 2', 1e6 / 1e9),
        ('theta-3', 'effective population size in 4Ne substitutions for species 3', 1e6 / 1e9),
        ('theta-12', 'effective population size in 4Ne substitutions for species 12 (first ancestral)', 1e6 / 1e9),
        ('theta-123', 'effective population size in 4Ne substitutions for species 123 (ancestral to all)', 1e6 / 1e9),
        ('rho', 'recombination rate in substitutions', 0.4),
        ('outgroup', 'total height of tree with outgroup', 1e6 / 1e9)
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


    init_parameters = (
        options.split_12,
        options.split_123,
        1 / (options.theta_1 / 2),
        1 / (options.theta_2 / 2),
        1 / (options.theta_3 / 2),
        1 / (options.theta_12 / 2),
        1 / (options.theta_123 / 2),
        options.rho
    )
    if options.has_outgroup: # FIXME: is this correct?
        init_parameters += (options.outgroup,)

    output_header = ['split.time.12', 'split.time.123',
                     'theta.1', 'theta.2', 'theta.3', 'theta.12', 'theta.123',
                     'rho']
    if options.outgroup:
        output_header.append("outgroup")

    forwarders = [Forwarder.fromDirectory(arg) for arg in options.alignments]   
    log_likelihood = Likelihood(ILSModel(options.states_12, options.states_123), forwarders)

    if options.logfile:
        with open(options.logfile, 'w') as logfile:

            if options.header:
                print >> logfile, '\t'.join(output_header)

            mle_parameters = maximum_likelihood_estimate(log_likelihood,
                                                         init_parameters,
                                                         optimizer_method=options.optimizer,
                                                         log_file=logfile,
                                                         log_param_transform=transform)
    else:
        mle_parameters = maximum_likelihood_estimate(log_likelihood, init_parameters,
                                                     optimizer_method=options.optimizer)

    max_log_likelihood = log_likelihood(mle_parameters)

    with open(options.outfile, 'w') as outfile:
        if options.header:
            print >> outfile, '\t'.join(output_header)
        print >> outfile, '\t'.join(map(str, transform(mle_parameters) + (max_log_likelihood,)))


if __name__ == '__main__':
    main()
