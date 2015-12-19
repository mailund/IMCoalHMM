#!/usr/bin/env python

"""Script for estimating parameters in an isolation model.
"""

from argparse import ArgumentParser

from IMCoalHMM.isolation_model import IsolationModel
from IMCoalHMM.likelihood import Likelihood, maximum_likelihood_estimate
from pyZipHMM import Forwarder
import isolation_model2


def transform(params):
    if len(params)==4:
        split_time, coal_rate, recomb_rate, outgroup = params
        return split_time, 2 / coal_rate, recomb_rate, outgroup
    split_time, coal_rate, recomb_rate = params
    return split_time, 2 / coal_rate, recomb_rate


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

    parser.add_argument("--states",
                        type=int,
                        default=10,
                        help="Number of intervals used to discretize the time (10)")
                        
    parser.add_argument("--optimizer",
                        type=str,
                        default="Nelder-Mead",
                        help="Optimization algorithm to use for maximizing the likelihood (Nealder-Mead)",
                        choices=['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC'])

    parser.add_argument("--verbose",
                        default=False,
                        action='store_true')                        
                        
    parser.add_argument("--emissionComplicated", default=False, action="store_true", help="This will use an emission matrix which is not an approximation.")
    parser.add_argument('--outgroup', action='store_true', default=False, help="This indicates that the alignemnts are not pairwise but threewise and that the last entry will be ")

    
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
    if options.outgroup and not options.emissionComplicated :
        parser.error("You can't have an outgroup without the complicated emission probabilities!")

    # get options
    no_states = options.states
    split = options.split
    theta = options.theta
    rho = options.rho

    init_split = split
    init_coal = 1 / (theta / 2)
    init_recomb = rho
    if options.outgroup:
        init_outgroup = init_split*3



    forwarders = [Forwarder.fromDirectory(arg) for arg in options.alignments]
    
    if options.emissionComplicated:
        if options.outgroup:
            basic_log_likelihood = Likelihood(isolation_model2.IsolationModel(no_states, outgroup=True), forwarders)
        else:
            basic_log_likelihood = Likelihood(isolation_model2.IsolationModel(no_states, outgroup=False), forwarders)    
    else:
        basic_log_likelihood = Likelihood(IsolationModel(no_states), forwarders)
        
    if options.verbose:
        print "--verbose option activated"
        def log_likelihood(params):
            val=basic_log_likelihood(params)
            print str(params)+"="+str(val)
            return val
    else:
        log_likelihood=basic_log_likelihood

    if options.logfile:
        with open(options.logfile, 'w') as logfile:

            if options.header:
                print >> logfile, '\t'.join(['split.time', 'theta', 'rho'])
            if options.outgroup and options.emissionComplicated:
                mle_parameters = maximum_likelihood_estimate(log_likelihood,
                                                             (init_split, init_coal, init_recomb, init_outgroup),
                                                             optimizer_method=options.optimizer,
                                                             log_file=logfile,
                                                             log_param_transform=transform)
            else:
                mle_parameters = maximum_likelihood_estimate(log_likelihood,
                                                             (init_split, init_coal, init_recomb),
                                                             optimizer_method=options.optimizer,
                                                             log_file=logfile,
                                                             log_param_transform=transform)
    else:
        if options.outgroup and options.emissionComplicated:
            mle_parameters = maximum_likelihood_estimate(log_likelihood, (init_split, init_coal, init_recomb, init_outgroup),
                                                     optimizer_method=options.optimizer)
        else:
            mle_parameters = maximum_likelihood_estimate(log_likelihood, (init_split, init_coal, init_recomb),
                                                     optimizer_method=options.optimizer)

    max_log_likelihood = log_likelihood(mle_parameters)

    with open(options.outfile, 'w') as outfile:
        if options.header:
            print >> outfile, '\t'.join(['split.time', 'theta', 'rho', 'log.likelihood'])
        print >> outfile, '\t'.join(map(str, transform(mle_parameters) + (max_log_likelihood,)))


if __name__ == '__main__':
    main()
