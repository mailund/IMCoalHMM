'''Script for estimating parameters in an isolation model.
'''

from optparse import OptionParser

from IMCoalHMM.isolation_model import IsolationModel, MinimizeWrapper
from IMCoalHMM.likelihood import Likelihood, maximum_likelihood_estimate
from pyZipHMM import Forwarder

def main():
    usage="""%prog [options] <forwarder dir>

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
                      
    parser.add_option("--states",
                      dest="states",
                      type="int",
                      default=10,
                      help="Number of intervals used to discretize the time (10)")

    optimized_params = [
            ('split', 'split time in substitutions', 1e6/1e9),
            ('theta', 'effective population size in 4Ne substitutions', 1e6/1e9),
            ('rho', 'recombination rate in substitutions', 0.4),
            ]

    for (cname, desc, default) in optimized_params:
        parser.add_option("--%s" % cname,
                          dest=cname,
                          type="float",
                          default=default,
                          help="Initial guess at the %s (%g)" % (desc, default))

    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.error("Input alignment not provided!")

    # get options
    no_states = options.states
    split = options.split
    theta = options.split
    rho = options.rho
    
    forwarder = Forwarder.fromDirectory(args[0])
    
    init_split = split
    init_coal = 1/(theta/2)
    init_recomb = rho
    
    logL = Likelihood(IsolationModel(), forwarder)

    if options.logfile:
        with open(options.logfile, 'w') as logfile:

            if options.include_header:
                print >>logfile, '\t'.join(['split.time', 'theta', 'rho', 'logL'])

            def transform(params):
                split, coal_rate, recomb_rate = params
                return split, 2/coal_rate, recomb_rate

            mle_split_time, mle_coal_rate, mle_recomb_rate = \
                maximum_likelihood_estimate(MinimizeWrapper(logL, no_states),
                                            (init_split, init_coal, init_recomb),
                                            log_file = logfile,
                                            log_param_transform = transform)
    else:
        mle_split_time, mle_coal_rate, mle_recomb_rate = \
                maximum_likelihood_estimate(MinimizeWrapper(logL, no_states),
                                            (init_split, init_coal, init_recomb))

    maxL = logL(no_states, mle_split_time, mle_coal_rate, mle_recomb_rate)

    mle_theta = 2/mle_coal_rate
    
    with open(options.outfile, 'w') as outfile:
        if options.include_header:
            print >>outfile, '\t'.join(['split.time', 'theta', 'rho', 'logL'])
        print >>outfile, '\t'.join(map(str,[mle_split_time, 
                                            mle_theta,
                                            mle_recomb_rate,
                                            maxL]))
    

if __name__ == '__main__':
    main()
