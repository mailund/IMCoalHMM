#!/usr/bin/env python

"""Script for estimating parameters in an initial migration model.
"""

from perfectLikelihood import Coal_times_log_lik
from newick_count import count_tmrca
from argparse import ArgumentParser

from IMCoalHMM.likelihood import Likelihood
#from IMCoalHMM.isolation_with_migration_model import IsolationMigrationModel
from pyZipHMM import Forwarder

from isolation_with_migration_model2 import IsolationMigrationModelConstantBreaks
from isolation_with_migration_model2 import IsolationMigrationModel# as IsolationMigrationModel2

from IMCoalHMM.mcmc import MCMC, MC3
from mcmc3 import MC3 as MC3adap
from mcmc3 import LogNormPrior, ExpLogNormPrior

from global_scaling import Global_scaling
from alg4_scaling import AM4_scaling
from operator import itemgetter
from copy import deepcopy


from math import log,exp

from numpy import array, dot

import sys


def printPyZipHMM(Matrix):
    """
    This is copied from variable-migration-model to make a test
    """
    finalString=""
    for i in range(Matrix.getWidth()):
        for j in range(Matrix.getHeight()):
            finalString=finalString+" "+str(Matrix[i,j])
        finalString=finalString+"\n"
    return finalString

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
    usage = """%(prog)s [options] alignments...

This program estimates the parameters of an isolation model with an initial migration period with two species
and uniform coalescence and recombination rates."""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.7")

    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/dev/stdout",
                        help="Output file for the estimate (/dev/stdout)")

    parser.add_argument("--logfile",
                        type=str,
                        default=None,
                        help="Log for sampled points in all chains for the MCMCMC during the run." \
                             "This parameter is only valid when running --mc3.")

    parser.add_argument("--ancestral-states",
                        type=int,
                        default=10,
                        help="Number of intervals used to discretize the time in the ancestral population (10)")
    
    parser.add_argument("--migration-states",
                        type=int,
                        default=10,
                        help="Number of intervals used to discretize the time in the migration period (10)")

    parser.add_argument("-n", "--samples",
                        type=int,
                        default=500,
                        help="Number of samples to draw (500)")
    
    parser.add_argument("--transform", type=int, default=1, help="the transformation to use. 1 is dependent transformation and 2 is independent transformation(scaling)")


    parser.add_argument("--mc3", help="Run a Metropolis-Coupled MCMC", action="store_true")
    parser.add_argument("--mc3-chains", type=int, default=3, help="Number of MCMCMC chains")
    parser.add_argument("--temperature-scale", type=float, default=10.0,
                        help="The scale by which higher chains will have added temperature." \
                             "Chain i will have temperature scale*i.")
    parser.add_argument('--mc3_switching', type=int, default=1, help='the number of switches per thinning period')
    parser.add_argument('--mc3_jump_accept', type=float, default=0.234, help='the swap acceptance probability that the chain is adapting against. Default is 0.234.')
    parser.add_argument('--mc3_flip_suggestions', type=int, default=1, help='The number of times after each step a flip is suggested. It has to be at least one, default is one.')
    parser.add_argument('--mc3_sort_chains', action='store_true', default=False, help='If announced, this will sort each chain according to its posterior value, making it closer to the stationary')
    parser.add_argument('--mc3_fixed_temp_max', default=None, type=float, help='If applied, this will make the temperature gaps equally crossable and the maximum is said parameter.')

    
    
    
    parser.add_argument('--adap', default=0, type=int, help='this number tells what adaption to use. Options=1,3. ')
    parser.add_argument('--adap_step_size', default=1.0, type=float, help='this number is the starting step size of the adaption')
    parser.add_argument('--adap_step_size_marginal', default=0.1, type=float, help='this number is the starting step size of the adaption of the marginals')
    parser.add_argument('--adap_harmonic_power', default=0.5, type=float, help='this number is the power of the harmonical decrease in scew with adaption. It tells how fast the adaption vanishes.')
    parser.add_argument('--adap_desired_accept', default=0.234, type=float, help='this number is the acceptance rate that the adaptive algorithm strives for')
    parser.add_argument('--adap3_correlates_begin', default=100, type=int, help='In adaption scheme 3, this chooses when we should start simulate proposals using the empirical covariance.')
    parser.add_argument('--adap3_tracking_begin', default=50, type=int, help='In adaption scheme 3, this chooses when we should start simulate proposals using the empirical covariance.')
    parser.add_argument('--adap3_from_identical', default=0.2, type=float, help='How big proportion of the time after correlates_begin will we suggest independents with same variance.')
    parser.add_argument('--adap3_from_independent', default=0, type=float, help='Will we not use the correlates. If stated the covariance matrix will be estimated without off-diagonal entries.')

    parser.add_argument('--constant_break_points', default=False, action="store_true", help='If enabled, the break points will be fixed throughout the analysis but the epochs will change')
    parser.add_argument('--breakpoints_tail_pieces', default=0, type=int, help='this produce a tail of last a number of pieces on the breakpoints')
    parser.add_argument('--breakpoints_time', default=1.0, type=float, help='this number moves the breakpoints up and down. Smaller values will give sooner timeperiods.')

    
    parser.add_argument("-k", "--thinning",
                        type=int,
                        default=100,
                        help="Number of MCMC steps between samples (100)")

    parser.add_argument("--sample-priors", help="Sample independently from the priors", action="store_true")
    parser.add_argument("--mcmc-priors", help="Run the MCMC but use the prior as the posterior", action="store_true")
    parser.add_argument("--sd_multiplyer", type=float, default=0.2, help="The proportion each proposal suggest changes of all its variance(defined by the transformToI and transformFromI)")
    parser.add_argument("--mixture", help="Every third is sampled from the standard distribution where log increments are independent N(0,sd=0.1)", action="store_true")

    parser.add_argument('--treefile', type=str, help='File containing newick formats of the trees to use as input')
    parser.add_argument('--use_trees_as_data', action='store_true', help='if so, the program will use trees as input data instead of alignments')


    meta_params = [
        ('isolation-period', 'time where the populations have been isolated', 1e6 / 1e9),
        ('migration-period', 'time period where the populations exchanged genes', 1e6 / 1e9),
        ('theta', 'effective population size in 4Ne substitutions', 1e6 / 1e9),
        ('rho', 'recombination rate in substitutions', 0.4),
        ('migration-rate', 'migration rate in number of migrations per substitution', 200.0),
        ('Ngmu4', 'substitutions pr 4ngmu year', 4*25*20000*1e-9)
    ]

    for parameter_name, description, default in meta_params:
        parser.add_argument("--%s" % parameter_name,
                            type=float,
                            default=default,
                            help="Meta-parameter mean of the %s (%g)" % (description, default))

    parser.add_argument('alignments', nargs='*', help='Alignments in ZipHMM format')

    options = parser.parse_args()
    if not options.use_trees_as_data:
        if len(options.alignments) < 1 and not (options.sample_priors or options.mcmc_priors):
            parser.error("Input alignment not provided!")

        if len(options.alignments) > 0 and options.mcmc_priors:
            parser.error("You should not provide alignments to the command line when sampling from the prior")

        if options.logfile and not options.mc3:
            parser.error("the --logfile option is only valid together with the --mc3 option.")

#     def transformToI(inarray):
#         inarray=map(log,inarray)
#         tmp= dot(array(    [[4.9875799, -1.3487765,      3.1201993,      1.2686557,     -5.7025884], \
#                                 [0,          9.9498989,      10.2635097,    -0.0162581,     -2.8033441],\
#                                 [0,          0,              4.2651228,      7.1733158,     -2.9267196],\
#                                 [0,          0,              0,              6.2813348,      1.3268374],\
#                                 [0,          0,              0,              0,              7.8601317]]).transpose(), inarray) 
#         return map(exp,tmp)
#     
#     def transformFromI(inarray):
#         inarray=map(log,inarray)
#         tmp=dot(array([[0.20049804,  0.02717887, -0.21207935,  0.20177098,  0.04212849],\
#                            [0,           0.10050353, -0.24184978,  0.27645379, -0.10087487],\
#                            [0,           0,           0.23445984, -0.26775431,  0.13249964],\
#                            [0,           0,           0,           0.15920183, -0.02687422],\
#                            [0,           0,           0,           0,           0.12722433]]).transpose(), inarray)
#         return map(exp,tmp)
#     
#     def transformToI2(inarray):
#         inarray=map(log,inarray)
#         tmp= dot(array(    [[4.982471, 0,      0,      0,     0], \
#                                 [0,          6.749068,      0,    0,     0],\
#                                 [0,          0,              2.457224,      0,     0],\
#                                 [0,          0,              0,              2.238575,      0],\
#                                 [0,          0,              0,              0,              2.874258]]).transpose(), inarray) 
#         return map(exp,tmp)
#     
#     def transformFromI2(inarray):
#         inarray=map(log,inarray)
#         tmp=dot(array([[0.2007036,  0, 0,  0,  0],\
#                            [0,           0.1481686, 0,  0, 0],\
#                            [0,           0,           0.4069633, 0,  0],\
#                            [0,           0,           0,           0.4467127, 0],\
#                            [0,           0,           0,           0,           0.3479159]]).transpose(), inarray)
#         return map(exp,tmp)
    
    # Specify priors and proposal distributions... 
    # I am sampling in log-space to make it easier to make a random walk
    means=array([options.isolation_period, options.migration_period,1/(options.theta/2), options.rho, options.migration_rate])
    isolation_period_prior = LogNormPrior(log(means[0]), proposal_sd=options.sd_multiplyer)
    migration_period_prior = LogNormPrior(log(means[1]), proposal_sd=options.sd_multiplyer)
    coal_prior = LogNormPrior(log(means[2]), proposal_sd=options.sd_multiplyer)
    rho_prior = LogNormPrior(log(means[3]), proposal_sd=options.sd_multiplyer)
    migration_rate_prior = ExpLogNormPrior(means[4], proposal_sd=options.sd_multiplyer)
    priors = [isolation_period_prior, migration_period_prior,
              coal_prior, rho_prior, migration_rate_prior]
    
    no_params=5
    
    
    if options.constant_break_points:
        model = IsolationMigrationModelConstantBreaks(options.migration_states+ options.ancestral_states, 
                                                      breaktail=options.breakpoints_tail_pieces,breaktimes=options.breakpoints_time)
    else:
        model= IsolationMigrationModel(options.migration_states, options.ancestral_states)
        
    forwarders = [Forwarder.fromDirectory(arg) for arg in options.alignments]
    
    base_log_likelihood= Likelihood(model, forwarders)
    def log_likelihood(params):
        return 0,0,base_log_likelihood(params)
        
    print log_likelihood(array([0.2]*5))

    # If we only want to sample from the priors we simply collect random points from these
    if options.sample_priors:
        with open(options.outfile, 'w') as outfile:
            print >> outfile, '\t'.join(['isolation.period', 'migration.period',
                                         'theta', 'rho', 'migration',
                                         'log.prior', 'log.likelihood', 'log.posterior'])
            for _ in xrange(options.samples):
                params = [prior.sample() for prior in priors]
                prior = sum(log(prior.pdf(p)) for prior, p in zip(priors, params))
                likelihood = 0.0
                posterior = prior + likelihood
                print >> outfile, '\t'.join(map(str, transform(params) + (prior, likelihood, posterior)))
                outfile.flush()

        sys.exit(0) # Successful termination

    if options.mc3:
        if options.adap==1 or options.adap==3:
            if options.adap==1:
                adap = Global_scaling(params=[options.adap_harmonic_power, options.adap_step_size], alphaDesired=options.adap_desired_accept)

            elif options.adap==3:
                toTakeMaxFrom=[1-options.adap3_from_identical-options.adap3_from_independent, options.adap3_from_independent,options.adap3_from_identical]
                max_index,_ = max(enumerate(toTakeMaxFrom), key=itemgetter(1))
                adap=AM4_scaling(startVal=no_params*[1.0], 
                     params=[options.adap_harmonic_power, options.adap_step_size, 
                                                       (options.adap3_tracking_begin, options.adap3_correlates_begin),
                                                       (options.adap3_from_identical,options.adap3_from_independent), max_index], 
                     alphaDesired=options.adap_desired_accept)
            chain_structure=[1]*options.mc3_chains
            adapts=[] #we have to make a list of adaptors
            for _ in range(options.mc3_chains):
                adapts.append(deepcopy(adap))
            mcmc = MC3adap(priors, log_likelihood=log_likelihood, accept_jump=options.mc3_jump_accept, flip_suggestions=options.mc3_flip_suggestions,
                sort=options.mc3_sort_chains, chain_structure=chain_structure, thinning=options.thinning, switching=1, transferminator=adapts, 
                mixtureWithScew=options.adap , mixtureWithSwitch=0, switcher=None,temperature_scale=1,
                startVal=None, fixedMax=options.mc3_fixed_temp_max, printFrequency=0)
        else:
            mcmc = MC3(priors, input_files=options.alignments,
                       model=model,
                       thinning=options.thinning, no_chains=options.mc3_chains,
                       switching=options.thinning/10,
                       temperature_scale=options.temperature_scale)
    else:
        if options.use_trees_as_data:
            cT,counts=count_tmrca(subs=options.Ngmu4, filename=options.treefile, align3=False) #align3 is false, because we only want one alignment. 

            log_likelihood=Coal_times_log_lik(times=cT,counts=counts,model=model)
        else:
            
            log_likelihood = Likelihood(IsolationMigrationModel(options.migration_states,
                                                            options.ancestral_states),forwarders)
                                    
        if options.transform==1:
            mcmc = MCMC(priors, log_likelihood, thinning=options.thinning, transformToI=transformToI, transformFromI=transformFromI)#, mixture=options.mixture)
        elif options.transform==2:
            mcmc = MCMC(priors, log_likelihood, thinning=options.thinning, transformToI=transformToI2, transformFromI=transformFromI2)#, mixture=options.mixture)
        else:
            parser.error("wrong transformation number")

    with open(options.outfile, 'w') as outfile:
        print >> outfile, '\t'.join(['isolation.period', 'migration.period',
                                     'theta', 'rho', 'migration',
                                     'log.prior', 'log.likelihood', 'log.posterior'])
        
        if options.logfile:
            with open(options.logfile, 'w') as logfile:
                print >> logfile, '\t'.join(['chain', 'isolation.period', 'migration.period',
                                             'theta', 'rho', 'migration',
                                             'log.prior', 'log.likelihood', 'log.posterior'])

                for _ in xrange(options.samples):
                    # Write main chain to output
                    params, prior, likelihood, posterior = mcmc.sample()
                    print >> outfile, '\t'.join(map(str, transform(params) + (prior, likelihood, posterior)))
                    outfile.flush()

                    # All chains written to the log
                    for chain_no, chain in enumerate(mcmc.chains):
                        params = chain.current_theta
                        prior = chain.current_prior
                        likelihood = chain.current_likelihood
                        posterior = chain.current_posterior
                        print >> logfile, '\t'.join(map(str, (chain_no,) + transform(params) +
                                                        (prior, likelihood, posterior)))
                    logfile.flush()
                    

        else:
            for _ in xrange(options.samples):
                if options.adap>0:
                    all=mcmc.sample()
                    for i in range(options.mc3_chains):
                        params, prior, likelihood, posterior, accepts, rejects,nonSwapAdapParam,swapAdapParam,squaredJump=all[i]
                        if options.adap==3:
                            outfile.write('\t'.join(map(str, transform(params) + (prior, likelihood, posterior, accepts, rejects)+tuple([nonSwapAdapParam[0]])+tuple(swapAdapParam[:3])))+'\t')
                        else:
                            outfile.write('\t'.join(map(str, transform(params) + (prior, likelihood, posterior, accepts, rejects)+tuple(nonSwapAdapParam)+tuple(swapAdapParam)))+'\t')
                    print >> outfile,str(all[-1])
                    outfile.flush()
                else:
                    params, prior, likelihood, posterior = mcmc.sample()
                    print >> outfile, '\t'.join(map(str, transform(params) + (prior, likelihood, posterior)))
                    outfile.flush()
#                 if _%int(options.samples/5)==0:
#                     print >> outfile, printPyZipHMM(mcmc.current_transitionMatrix)
#                     print >> outfile, printPyZipHMM(mcmc.current_initialDistribution)

    if options.mc3:
        mcmc.terminate()

if __name__ == '__main__':
    main()
