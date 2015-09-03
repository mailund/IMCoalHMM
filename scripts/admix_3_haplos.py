"""Makes mcmc with the variable migration model"""

from pyZipHMM import Forwarder
from newick_count import count_tmrca
from perfectLikelihood import Coal_times_log_lik

from argparse import ArgumentParser
from IMCoalHMM.admixture_3_haplos_model import Admixture3HModel 
from likelihood2 import Likelihood
from IMCoalHMM.likelihood import maximum_likelihood_estimate

from mcmc3 import MCMC, MC3, LogNormPrior, ExpLogNormPrior, UniformPrior, MCG
from math import log,floor
from numpy.random import permutation, randint, random
from copy import deepcopy
from numpy import array
from global_scaling import Global_scaling
from alg4_scaling import AM4_scaling
from datetime import datetime
from marginal_scaling import MarginalScaler
from marginal_scaler_maxer import MarginalScalerMax
from operator import itemgetter

def printPyZipHMM(Matrix):
    finalString=""
    for i in range(Matrix.getWidth()):
        for j in range(Matrix.getHeight()):
            finalString=finalString+" "+str(Matrix[i,j])
        finalString=finalString+"\n"
    return finalString

def main():
    """
    Run the main script.
    """
    usage = """%(prog)s [options] <forwarder dirs>

This program estimates the coalescence and migration rates over time together with a constant
recombination rate."""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.0")

    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/dev/stdout",
                        help="Output file for the estimate (/dev/stdout)")

    parser.add_argument("--logfile",
                        type=str,
                        default=None,
                        help="Log for all points estimated in the optimization")
    
    parser.add_argument("--optimizer",
                        type=str,
                        default="Nelder-Mead",
                        help="Optimization algorithm to use for maximizing the likelihood (Nealder-Mead)",
                        choices=['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC'])

    optimized_params = [
        ('adm_time', 'admixture time in substitutions', 1e6 / 1e9),
        ('split_123', 'Second split time in substitutions', 1e6 / 1e9),
        ('theta_12a', 'effective population size in 4Ne substitutions for species 12 before admixture', 1e6 / 1e9),
        ('theta_12b', 'effective population size in 4Ne substitutions for species 12 after admixture', 1e6 / 1e9),
        ('theta_3a', 'effective population size in 4Ne substitutions for species 3 before admixture', 1e6 / 1e9),
        ('theta_3b', 'effective population size in 4Ne substitutions for species 3 after admixture', 1e6 / 1e9),
        ('theta_last', 'effective population size in 4Ne substitutions for species 123 (ancestral to all)', 1e6 / 1e9),
        ('p', 'proportion of population 12 going to population 3',0.1),
        ('q', 'proportion of population 3 going to population 12', 0.1),
        ('rho', 'recombination rate in substitutions', 0.4),
        ('outgroup', 'total height of tree with outgroup', 1e6 / 1e9)
    ]

    for parameter_name, description, default in optimized_params:
        parser.add_argument("--%s" % parameter_name,
                            type=float,
                            default=default,
                            help="Initial guess at the %s (%g)" % (description, default))
        
    parser.add_argument("-n", "--samples",
                        type=int,
                        default=500,
                        help="Number of samples to draw (500)")    
    
    parser.add_argument("-k", "--thinning",
                        type=int,
                        default=100,
                        help="Number of MCMC steps between samples (100)")
    
    parser.add_argument('--alignments', nargs='+', help='Alignments in ZipHMM format')
    
    parser.add_argument('--mc3', action='store_true', default=False, help='this will use mc3 method to ')
    parser.add_argument('--mcg', action='store_true',default=False, help='this will use the multiple-try method')
    parser.add_argument('--parallels', type=int, default=0, help='the number of parallel chains to run in mc3 or number of tries in multiple-try')
    parser.add_argument('--mc3_switching', type=int, default=1, help='the number of switches per thinning period')
    parser.add_argument('--mc3_jump_accept', type=float, default=0.234, help='the swap acceptance probability that the chain is adapting against. Default is 0.234.')
    parser.add_argument('--mc3_flip_suggestions', type=int, default=1, help='The number of times after each step a flip is suggested. It has to be at least one, default is one.')
    parser.add_argument('--mc3_sort_chains', action='store_true', default=False, help='If announced, this will sort each chain according to its posterior value, making it closer to the stationary')
    parser.add_argument('--mc3_mcg_setup', nargs='+', type=int, help='This is a list of sizes of mcgs. Between the mcgs mc3 is made. Overwrites mc3 and mcg. The sum has to equal the number ')
    parser.add_argument('--mc3_fixed_temp_max', default=None, type=float, help='If applied, this will make the temperature gaps equally crossable and the maximum is said parameter.')
    
#    parser.add_argument('--treefile', type=str, help='File containing newick formats of the trees to use as input')
    
    parser.add_argument("--sd_multiplyer", type=float, default=0.1, help="The proportion each proposal suggest changes of all its variance(defined by the transformToI and transformFromI)")
    #parser.add_argument('--change_often', nargs='+', default=[], help='put here indices of the variables that should be changed more often')
    parser.add_argument('--switch', default=0, type=int, help='this number is how many times between two switchsteps')
    parser.add_argument('--no_mcmc', default=False, action="store_true", help="this will make the program maximize the function(and not explore by mcmc).")
    
    parser.add_argument('--adap', default=0, type=int, help='this number tells what adaption to use')
    parser.add_argument('--adap_step_size', default=1.0, type=float, help='this number is the starting step size of the adaption')
    parser.add_argument('--adap_step_size_marginal', default=0.1, type=float, help='this number is the starting step size of the adaption of the marginals')
    parser.add_argument('--adap_harmonic_power', default=0.5, type=float, help='this number is the power of the harmonical decrease in scew with adaption. It tells how fast the adaption vanishes.')
    parser.add_argument('--adap_desired_accept', default=0.234, type=float, help='this number is the acceptance rate that the adaptive algorithm strives for')
    parser.add_argument('--adap4_mediorizing', default=False, action='store_true', help='In adaptive scheme 4 there is a choice between making it hard for parameters to vanish or not. If this is stated, it is not.')
    parser.add_argument('--adap4_proportion', default=0.5, type=float, help='In adaption scheme 4 one can target those distributions which means a lot. The larger proportion, the more targeted.')
    parser.add_argument('--adap3_correlates_begin', default=100, type=int, help='In adaption scheme 3, this chooses when we should start simulate proposals using the empirical covariance.')
    parser.add_argument('--adap3_from_identical', default=0.2, type=float, help='How big proportion of the time after correlates_begin will we suggest independents with same variance.')
    parser.add_argument('--adap3_from_independent', default=0, type=float, help='Will we not use the correlates. If stated the covariance matrix will be estimated without off-diagonal entries.')
    
    parser.add_argument('--printPyMatrices', default=0, type=int, help='How many times should transitionmatrix and initialdistribution be printed for the chain(s) with the correct temperature')
    parser.add_argument('--startWithGuess', action='store_true', help='should the initial step be the initial parameters(otherwise simulated from prior).')
#    parser.add_argument('--use_trees_as_data', action='store_true', help='if so, the program will use trees as input data instead of alignments')
    parser.add_argument('--record_steps', action='store_true',default=False, help='if so, the program will output the coalescence times of every tenth ')
#    parser.add_argument('--breakpoints_time', default=1.0, type=float, help='this number moves the breakpoints up and down. Smaller values will give sooner timeperiods.')
    parser.add_argument('--intervals', nargs='+', default=[5,5,5], type=int, help='This is the setup of the intervals. They will be scattered equally around the breakpoints')
#    parser.add_argument('--breakpoints_tail_pieces', default=0, type=int, help='this produce a tail of last a number of pieces on the breakpoints')
#    parser.add_argument('--migration_uniform_prior', default=0, type=int, help='the maximum of the uniform prior on the migration rate is provided here. If nothing, the exponential prior is used.')



    options = parser.parse_args()
    if len(options.alignments) < 1:
        parser.error("No input alignment given")
    if not len(options.intervals)==3:
        parser.error("Wrong number number of numbers of intervals given. There should be 3 numbers after --intervals")

        
    init_parameters = (
        options.adm_time,
        options.split_123,
        1 / (options.theta_12a / 2),
        1 / (options.theta_12b / 2),
        1 / (options.theta_3a / 2),
        1 / (options.theta_3b / 2),
        1 / (options.theta_last / 2),
        options.p,
        options.q,
        options.rho
    )
    
    
    names=['admixture_time', 'split_time','theta12_before_adm', 'theta12_after_adm', 'theta3_before_adm','theta3_after_adm','theta_ancestral','recombination_rate']
    
    priors=[]
    priors.append(LogNormPrior(log(init_parameters[0]), proposal_sd=options.sd_multiplyer))
    priors.append(LogNormPrior(log(init_parameters[1]), proposal_sd=options.sd_multiplyer))
    for mean_value in init_parameters[2:-3]:
        priors.append(LogNormPrior(log(mean_value), proposal_sd=options.sd_multiplyer))
    priors.append(LogNormPrior(log(init_parameters[-3]), proposal_sd=options.sd_multiplyer))
    priors.append(UniformPrior(init_parameters[-2], until=0.5, proposal_sd=options.sd_multiplyer))
    priors.append(UniformPrior(init_parameters[-1], until=0.5, proposal_sd=options.sd_multiplyer))
    
    
    def transform(params):
        split_time_12, split_time_123, coal_rate_1, coal_rate_2, coal_rate_3, \
            coal_rate_12, coal_rate_123, recomb_rate,p,q = params
        return split_time_12, split_time_123, \
           2 / coal_rate_1, 2 / coal_rate_2, 2 / coal_rate_3, \
           2 / coal_rate_12, 2 / coal_rate_123, \
           recomb_rate,p,q
        
      
    # load alignments
    model = Admixture3HModel(options.intervals[0], options.intervals[1], options.intervals[2])
    forwarders = [Forwarder.fromDirectory(arg) for arg in options.alignments]
    log_likelihood = Likelihood(model, forwarders)

    

    toTakeMaxFrom=[1-options.adap3_from_identical-options.adap3_from_independent, options.adap3_from_independent,options.adap3_from_identical]
    max_index,_ = max(enumerate(toTakeMaxFrom), key=itemgetter(1))
    no_params=len(init_parameters)
    if options.adap==1:
        adap=(Global_scaling(params=[options.adap_harmonic_power, options.adap_step_size], alphaDesired=options.adap_desired_accept))
    elif options.adap==2:
        adap=(MarginalScaler(startVal=[0.1]*no_params, params=[options.adap_harmonic_power, options.adap_step_size], alphaDesired=options.adap_desired_accept))
    elif options.adap==3:
        adap=AM4_scaling(startVal=no_params*[1.0], params=[options.adap_harmonic_power, options.adap_step_size, options.adap3_correlates_begin,(options.adap3_from_identical,options.adap3_from_independent), max_index], alphaDesired=options.adap_desired_accept)
    elif options.adap==4:
        adap=(MarginalScalerMax(startVal=[0.1]*no_params, params=[options.adap_harmonic_power, options.adap_step_size, options.adap4_mediorizing, options.adap_step_size_marginal], alphaDesired=options.adap_desired_accept, targetProportion=options.adap4_proportion))
    else:
        adap=None
    
    namesAndAdap=names
    namesAndAdap.extend(['log.prior', 'log.likelihood', 'log.posterior', 'accepts', 'rejects'])
    if options.adap>0: #change names
        namesAndAdap.extend(adap.NONSWAP_PARAM)
        for j in adap.SWAP_PARAM:
            if j==0:
                namesAndAdap.extend(   (i+"adap" for i in names)    )
            else:
                namesAndAdap.append(j) 
    #namesAndAdap.append('latestJump')          
    if options.printPyMatrices>0:
        printFrequency=options.samples/options.printPyMatrices
    else:
        printFrequency=0
    print printFrequency
    if options.startWithGuess:
        startVal=init_parameters
    else:
        startVal=None
    print "fixedMax="+str(options.mc3_fixed_temp_max)
        
    if options.no_mcmc:
        def simple_log_likelihood(parameters):
            return log_likelihood(parameters)[2]
        mle_parameters = \
            maximum_likelihood_estimate(simple_log_likelihood, init_parameters,
                                        optimizer_method=options.optimizer)

        max_log_likelihood = simple_log_likelihood(mle_parameters)
        with open(options.outfile, 'w') as outfile:
            if options.header:
                print >> outfile, '\t'.join(['isolation.period', 'migration.period',
                                             'theta', 'rho', 'migration', 'log.likelihood'])
            print >> outfile, '\t'.join(map(str, transform(mle_parameters) + (max_log_likelihood,)))
    else:
        
        if options.mc3 or options.mc3_mcg_setup:
            if options.mc3_mcg_setup and options.parallels>0:
                if not sum(options.mc3_mcg_setup)==options.parallels:
                    print "The requested number of parallel chains do not match those requested"
            if not options.mc3_mcg_setup:
                chain_structure=[1]*options.parallels
            else:
                chain_structure=options.mc3_mcg_setup
            adapts=[] #we have to make a list of adaptors
            no_chains=len(chain_structure)
            for _ in range(no_chains):
                adapts.append(deepcopy(adap))
            if options.adap>0:
                mcmc=MC3(priors, log_likelihood=log_likelihood, accept_jump=options.mc3_jump_accept, flip_suggestions=options.mc3_flip_suggestions,#models=(model_11,model_12,model_22), input_files=(options.alignments11, options.alignments12,options.alignments22),
                    sort=options.mc3_sort_chains, chain_structure=chain_structure, thinning=options.thinning, switching=1, transferminator=adapts, 
                    mixtureWithScew=options.adap , mixtureWithSwitch=options.switch,temperature_scale=1,
                    startVal=startVal, fixedMax=options.mc3_fixed_temp_max, printFrequency=printFrequency)
            else:
                mcmc=MC3(priors, log_likelihood=log_likelihood, accept_jump=options.mc3_jump_accept, flip_suggestions=options.mc3_flip_suggestions,#models=(model_11,model_12,model_22), input_files=(options.alignments11, options.alignments12,options.alignments22),
                    sort=options.mc3_sort_chains,chain_structure=chain_structure, thinning=options.thinning, switching=1, #transferminator=adapts, 
                    mixtureWithScew=options.adap , mixtureWithSwitch=options.switch,temperature_scale=1,
                    startVal=startVal, fixedMax=options.mc3_fixed_temp_max, printFrequency=printFrequency)     
        elif options.mcg and not options.mc3_mcg_setup:
            mcmc=MCG(priors,log_likelihood=log_likelihood,probs=options.parallels,transferminator=adap, startVal=startVal, printFrequency=printFrequency)
        elif not options.startWithGuess:
            mcmc = MCMC(priors, log_likelihood, thinning=options.thinning, transferminator=adap, startVal=startVal, printFrequency=printFrequency)
        else:
            mcmc = MCMC(priors, log_likelihood, thinning=options.thinning, transferminator=adap, mixtureWithScew=options.adap, startVal=startVal, printFrequency=printFrequency)
    
        
        print "before starting to simulate"
        with open(options.outfile, 'w') as outfile:
            if not options.mc3:
                print >> outfile, '\t'.join(namesAndAdap)
            else:
                print >> outfile, '\t'.join(namesAndAdap*no_chains)+'\t'+'flips'
            for j in xrange(options.samples):
                print "sample "+str(j)+" time: " + str(datetime.now())
                if options.record_steps:
                    params, prior, likelihood, posterior, accepts, rejects,nonSwapAdapParam,swapAdapParam,squaredJump, latestSuggest,latestInit = mcmc.sampleRecordInitialDistributionJumps()
                    print >> outfile, '\t'.join(map(str, transform(params) + (prior, likelihood, posterior, accepts, rejects)+tuple(nonSwapAdapParam)+tuple(swapAdapParam)+(squaredJump,0))+transform(latestSuggest))+\
                        printPyZipHMM(latestInit[0]).rstrip()+printPyZipHMM(latestInit[1]).rstrip()+printPyZipHMM(latestInit[2]).rstrip()
                elif options.mc3 or options.mc3_mcg_setup:
                    all=mcmc.sample()
                    for i in range(no_chains):
                        params, prior, likelihood, posterior, accepts, rejects,nonSwapAdapParam,swapAdapParam,squaredJump=all[i]
                        if options.adap==3:
                            outfile.write('\t'.join(map(str, transform(params) + (prior, likelihood, posterior, accepts, rejects)+tuple(nonSwapAdapParam)+tuple(swapAdapParam[:3])))+'\t')
                        else:
                            outfile.write('\t'.join(map(str, transform(params) + (prior, likelihood, posterior, accepts, rejects)+tuple(nonSwapAdapParam)+tuple(swapAdapParam)))+'\t')
                    print >> outfile,str(all[-1])
                else:
                    params, prior, likelihood, posterior, accepts, rejects,nonSwapAdapParam,swapAdapParam,squaredJump= mcmc.sample()
                    if j%options.thinning==0:#reducing output a little
                        print >> outfile, '\t'.join(map(str, transform(params) + (prior, likelihood, posterior, accepts, rejects)+tuple(nonSwapAdapParam)+tuple(swapAdapParam)))
                outfile.flush()
                if not options.record_steps and not options.mc3 and not options.mcg and not options.mc3_mcg_setup:
                    if j%max(int(options.samples/5),1)==0:
                        for i in range(3):
                            print >> outfile, printPyZipHMM(mcmc.current_transitionMatrix[i])
                            print >> outfile, printPyZipHMM(mcmc.current_initialDistribution[i])
            
            if not options.record_steps and not options.mc3 and not options.mcg and not options.mc3_mcg_setup:
                for i in range(3):
                    print >> outfile, printPyZipHMM(mcmc.current_transitionMatrix[i])
                    print >> outfile, printPyZipHMM(mcmc.current_initialDistribution[i])
                acc=mcmc.getSwitchStatistics()
                print >> outfile, "Accepted switches"
                for i in acc[0]:
                    print >> outfile, str(i)+"    "+str(acc[0][i])
                print >> outfile, "Rejected switches"
                for i in acc[1]:
                    print >> outfile, str(i)+"    "+str(acc[1][i])
            outfile.flush()
        if options.mc3 or options.mcg or options.mc3_mcg_setup:
            mcmc.terminate()
        

if __name__ == '__main__':
    main()
