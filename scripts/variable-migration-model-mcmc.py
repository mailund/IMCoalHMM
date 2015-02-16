"""Makes mcmc with the variable migration model"""

from pyZipHMM import Forwarder
from newick_count import count_tmrca
from perfectLikelihood import Coal_times_log_lik

from argparse import ArgumentParser
from variable_migration_model2 import VariableCoalAndMigrationRateModel #til mcmc2
#from IMCoalHMM.variable_migration_model import VariableCoalAndMigrationRateModel #til mcmc3
from likelihood2 import Likelihood

from mcmc3 import MCMC, MC3, LogNormPrior, ExpLogNormPrior, MCG
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

    optimized_params = [
        ('theta', 'effective population size in 4Ne substitutions', 1e6 / 1e9),
        ('rho', 'recombination rate in substitutions', 0.4),
        ('migration-rate', 'migration rate in number of migrations per substitution', 100.0),
        ('Ngmu4', 'substitutions per 4Ng years', 4*20000*25*1e-9) #it is only used when we use trees as input data. 
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
    
    parser.add_argument('-a11', '--alignments11', nargs='+',
                        help='Alignments of two sequences from the first population')
    parser.add_argument('-a12', '--alignments12', nargs='+',
                        help='Alignments of two sequences, one from each population')
    parser.add_argument('-a22', '--alignments22', nargs='+',
                        help='Alignments of two sequences from the second population')
    
    parser.add_argument('--mc3', action='store_true', default=False, help='this will use mc3 method to ')
    parser.add_argument('--mcg', action='store_true',default=False, help='this will use the multiple-try method')
    parser.add_argument('--parallels', type=int, default=0, help='the number of parallel chains to run in mc3 or number of tries in multiple-try')
    parser.add_argument('--mc3_switching', type=int, default=1, help='the number of switches per thinning period')
    parser.add_argument('--mc3_jump_accept', type=float, default=0.234, help='the swap acceptance probability that the chain is adapting against. Default is 0.234.')
    parser.add_argument('--mc3_flip_suggestions', type=int, default=1, help='The number of times after each step a flip is suggested. It has to be at least one, default is one.')
    parser.add_argument('--mc3_sort_chains', action='store_true', default=False, help='If announced, this will sort each chain according to its posterior value, making it closer to the stationary')
    parser.add_argument('--mc3_mcg_setup', nargs='+', type=int, help='This is a list of sizes of mcgs. Between the mcgs mc3 is made. Overwrites mc3 and mcg. The sum has to equal the number ')
    
    parser.add_argument('--treefile', type=str, help='File containing newick formats of the trees to use as input')
    
    parser.add_argument("--sd_multiplyer", type=float, default=0.2, help="The proportion each proposal suggest changes of all its variance(defined by the transformToI and transformFromI)")
    #parser.add_argument('--change_often', nargs='+', default=[], help='put here indices of the variables that should be changed more often')
    parser.add_argument('--switch', default=0, type=int, help='this number is how many times between two switchsteps')
    
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
    
    parser.add_argument('--startWithGuess', action='store_true', help='should the initial step be the initial parameters(otherwise simulated from prior).')
    parser.add_argument('--use_trees_as_data', action='store_true', help='if so, the program will use trees as input data instead of alignments')
    parser.add_argument('--record_steps', action='store_true',default=False, help='if so, the program will output the coalescence times of every tenth ')
    parser.add_argument('--breakpoints_time', default=1.0, type=float, help='this number moves the breakpoints up and down. Smaller values will give sooner timeperiods.')
    parser.add_argument('--intervals', nargs='+', default=[5,5,5,5], type=int, help='This is the setup of the intervals. They will be scattered equally around the breakpoints')

    options = parser.parse_args()
    if not options.use_trees_as_data:
        if len(options.alignments11) < 1:
            parser.error("Input alignment for the 11 system not provided!")
        if len(options.alignments12) < 1:
            parser.error("Input alignment for the 12 system not provided!")
        if len(options.alignments22) < 1:
            parser.error("Input alignment for the 22 system not provided!")
        
        
    

    # get options
    theta = options.theta
    rho = options.rho

    init_coal = 1 / (theta / 2)
    init_mig = options.migration_rate
    init_recomb = rho

    # FIXME: I don't know what would be a good choice here...
    # intervals = [4] + [2] * 25 + [4, 6]
    intervals=options.intervals
    no_epochs = len(intervals)
    no_params=no_epochs*4+1
    
    incr=range(1,no_epochs+1)*4
    beforeNames=['pop1_coalRate_']*no_epochs+['pop2_coalRate_']*no_epochs+['pop12_migRate_']*no_epochs+['pop21_migRate_']*no_epochs
    names=[a+str(r) for a,r in zip(beforeNames,incr)]+['recombRate']
    
    coalRate1Priors=[]
    coalRate2Priors=[]
    migRate12Priors=[]
    migRate21Priors=[]
    recombRatePrior=[LogNormPrior(log(init_recomb), proposal_sd=options.sd_multiplyer)]
    for i in range(no_epochs):
        coalRate1Priors.append(LogNormPrior(log(init_coal), proposal_sd=options.sd_multiplyer))
        coalRate2Priors.append(LogNormPrior(log(init_coal), proposal_sd=options.sd_multiplyer))
        migRate12Priors.append(ExpLogNormPrior(init_mig, proposal_sd=options.sd_multiplyer))
        migRate12Priors.append(ExpLogNormPrior(init_mig, proposal_sd=options.sd_multiplyer))

    priors = coalRate1Priors+coalRate2Priors+migRate12Priors+migRate21Priors+recombRatePrior


    def transform(parameters):
        coal_rates_1 = tuple(parameters[0:no_epochs])
        coal_rates_2 = tuple(parameters[no_epochs:(2 * no_epochs)])
        mig_rates_12 = tuple(parameters[(2 * no_epochs):(3 * no_epochs)])
        mig_rates_21 = tuple(parameters[(3 * no_epochs):(4 * no_epochs)])
        recomb_rate = parameters[-1]
        theta_1 = tuple([2 / coal_rate for coal_rate in coal_rates_1])
        theta_2 = tuple([2 / coal_rate for coal_rate in coal_rates_2])
        return theta_1 + theta_2 + mig_rates_12 + mig_rates_21 + (recomb_rate,)
    
    def switchChooser(inarray):
        if randint(0,2)==1:
            return simpleConstant12Rate(inarray)
        else: 
            return switchColumns(inarray)
    
    def weightedSwitchColumns(inarray):
        '''This is not perfect for large datasets'''
        ans=deepcopy(inarray)
        length=(len(inarray)-1)
        x=randint(no_epochs)
        y=randint(no_epochs)
        epoch1,epoch2=min(x,y),max(x,y)
        for i in range(epoch1,epoch2+1):
            u=random()
            v=randint(3)
            if v==0:
                ans[i],ans[no_epochs+i]=(ans[no_epochs+i]+ans[i])*u,(ans[no_epochs+i]+ans[i])*(1-u)  #coalescence rates
                ans[i+2*no_epochs], ans[i+3*no_epochs]=(ans[i+3*no_epochs]+ ans[i+2*no_epochs])*u,(ans[i+3*no_epochs]+ ans[i+2*no_epochs])*(1-u) #migration rates
            elif v==1:
                ans[i],ans[no_epochs+i]=(ans[no_epochs+i]+ans[i])*u,(ans[no_epochs+i]+ans[i])*(1-u)  #coalescence rates
            else:
                ans[i+2*no_epochs], ans[i+3*no_epochs]=(ans[i+3*no_epochs]+ ans[i+2*no_epochs])*u,(ans[i+3*no_epochs]+ ans[i+2*no_epochs])*(1-u) #migration rates
        return ans, "col"+str(epoch1)+"-"+str(epoch2)+"w"+str(v)+"-"+str(int(u*5.0))
    
    
    #This also has too few accepts. 
    indexOfInterest=[(1,8),(2,9),(3,10),(5,12),(6,13),(7,14)]
    def simpleConstant12Rate(inarray):
        ans=deepcopy(inarray)
        #draws a subset to be reweighted
        rstring="scr"
        for n in range(6):
            if random()<0.5:
                rstring+="-"+str(indexOfInterest[n])
                u=random()
                if random()<0.5:
                    rstring+=str(int(u*5.0))
                    ans[indexOfInterest[n][0]]*=u
                    ans[indexOfInterest[n][1]]/=u
                else:
                    rstring+=str(int(u*5.0))
                    ans[indexOfInterest[n][1]]*=u
                    ans[indexOfInterest[n][0]]/=u
        return ans, rstring
                
    
    

    #THIS HAS VERY FEW ACCEPTS SO ABANDONED. 
#    def switchRows(inarray):
#        ans=[inarray[no_epochs*4]]*(no_epochs*4+1)
#        draw=randint(0,6)
#        if draw==0:
#            h=[0]
#        elif draw==1:
#            h=[1]
#        elif draw==2:
#            h=[0,1]
#        elif draw==3:
#            h=[0,2]
#        elif draw==4:
#            h=[1,3]
#        else:
#            h=range(3)
#        perm=permutation(no_epochs)
#        for i in range(len(perm)):
#            for j in h:#a subset of the four categories c1,c2,mig12,mig21
#                 ans[j*no_epochs+i]=inarray[j*no_epochs+perm[i]]
#        return array(ans), "row"+str(draw)+"-"+str(perm)
        
    def switchColumns(inarray):
        ans=deepcopy(inarray)
        x=randint(no_epochs)
        y=randint(no_epochs)
        epoch1,epoch2=min(x,y),max(x,y)
        for i in range(epoch1,epoch2+1):
            ans[i],ans[no_epochs+i]=ans[no_epochs+i],ans[i] #coalescence rates
            ans[i+2*no_epochs], ans[i+3*no_epochs]=ans[i+3*no_epochs], ans[i+2*no_epochs] #migration rates
        return ans, "col"+str(epoch1)+"-"+str(epoch2)
    
    # load alignments
    model_11 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_11, intervals, breaktimes=options.breakpoints_time)
    model_12 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_12, intervals, breaktimes=options.breakpoints_time)
    model_22 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_22, intervals, breaktimes=options.breakpoints_time)
    
    if options.use_trees_as_data:
        leftT,rightT,combinedT,counts=count_tmrca(subs=options.Ngmu4,filename=options.treefile)
        
        log_likelihood_11=Coal_times_log_lik(times=leftT,counts=counts, model=model_11)
        log_likelihood_12=Coal_times_log_lik(times=combinedT,counts=counts, model=model_12)
        log_likelihood_22=Coal_times_log_lik(times=rightT,counts=counts, model=model_22)
    else:
        forwarders_11 = [Forwarder.fromDirectory(alignment) for alignment in options.alignments11]
        forwarders_12 = [Forwarder.fromDirectory(alignment) for alignment in options.alignments12]
        forwarders_22 = [Forwarder.fromDirectory(alignment) for alignment in options.alignments22]
        
        log_likelihood_11 = Likelihood(model_11, forwarders_11)
        log_likelihood_12 = Likelihood(model_12, forwarders_12)
        log_likelihood_22 = Likelihood(model_22, forwarders_22)

    def log_likelihood(parameters):
        a1=log_likelihood_11(parameters)
        a2=log_likelihood_12(parameters)
        a3=log_likelihood_22(parameters)
        return ((a1[0],a2[0],a3[0]), (a1[1],a2[1],a3[1]), a1[2]+a2[2]+a3[2])
    
    def likelihoodWrapper():
        if options.use_trees_as_data:
            leftT,rightT,combinedT,counts=count_tmrca(subs=options.Ngmu4,filename=options.treefile)
            
            log_likelihood_11=Coal_times_log_lik(times=leftT,counts=counts, model=model_11)
            log_likelihood_12=Coal_times_log_lik(times=combinedT,counts=counts, model=model_12)
            log_likelihood_22=Coal_times_log_lik(times=rightT,counts=counts, model=model_22)
        else:
            forwarders_11 = [Forwarder.fromDirectory(alignment) for alignment in options.alignments11]
            forwarders_12 = [Forwarder.fromDirectory(alignment) for alignment in options.alignments12]
            forwarders_22 = [Forwarder.fromDirectory(alignment) for alignment in options.alignments22]
            
            log_likelihood_11 = Likelihood(model_11, forwarders_11)
            log_likelihood_12 = Likelihood(model_12, forwarders_12)
            log_likelihood_22 = Likelihood(model_22, forwarders_22)
        def log_likelihood(parameters):
            a1=log_likelihood_11(parameters)
            a2=log_likelihood_12(parameters)
            a3=log_likelihood_22(parameters)
            return ((a1[0],a2[0],a3[0]), (a1[1],a2[1],a3[1]), a1[2]+a2[2]+a3[2])
        return log_likelihood
    

    toTakeMaxFrom=[1-options.adap3_from_identical-options.adap3_from_independent, options.adap3_from_independent,options.adap3_from_identical]
    max_index,_ = max(enumerate(toTakeMaxFrom), key=itemgetter(1))

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

    if options.startWithGuess:
        startVal=[init_coal]*8+[init_mig]*8+[init_recomb]
    else:
        startVal=None

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
        print likelihoodWrapper()
        if options.adap>0:
            mcmc=MC3(priors, log_likelihood=log_likelihood, accept_jump=options.mc3_jump_accept, flip_suggestions=options.mc3_flip_suggestions,#models=(model_11,model_12,model_22), input_files=(options.alignments11, options.alignments12,options.alignments22),
                sort=options.mc3_sort_chains, chain_structure=chain_structure, thinning=options.thinning, switching=1, transferminator=adapts, 
                mixtureWithScew=options.adap , mixtureWithSwitch=options.switch, switcher=switchChooser,temperature_scale=1,startVal=startVal)
        else:
            mcmc=MC3(priors, log_likelihood=log_likelihood, accept_jump=options.mc3_jump_accept, flip_suggestions=options.mc3_flip_suggestions,#models=(model_11,model_12,model_22), input_files=(options.alignments11, options.alignments12,options.alignments22),
                sort=options.mc3_sort_chains,chain_structure=chain_structure, thinning=options.thinning, switching=1, #transferminator=adapts, 
                mixtureWithScew=options.adap , mixtureWithSwitch=options.switch, switcher=switchChooser,temperature_scale=1,startVal=startVal)     
    elif options.mcg and not options.mc3_mcg_setup:
        mcmc=MCG(priors,log_likelihood=log_likelihood,probs=options.parallels,transferminator=adap, startVal=startVal)
    elif not options.startWithGuess:
        mcmc = MCMC(priors, log_likelihood, thinning=options.thinning, transferminator=adap, startVal=startVal)
    else:
        mcmc = MCMC(priors, log_likelihood, thinning=options.thinning, transferminator=adap, mixtureWithScew=options.adap, startVal=startVal)

    
    print "before starting to simulate"
    with open(options.outfile, 'w') as outfile:
        if not options.mc3:
            print >> outfile, '\t'.join(names+['log.prior', 'log.likelihood', 'log.posterior', 'accepts', 'rejects', 'adapParam','latestjump'])
        else:
            basenames=names+['log.prior', 'log.likelihood', 'log.posterior', 'accepts', 'rejects', 'theta','latestjump']
            print >> outfile, '\t'.join(basenames*options.parallels)+'\t'+'flips'
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
