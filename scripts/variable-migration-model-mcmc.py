"""Makes mcmc with the variable migration model"""

from pyZipHMM import Forwarder
from newick_count import count_tmrca
from perfectLikelihood import Coal_times_log_lik

from argparse import ArgumentParser
from variable_migration_model2 import VariableCoalAndMigrationRateModel
from likelihood2 import Likelihood

from mcmc2 import MCMC, MC3, LogNormPrior, ExpLogNormPrior
from math import log,floor
from numpy.random import permutation, randint, random
from copy import deepcopy
from numpy import array
from global_scaling import Global_scaling
from alg4_scaling import AM4_scaling

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
    parser.add_argument('--mc3_chains', type=int, default=8, help='the number of parallel chains to run in mc3')
    
    parser.add_argument('--treefile', type=str, help='File containing newick formats of the trees to use as input')
    
    parser.add_argument("--sd_multiplyer", type=float, default=0.2, help="The proportion each proposal suggest changes of all its variance(defined by the transformToI and transformFromI)")
    #parser.add_argument('--change_often', nargs='+', default=[], help='put here indices of the variables that should be changed more often')
    parser.add_argument('--switch', default=0, type=int, help='this number is how many times between two switchsteps')
    parser.add_argument('--adap', default=0, type=int, help='this number tells what adaption to use')
    parser.add_argument('--adap_step_size', default=1.0, type=float, help='this number is the starting step size of the adaption')
    parser.add_argument('--adap_harmonic_power', default=0.5, type=float, help='this number is the power of the harmonical decrease in scew with adaption. It tells how fast the adaption vanishes.')
    parser.add_argument('--adap_desired_accept', default=0.234, type=float, help='this number is the acceptance rate that the adaptive algorithm strives for')
    parser.add_argument('--startWithGuess', action='store_true', help='should the initial step be the initial parameters(otherwise simulated from prior).')
    parser.add_argument('--use_trees_as_data', action='store_true', help='if so, the program will use trees as input data instead of alignments')
    parser.add_argument('--record_steps', action='store_true',default=False, help='if so, the program will output the coalescence times of every tenth ')
    
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
    intervals = [5, 5, 5, 5]
    no_epochs = len(intervals)
    
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
    model_11 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_11, intervals)
    model_12 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_12, intervals)
    model_22 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_22, intervals)
    
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
    


    if options.mc3:
        no_chains=options.mc3_chains
        adapts=[]
        for i in range(no_chains):
            if options.adap>0:
                adapts.append(Global_scaling(params=[options.adap_harmonic_power, options.adap_step_size], alphaDesired=options.adap_desired_accept))
        mcmc=MC3(priors,models=(model_11,model_12,model_22), input_files=(options.alignments11, options.alignments12,options.alignments22),
                 no_chains=no_chains, thinning=options.thinning, switching=max(1,int(options.thinning/10)), transferminator=adapts, 
                 mixtureWithScew=options.adap , mixtureWithSwitch=options.switch, switcher=switchChooser)           
    elif not options.startWithGuess:
        if options.adap>0:
            var=options.sd_multiplyer
            if options.adap>1:
                adap=AM4_scaling(size=17, params=[options.adap_harmonic_power, options.adap_step_size], alphaDesired=options.adap_desired_accept)
            else:
                adap=Global_scaling(params=[options.adap_harmonic_power, options.adap_step_size], alphaDesired=options.adap_desired_accept)
            mcmc = MCMC(priors, log_likelihood, thinning=options.thinning, transferminator=adap, mixtureWithScew=options.adap , mixtureWithSwitch=options.switch, switcher=switchChooser)
        else:
            mcmc = MCMC(priors, log_likelihood, thinning=options.thinning, mixtureWithSwitch=options.switch, switcher=switchChooser)
    else:
        thetaGuess=[init_coal]*8+[init_mig]*8+[init_recomb]
        mcmc = MCMC(priors, log_likelihood, thinning=options.thinning, startVal=thetaGuess)

    
    with open(options.outfile, 'w') as outfile:
        print >> outfile, '\t'.join(names+['log.prior', 'log.likelihood', 'log.posterior', 'accepts', 'rejects', 'theta'])
        for j in xrange(options.samples):
            params, prior, likelihood, posterior, accepts, rejects,adapParam,squaredJump, latestInit = mcmc.sample()
            if options.record_steps:
                print >> outfile, '\t'.join(map(str, transform(params) + (prior, likelihood, posterior, accepts, rejects,adapParam,squaredJump)))+\
                    printPyZipHMM(latestInit[0]).rstrip()+printPyZipHMM(latestInit[1]).rstrip()+printPyZipHMM(latestInit[2]).rstrip()
            else:
                print >> outfile, '\t'.join(map(str, transform(params) + (prior, likelihood, posterior, accepts, rejects,adapParam)))
            outfile.flush()
            if not options.record_steps:
                if j%max(int(options.samples/5),1)==0:
                    for i in range(3):
                        print >> outfile, printPyZipHMM(mcmc.current_transitionMatrix[i])
                        print >> outfile, printPyZipHMM(mcmc.current_initialDistribution[i])
        if not options.record_steps:
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
        

if __name__ == '__main__':
    main()
