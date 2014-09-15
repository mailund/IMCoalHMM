#!/uroalHMM.likelihood import Likelihood, maximum_likelihood_estimate
from pyZipHMM import Forwarder

from argparse import ArgumentParser
from IMCoalHMM.variable_migration_model import VariableCoalAndMigrationRateModel
from IMCoalHMM.likelihood import Likelihood, maximum_likelihood_estimate

from mcmc2 import MCMC, MC3, LogNormPrior, ExpLogNormPrior
from math import log
from numpy.random import permutation, randint
from copy import deepcopy
from numpy import array

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
    
    parser.add_argument("--sd_multiplyer", type=float, default=0.2, help="The proportion each proposal suggest changes of all its variance(defined by the transformToI and transformFromI)")
    parser.add_argument('--change_often', nargs='+', default=[], help='put here indices of the variables that should be changed more often')
    parser.add_argument('--switch', default=0, type=int, help='this number is how many times between two switchsteps')
    parser.add_argument('--scew', default=0, type=int, help='this number is how many times between two scewsteps')
    
    options = parser.parse_args()

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
        migRate12Priors.append(LogNormPrior(log(init_mig), proposal_sd=options.sd_multiplyer))
        migRate12Priors.append(LogNormPrior(log(init_mig), proposal_sd=options.sd_multiplyer))

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
    
    multiplyerB=[1000]*17
    multiplyer=[0.001]*17
    print options.change_often
    for i in options.change_often:
        multiplyerB[int(float(i))]-=999
        multiplyer[int(float(i))]*=1000
    
    def makeSomeBig(inarray):
        return [a*x for a,x in zip(multiplyerB, inarray)]
        
    def makeSomeSmall(inarray):
        return [a*x for a,x in zip(multiplyer, inarray)]
    
    def switchChooser(inarray):
        if randint(0,2)==1:
            return switchRows(inarray)
        else: 
            return switchColumns(inarray)
    
    def switchRows(inarray):
        ans=[inarray[no_epochs*4]]*(no_epochs*4+1)
        perm=permutation(no_epochs)
        for i in range(len(perm)):
            for j in range(4):#the four categories c1,c2,mig12,mig21
                 ans[j*no_epochs+i]=inarray[j*no_epochs+perm[i]]
        return array(ans)
        
    def switchColumns(inarray):
        ans=inarray
        length=(len(inarray)-1)
        x=randint(no_epochs)
        y=randint(no_epochs)
        epoch1,epoch2=min(x,y),max(x,y)
        for i in range(epoch1,epoch2+1):
            ans[i],ans[no_epochs+i]=ans[no_epochs+i],ans[i] #coalescence rates
            ans[i+2*no_epochs], ans[i+3*no_epochs]=ans[i+3*no_epochs], ans[i+2*no_epochs] #migration rates
        return ans
    
    # load alignments
    forwarders_11 = [Forwarder.fromDirectory(alignment) for alignment in options.alignments11]
    forwarders_12 = [Forwarder.fromDirectory(alignment) for alignment in options.alignments12]
    forwarders_22 = [Forwarder.fromDirectory(alignment) for alignment in options.alignments22]

    model_11 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_11, intervals)
    model_12 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_12, intervals)
    model_22 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_22, intervals)
    log_likelihood_11 = Likelihood(model_11, forwarders_11)
    log_likelihood_12 = Likelihood(model_12, forwarders_12)
    log_likelihood_22 = Likelihood(model_22, forwarders_22)

    def log_likelihood(parameters):
        return log_likelihood_11(parameters) + log_likelihood_12(parameters) + log_likelihood_22(parameters)

    
    mcmc = MCMC(priors, log_likelihood, thinning=options.thinning, transformToI=makeSomeBig, transformFromI=makeSomeSmall, mixtureWithScew=1 , mixtureWithSwitch=1, switcher=switchChooser)
    
    
    with open(options.outfile, 'w') as outfile:
        print >> outfile, '\t'.join(names+['log.prior', 'log.likelihood', 'log.posterior', 'accepts', 'rejects'])
        for _ in xrange(options.samples):
            params, prior, likelihood, posterior, accepts, rejects = mcmc.sample()
            print >> outfile, '\t'.join(map(str, transform(params) + (prior, likelihood, posterior, accepts, rejects)))
    

if __name__ == '__main__':
    main()
