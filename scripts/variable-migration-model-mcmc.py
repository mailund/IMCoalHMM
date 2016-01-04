"""Makes mcmc with the variable migration model"""

from pyZipHMM import Forwarder
from newick_count import count_tmrca
from perfectLikelihood import Coal_times_log_lik

from argparse import ArgumentParser
from variable_migration_model2 import VariableCoalAndMigrationRateModel
from variable_migration_model_with_ancestral import VariableCoalAndMigrationRateAndAncestralModel
#from IMCoalHMM.variable_migration_model import VariableCoalAndMigrationRateModel 
from likelihood2 import Likelihood, maximum_likelihood_estimate

from mcmc3 import MCMC, MC3, LogNormPrior, ExpLogNormPrior, UniformPrior, MCG
from math import log,floor
from numpy.random import permutation, randint, random
from copy import deepcopy
from numpy import array
from global_scaling import Global_scaling
from global_scaling_fixer import Global_scaling_fixp
from alg4_scaling import AM4_scaling
from datetime import datetime
from marginal_scaling import MarginalScaler
from marginal_scaler_maxer import MarginalScalerMax
from operator import itemgetter

from ParticleSwarm import Optimiser, OptimiserParallel

def printPyZipHMM(Matrix):
    finalString=""
    for i in range(Matrix.getWidth()):
        for j in range(Matrix.getHeight()):
            finalString=finalString+" "+str(Matrix[i,j])
        finalString=finalString+"\n"
    return finalString

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
parser.add_argument('--mc3_fixed_temp_max', default=None, type=float, help='If applied, this will make the temperature gaps equally crossable and the maximum is said parameter.')

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
parser.add_argument('--adap3_tracking_begin', default=50, type=int, help='In adaption scheme 3, this chooses when we should start simulate proposals using the empirical covariance.')
parser.add_argument('--adap3_from_identical', default=0.2, type=float, help='How big proportion of the time after correlates_begin will we suggest independents with same variance.')
parser.add_argument('--adap3_from_independent', default=0, type=float, help='Will we not use the correlates. If stated the covariance matrix will be estimated without off-diagonal entries.')

parser.add_argument('--printPyMatrices', default=0, type=int, help='How many times should transitionmatrix and initialdistribution be printed for the chain(s) with the correct temperature')
parser.add_argument('--startWithGuess', action='store_true', help='should the initial step be the initial parameters(otherwise simulated from prior).')
parser.add_argument('--startWithGuessElaborate', nargs='+', default=[], type=float, help='should the initial step be the initial parameters(otherwise simulated from prior).')
parser.add_argument('--use_trees_as_data', action='store_true', help='if so, the program will use trees as input data instead of alignments')
parser.add_argument('--record_steps', action='store_true',default=False, help='if so, the program will output the coalescence times of every tenth ')
parser.add_argument('--breakpoints_time', default=1.0, type=float, help='this number moves the breakpoints up and down. Smaller values will give sooner timeperiods.')
parser.add_argument('--intervals', nargs='+', default=[5,5,5,5], type=int, help='This is the setup of the intervals. They will be scattered equally around the breakpoints')
parser.add_argument('--breakpoints_tail_pieces', default=0, type=int, help='this produce a tail of last a number of pieces on the breakpoints')
parser.add_argument('--migration_uniform_prior', default=0, type=int, help='the maximum of the uniform prior on the migration rate is provided here. If nothing, the exponential prior is used.')
parser.add_argument('--fix_params', nargs='+', default=[], type=int, help="the index of the parameters who will be fixed to their starting value throughout simulations. For now it only works when adap=1.")
parser.add_argument('--fix_time_points', nargs='+',default=[], help='this will fix the specified time points. Read source code for further explanation')
parser.add_argument('--fix_parameters_to_be_equal', type=str, default="", help="FOR NOW THIS ONLY WORKS WITH no_mcmc. a comma and colon separated string. commas separate within group and colons separate between groups. If a startWithGuessElaborate is specified this will use the relationsships between the parameters in that as fixed. ")
#One should specify a list of numbers, where the (2n-1)'th number is the index of the time interval one wants set. You can not specify 0 as that is always at time 0.0.
#The 2n'th number is time point measuered in substitions. It will generally be around 10^-4-10^-2.
parser.add_argument('--single_scaling', action='store_true', default=False, help='''if fixed_time_points is set, this will add a parameter to the model scaling 'the time points of fixed_time_points]' up and down(linearly). Default value is 1 of course.''') 
parser.add_argument('--joint_scaling', nargs='+', default=[],type=int, help='The specified fixed_time_points will be scaled up and down and will generate a parameter in the model.')
parser.add_argument('--no_mcmc', action="store_true", default=False, help='If stated this will maximize the function without mcmc using nelder-mead optimization method. Fix params should work with this. ')
parser.add_argument('--last_epoch_ancestral', action='store_true', default=False, help="if stated the last epoch will be an ancestral epoch with no migration an all lineages in the same population. They are dead parameters in the eyes of the other options.")
parser.add_argument("--optimizer",
                    type=str,
                    default="Nelder-Mead",
                    help="If no_mcmc is stated this will be the choice of optimizer.",
                    choices=['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'Particle-Swarm'])

options = parser.parse_args()
if not options.use_trees_as_data:
    if len(options.alignments11) < 1:
        parser.error("Input alignment for the 11 system not provided!")
    if len(options.alignments12) < 1:
        parser.error("Input alignment for the 12 system not provided!")
    if len(options.alignments22) < 1:
        parser.error("Input alignment for the 22 system not provided!")
    
if options.joint_scaling and options.single_scaling:
    parser.error("Joint and single scaling not optional at the same time")

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
    if options.migration_uniform_prior:
        migRate12Priors.append(UniformPrior(init_mig, options.migration_uniform_prior, proposal_sd=options.sd_multiplyer))
        migRate12Priors.append(UniformPrior(init_mig, options.migration_uniform_prior,proposal_sd=options.sd_multiplyer))
    else:
        migRate12Priors.append(ExpLogNormPrior(init_mig, proposal_sd=options.sd_multiplyer))
        migRate12Priors.append(ExpLogNormPrior(init_mig, proposal_sd=options.sd_multiplyer))

priors = coalRate1Priors+coalRate2Priors+migRate12Priors+migRate21Priors+recombRatePrior



#making the right fixed_time_points-function:
def single_scaler(args):
    scale=args[0]
    times=[(f,scale*t) for f,t in fixed_time_points]
    return times
def joint_scaler(args):
    assert len(args)==len(options.joint_scaling), "the given scalers in args="+str(args)+" does not match the previously specified scalers in joint_scaling="+str(options.joint_scaling)
    times=[(f,t) for f,t in fixed_time_points]
    for ind,arg in zip(options.joint_scaling,args):
        times[ind]=(times[ind][0], times[ind][1]*arg)
    return times
def fix_scaler():
    return fixed_time_points
if options.single_scaling:
    priors.append(UniformPrior(1.0, 10.0, proposal_sd=options.sd_multiplyer))
    fixed_time_points=[(int(f),float(t)) for f,t in zip(options.fix_time_points[::2],options.fix_time_points[1::2])]
    fixed_time_pointer=single_scaler
    no_params+=1
elif options.joint_scaling:
    fixed_time_points=[]
    for f,t in zip(options.fix_time_points[::2],options.fix_time_points[1::2]):
        fixed_time_points.append((int(f), float(t)))
    for i in options.joint_scaling:
        if i==len(fixed_time_points)-1: #then this is for the last time interval
            priors.append(UniformPrior(1.0,10.0, proposal_sd=options.sd_multiplyer, a=0.95))
        else:
            priors.append(UniformPrior(1.0,fixed_time_points[i+1][1]*9.0/(fixed_time_points[i][1]*10.0), proposal_sd=options.sd_multiplyer,a=0.95)) #this will leave a small band of error
    fixed_time_pointer=joint_scaler
    no_params+=len(options.joint_scaling)
elif options.fix_time_points:
    print "we are in options.fix_time_points"
    fixed_time_points=[(int(f),float(t)) for f,t in zip(options.fix_time_points[::2],options.fix_time_points[1::2])]

    fixed_time_pointer=fix_scaler
else:
    fixed_time_pointer=None #in stead on could assign it the function that returns []. 
    

def transform(parameters):
    coal_rates_1 = tuple(parameters[0:no_epochs])
    coal_rates_2 = tuple(parameters[no_epochs:(2 * no_epochs)])
    mig_rates_12 = tuple(parameters[(2 * no_epochs):(3 * no_epochs)])
    mig_rates_21 = tuple(parameters[(3 * no_epochs):(4 * no_epochs)])
    recomb_rate = parameters[len(coal_rates_1)*4]
    theta_1 = tuple([2 / coal_rate for coal_rate in coal_rates_1])
    theta_2 = tuple([2 / coal_rate for coal_rate in coal_rates_2])
    if options.single_scaling:
        return theta_1 + theta_2 + mig_rates_12 + mig_rates_21 + (recomb_rate,)+(parameters[-1],)
    if options.joint_scaling:
         return theta_1 + theta_2 + mig_rates_12 + mig_rates_21 + (recomb_rate,)+tuple(parameters[(len(coal_rates_1)*4+1):])    
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
if not options.last_epoch_ancestral:
    model_11 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_11, intervals, breaktimes=options.breakpoints_time, breaktail=options.breakpoints_tail_pieces, time_modifier=fixed_time_pointer)
    model_12 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_12, intervals, breaktimes=options.breakpoints_time, breaktail=options.breakpoints_tail_pieces, time_modifier=fixed_time_pointer)
    model_22 = VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_22, intervals, breaktimes=options.breakpoints_time, breaktail=options.breakpoints_tail_pieces, time_modifier=fixed_time_pointer)
else:
    model_11 = VariableCoalAndMigrationRateAndAncestralModel(VariableCoalAndMigrationRateModel.INITIAL_11, intervals, breaktimes=options.breakpoints_time, breaktail=options.breakpoints_tail_pieces, time_modifier=fixed_time_pointer)
    model_12 = VariableCoalAndMigrationRateAndAncestralModel(VariableCoalAndMigrationRateModel.INITIAL_12, intervals, breaktimes=options.breakpoints_time, breaktail=options.breakpoints_tail_pieces, time_modifier=fixed_time_pointer)
    model_22 = VariableCoalAndMigrationRateAndAncestralModel(VariableCoalAndMigrationRateModel.INITIAL_22, intervals, breaktimes=options.breakpoints_time, breaktail=options.breakpoints_tail_pieces, time_modifier=fixed_time_pointer)


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
    return ((a1[0][0],a2[0][0],a3[0][0]), (a1[1][0],a2[1][0],a3[1][0]), a1[2]+a2[2]+a3[2])

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
        return ((a1[0][0],a2[0][0],a3[0][0]), (a1[1][0],a2[1][0],a3[1][0]), a1[2]+a2[2]+a3[2])
    return log_likelihood


if options.startWithGuess or options.startWithGuessElaborate:
    #startVal=[2.0/0.000575675566598,2.0/0.00221160347741,2.0/0.000707559309234,2.0/0.00125938374711,2.0/0.00475558231719,2.0/0.000829398438542,2.0/0.000371427015082,2.0/0.000320768239201,127.278907998,124.475750838,105.490882058,131.840288312,137.498454174,114.216001115,123.259131284,101.646109897,1.42107787743]
    if options.single_scaling:
        startVal=[init_coal]*no_epochs*2+[init_mig]*no_epochs*2+[init_recomb]+[1.0]
    else:
        startVal=[init_coal]*no_epochs*2+[init_mig]*2*no_epochs+[init_recomb]
    if len(options.startWithGuessElaborate)!=0:
        if len(options.startWithGuessElaborate)==(len(options.intervals)*4+1):
            startVal=options.startWithGuessElaborate
        else:
            "StartWithGuessElaborate is ignored"
    if options.joint_scaling:
        startVal.extend([1.0]*len(options.joint_scaling))
else:
    startVal=None



toTakeMaxFrom=[1-options.adap3_from_identical-options.adap3_from_independent, options.adap3_from_independent,options.adap3_from_identical]
max_index,_ = max(enumerate(toTakeMaxFrom), key=itemgetter(1))

if options.adap==1:
    if options.fix_params:
        fixedParamDict={}
        for f in options.fix_params:
            if startVal is not None:
                fixedParamDict[f]=startVal[f]
            else:
                inits=[init_coal]*no_epochs*2+[init_mig]*2*no_epochs+[init_recomb]
                fixedParamDict[f]=inits[f]
        adap=Global_scaling_fixp(params=[options.adap_harmonic_power, options.adap_step_size], alphaDesired=options.adap_desired_accept, fixes=fixedParamDict)
    else:    
        adap=(Global_scaling(params=[options.adap_harmonic_power, options.adap_step_size], alphaDesired=options.adap_desired_accept))
elif options.adap==2:
    adap=(MarginalScaler(startVal=[0.1]*no_params, params=[options.adap_harmonic_power, options.adap_step_size], alphaDesired=options.adap_desired_accept))
elif options.adap==3:
    adap=AM4_scaling(startVal=no_params*[1.0], 
                     params=[options.adap_harmonic_power, options.adap_step_size, 
                                                       (options.adap3_tracking_begin, options.adap3_correlates_begin),
                                                       (options.adap3_from_identical,options.adap3_from_independent), max_index], 
                     alphaDesired=options.adap_desired_accept)
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


if options.no_mcmc:
    
    changers=[]
    parm_scale_dictionary={}
    if options.fix_parameters_to_be_equal:
        groups=options.fix_parameters_to_be_equal.split(":")
        for group in groups:
            members=map(int,group.split(","))
            leader=members[0]
            for i in members[1:]:
                changers.append(i)
                parm_scale_dictionary[i]=(leader, float(startVal[i])/float(startVal[leader]) )
    NeverChangeParam=startVal
    startVal=[s for n,s in enumerate(startVal) if n not in changers and n not in options.fix_params]
    print NeverChangeParam
    print startVal
    print changers
    print parm_scale_dictionary

    
         
    def fullparams(parameters):
        fullparm=[] #the parameters to feed log likelihood with
        count=0
        for i in xrange(4*no_epochs+1):#running through all the non-time-scaling parameters
            if i in options.fix_params: #
                fullparm.append(NeverChangeParam[i])
            elif i in changers:
                fullparm.append(-1)
            else:
                fullparm.append(parameters[count])
                count+=1
        fullparm=[f if f is not -1 else fullparm[parm_scale_dictionary[n][0]]*parm_scale_dictionary[n][1] for n,f in enumerate(fullparm)]
        fullparm.extend(parameters[count:])
        return fullparm
    test=fullparams(startVal)
    #making a wrapper to take care of fixed parameters and scaling parameters
    def lwrap(parameters):#parameters is the vector of only variable parameters
        parm=fullparams(parameters)
        print parm
        val=log_likelihood(array(parm))[2]
        print val
        return val
    sVal=startVal
    
    if options.optimizer=="Particle-Swarm":
        #inverting coalescence rates
        def lwrapwrap(parameters):
            fullp=fullparams(parameters)
            fullp=[f if n>=(len(fullp)-1) or n in options.fix_params else 1/f for n,f in enumerate(fullp)]
            print fullp
            val=log_likelihood(array(fullp))[2]
            print val
            return val
        if options.parallels>1:
            op=OptimiserParallel()
            mle_parameters = op.maximise(lwrapwrap, len(sVal), processes=options.parallels)
        else:
            op=Optimiser()
            mle_parameters = op.maximise(lwrapwrap, len(sVal))
        
        max_log_likelihood = lwrapwrap(mle_parameters)
        mle_parameters=[1.0/m for m in mle_parameters]
    else:     
        mle_parameters = \
            maximum_likelihood_estimate(lwrap, array(sVal),
                                         optimizer_method=options.optimizer)
        max_log_likelihood = lwrap(mle_parameters)        
    with open(options.outfile, 'w') as outfile:
        print >> outfile, '\t'.join(beforeNames+['recombRate'])
        print >> outfile, '\t'.join(map(str, transform(fullparams(mle_parameters)) + (max_log_likelihood,)))
else:

    
    print "fixedMax="+str(options.mc3_fixed_temp_max)
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
                mixtureWithScew=options.adap , mixtureWithSwitch=options.switch, switcher=switchChooser,temperature_scale=1,
                startVal=startVal, fixedMax=options.mc3_fixed_temp_max, printFrequency=printFrequency)
        else:
            mcmc=MC3(priors, log_likelihood=log_likelihood, accept_jump=options.mc3_jump_accept, flip_suggestions=options.mc3_flip_suggestions,#models=(model_11,model_12,model_22), input_files=(options.alignments11, options.alignments12,options.alignments22),
                sort=options.mc3_sort_chains,chain_structure=chain_structure, thinning=options.thinning, switching=1, #transferminator=adapts, 
                mixtureWithScew=options.adap , mixtureWithSwitch=options.switch, switcher=switchChooser,temperature_scale=1,
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
            
