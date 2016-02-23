#!/usr/bin/env python

"""Script for estimating parameters in an initial migration model.
"""

from argparse import ArgumentParser

from IMCoalHMM.likelihood import Likelihood, maximum_likelihood_estimate
#from IMCoalHMM.isolation_with_migration_model import IsolationMigrationModel
from pyZipHMM import Forwarder

from IMCoalHMM.isolation_with_migration_model import IsolationMigrationModel

from ParticleSwarm import Optimiser, OptimiserParallel

import isolation_with_migration_model2
from math import log, exp
from numpy import array
from sys import exc_info


def transform(params):
    """
    Translate the parameters to the input and output parameter space.
    """
    if len(params)==6:
        isolation_time, migration_time, coal_rate, recomb_rate, mig_rate, outgroup = params
        return isolation_time, migration_time, 2 / coal_rate, recomb_rate, mig_rate, outgroup
    
    isolation_time, migration_time, coal_rate, recomb_rate, mig_rate = params
    return isolation_time, migration_time, 2 / coal_rate, recomb_rate, mig_rate


usage = """%(prog)s [options] <forwarder dirs>

This program estimates the parameters of an isolation model with an initial migration period with two species
and uniform coalescence and recombination rates."""

parser = ArgumentParser(usage=usage, version="%(prog)s 1.2")

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

parser.add_argument("--ancestral-states",
                    type=int,
                    default=10,
                    help="Number of intervals used to discretize the time in the ancestral population (10)")
parser.add_argument("--migration-states",
                    type=int,
                    default=10,
                    help="Number of intervals used to discretize the time in the migration period (10)")

parser.add_argument("--optimizer",
                    type=str,
                    default="Nelder-Mead",
                    help="Optimization algorithm to use for maximizing the likelihood (Nealder-Mead)",
                    choices=['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'Particle-Swarm'])

parser.add_argument("--verbose",
                    default=False,
                    action='store_true')      

parser.add_argument("--emissionComplicated", default=False, action="store_true", help="This will use an emission matrix which is not an approximation.")
parser.add_argument('--outgroup', action='store_true', default=False, help="This indicates that the alignemnts are not pairwise but threewise and that the last entry will be ")

parser.add_argument('--constant_break_points', default=False, action="store_true", help='If enabled, the break points will be fixed throughout the analysis but the epochs will change')
parser.add_argument('--breakpoints_tail_pieces', default=0, type=int, help='this produce a tail of last a number of pieces on the breakpoints')
parser.add_argument('--breakpoints_time', default=1.0, type=float, help='this number moves the breakpoints up and down. Smaller values will give sooner timeperiods.')
parser.add_argument('--fix_params', nargs='+', default=[], type=int, help="the index of the parameters who will be fixed to their starting value throughout simulations. For now it only works when optimizer=Particle-Swarm")
parser.add_argument('--fix_parameters_to_be_equal', type=str, default="", help="a comma and colon separated string. commas separate within group and colons separate between groups. If a startWithGuessElaborate is specified this will use the relationsships between the parameters in that as fixed. ")
parser.add_argument('--parallels', type=int, default=0, help='the number of processes to spread particle swarm across')


optimized_params = [
    ('isolation-period', 'time where the populations have been isolated', 1e6 / 1e9),
    ('migration-period', 'time period where the populations exchanged genes', 1e6 / 1e9),
    ('theta', 'effective population size in 4Ne substitutions', 1e6 / 1e9),
    ('rho', 'recombination rate in substitutions', 0.4),
    ('migration-rate', 'migration rate in number of migrations per substitution', 200.0)
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
no_migration_states = options.migration_states
no_ancestral_states = options.ancestral_states
theta = options.theta
rho = options.rho

forwarders = [Forwarder.fromDirectory(arg) for arg in options.alignments]

init_isolation_time = options.isolation_period
init_migration_time = options.migration_period
init_coal = 1 / (theta / 2)
init_recomb = rho
init_migration = options.migration_rate
if options.outgroup:
    init_outgroup = (init_isolation_time+init_migration_time)*3

if options.emissionComplicated:
    if options.outgroup:
        base_log_likelihood = Likelihood(isolation_with_migration_model2.IsolationMigrationModel(no_migration_states, no_ancestral_states, outgroup=True), forwarders)
    elif options.constant_break_points:
        base_log_likelihood = Likelihood(isolation_with_migration_model2.IsolationMigrationModelConstantBreaks(no_migration_states+ no_ancestral_states, 
                                                                                                               breaktail=options.breakpoints_tail_pieces,
                                                                                                               breaktimes=options.breakpoints_time), forwarders)
    else:
        base_log_likelihood = Likelihood(isolation_with_migration_model2.IsolationMigrationModel(no_migration_states, no_ancestral_states), forwarders)
else:
    base_log_likelihood = Likelihood(IsolationMigrationModel(no_migration_states, no_ancestral_states), forwarders)
initial_parameters = (init_isolation_time, init_migration_time, init_coal, init_recomb, init_migration)
if options.outgroup:
    initial_parameters = (init_isolation_time, init_migration_time, init_coal, init_recomb, init_migration, init_outgroup)

if options.optimizer=="Particle-Swarm":
    init_parameters=initial_parameters
    parm_scale_dictionary={}
    if options.fix_parameters_to_be_equal:
        groups=options.fix_parameters_to_be_equal.split(":")
        for group in groups:
            members=map(int,group.split(","))
            leader=members[0]
            for i in members[1:]:
                parm_scale_dictionary[i]=(leader, float(init_parameters[i])/float(init_parameters[leader]) )
    
    #this will be a list of the form [(1,1.0),(2,1.0),('fixed',1120.2),(3,1.0), (3,1.0),(3,0.5)... ]. 
    #On index i of this list, eh[i][0] will be what completely variable parameter one should use to get the ith parameter of the likelihood. 
    #If eh[i][0] is 'fixed', it means that one does not need the variable parameters to set it - and in that case its value is eh[i][1].
    #If eh[i][0] is a number it means that we find the likparameter from saying varpar[eh[i][0]]*eh[i][1]
    #So we have two different numberings of parameters. One is the parameters of the likelihood and the other is the parameters of the maximizer.
    #We have |maximize_parameters|<=|likelihood_parameters|. As an extra twist we have the transformed_maximize_parameters because the maximizer Particle-Swarm only works on (0,1)^{no_params}
    no_params=len(init_parameters) #m
    eh=[0]*no_params
    
    count_of_variableParameter=0
    for n in range(no_params):
        if n not in options.fix_params and n not in parm_scale_dictionary:
            eh[n]=(count_of_variableParameter,1.0)
            count_of_variableParameter+=1
    
    for n,tup in parm_scale_dictionary.items():
        eh[n]=(eh[tup[0]][0], tup[1])
    for n in options.fix_params:
        eh[n]=('fixed', init_parameters[n])
        
    print " ---------- Dimension of optimising space is ", count_of_variableParameter, " --------------"
    print eh
    
    class log_transformfunc:
        def __init__(self,fro,to):
            self.fro=fro
            self.to=to
            
        def __call__(self, num):
            return exp(num*log(self.to/self.fro)+log(self.fro))
        
        def valid_input(self, input):
            if input*log(self.to/self.fro)+log(self.fro)<500:
                return True
            return False
            
    class linear_transformfunc:    
        def __init__(self,scale,offset=0):
            self.scale=scale
            self.offset=offset
            
        def __call__(self, num):
            return num*self.scale+self.offset
        
        def valid_input(self, input):
            return True
        
    listOfTransforms=[]
    listOfTransforms.append(linear_transformfunc(0.01))#split.time
    listOfTransforms.append(linear_transformfunc(0.01))#mig.period_length
    listOfTransforms.append(log_transformfunc(10,1e6))#coalescence rate
    listOfTransforms.append(linear_transformfunc(1.0))#recombination rate
    listOfTransforms.append(log_transformfunc(0.1,10000))#migrationsparameter
    if options.outgroup:
        listOfTransforms.append(linear_transformfunc(0.10))#outgroup
    
    def from_maxvar_to_likpar(small_params):
        """
        input: list of parameters that are optimized freely by the optimizer
        output: list of parameters that goes into the likelihood function
        Before output, The variables will be transformed according to the transform, they should have.
        """
        big_params=[]
        for lik_param,(var_param, value) in enumerate(eh):
            if var_param=='fixed':
                big_params.append(value)
            else:
                small_param=small_params[var_param]
                if listOfTransforms[lik_param].valid_input(small_param):
                    big_params.append((listOfTransforms[lik_param](small_params[var_param]))*value)
                else:
                    return array([-1.0]*len(eh)) #this will result in something that will be rejected later
        return array(big_params)
    
    def from_likpar_to_maxvar(big_params):
        small_params=[0]*count_of_variableParameter
        for n,(var_param,value) in enumerate(eh):
            if var_param!='fixed':
                small_params[var_param]=big_params[n]/value
        return small_params
    
    if options.verbose:
        def log_likelihood(params):
            bparams=from_maxvar_to_likpar(params)
            try:
                val=base_log_likelihood(bparams)
            except:
                print "short_params",params
                print "big_params",bparams
                print "Unexpected error:", exc_info()[0]
                raise
            
            print str(bparams)+"="+str(val)
            return val
    else:
        def log_likelihood(params):
            bparams=from_maxvar_to_likpar(params)
            val=base_log_likelihood(bparams)
            print str(bparams)+"="+str(val)
            return val
    
else:
    if options.verbose:
        def log_likelihood(params):
            val=base_log_likelihood(params)
            print str(params)+"="+str(val)
            return val
    else:
        log_likelihood=base_log_likelihood


if options.optimizer=="Particle-Swarm":
    if options.parallels>1:
        op=OptimiserParallel()
        result = \
            op.maximise(log_likelihood, count_of_variableParameter, processes=options.parallels)
    else:
        op=Optimiser()
        result = \
            op.maximise(log_likelihood, count_of_variableParameter)
    mle_parameters=result.best.positions
else:
    if options.logfile:
        with open(options.logfile, 'w') as logfile:

            if options.header:
                print >> logfile, '\t'.join(['isolation.period', 'migration.period',
                                             'theta', 'rho', 'migration'])

            mle_parameters = \
                maximum_likelihood_estimate(log_likelihood, initial_parameters,
                                            log_file=logfile, optimizer_method=options.optimizer,
                                            log_param_transform=transform)
    else:


        mle_parameters = \
            maximum_likelihood_estimate(log_likelihood, initial_parameters,
                                        optimizer_method=options.optimizer)

max_log_likelihood = log_likelihood(mle_parameters)
with open(options.outfile, 'w') as outfile:
    if options.header:
        print >> outfile, '\t'.join(['isolation.period', 'migration.period',
                                     'theta', 'rho', 'migration', 'log.likelihood'])
    if options.optimizer=="Particle-Swarm":
        print >> outfile, '\t'.join(map(str, transform(from_maxvar_to_likpar(mle_parameters)) + (max_log_likelihood,)))
    else:
        print >> outfile, '\t'.join(map(str, transform(mle_parameters) + (max_log_likelihood,)))


