

from argparse import ArgumentParser
from IMCoalHMM.admixture_3haplos_mediumstep import Admixture3HModel
from IMCoalHMM.likelihood import Likelihood
from ParticleSwarm import OptimiserParallel, Optimiser
from pyZipHMM import Forwarder
from numpy import array
import math


"""
Run the main script. This makes the model
           123
            |
            |_______tau_3
           / \
          /   \_____tau_2
         |   / \
         |  /  |
         \_/___|____tau_1
          |    |
          |    |
         12    3


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
                    default="Particle-Swarm",
                    help="Optimization algorithm to use for maximizing the likelihood (Particle-Swarm)",
                    choices=['Particle-Swarm'])#,'Nelder-Mead', 'DEA', 'Genetic-Algorithm', 'TNC'])

optimized_params = [
    ('tau_1', 'admixture time in substitutions', 1e6 / 1e9),
    ('tau_2', 'Time of split between 2 and 3 where 2 is going into the admixed population, 12, and 3 is not.', 2e6/1e9),
    ('tau_3', 'Time of the deepest split time', 3e6 / 1e9),
    ('theta_12', 'effective population size in 4Ne substitutions for species 12 which only exists after admixture', 1e6 / 1e9),
    ('theta_1', 'effective population size in 4Ne substitutions for species 1 which exists from tau_1 to tau_3', 1e6 / 1e9),
    ('theta_2', 'effective population size in 4Ne substitutions for species 2 which exists from tau_1 to tau_2', 1e6 / 1e9),
    ('theta_3', 'effective population size in 4Ne substitutions for species 3 which exists from 0 to tau_2', 1e6 / 1e9),
    ('theta_23', 'effective population size in 4Ne substitutions for species 23 which exists from tau_2 to tau_3', 1e6 / 1e9),
    ('theta_123', 'effective population size in 4Ne substitutions for species 123 which exists from tau_3 to infinity', 1e6 / 1e9),
    ('p', 'proportion of population 12 going to population 2',0.1),
    ('rho', 'recombination rate in substitutions', 0.4)#,
    #('outgroup', 'total height of tree with outgroup', 1e6 / 1e9)
]

parser.add_argument('-a1A', '--alignments1A', nargs='+',
                    help='Alignments of 3 sequences between 2 from the non-admixed population and sequence A from the admixed population')
parser.add_argument('-a1B', '--alignments1B', nargs='+',
                    help='Alignments of 3 sequences between 2 from the non-admixed population and sequence B from the admixed population')
parser.add_argument('-a2A', '--alignments2A', nargs='+',
                    help='Alignments of 3 sequences between 2 from the admixed population and sequence A from the non-admixed population')
parser.add_argument('-a2B', '--alignments2B', nargs='+',
                    help='Alignments of 3 sequences between 2 from the admixed population and sequence B from the non-admixed population')


for parameter_name, description, default in optimized_params:
    parser.add_argument("--%s" % parameter_name,
                        type=float,
                        default=default,
                        help="Initial guess at the %s (%g)" % (description, default))
    
parser.add_argument('--intervals', nargs='+', default=[3,3,3,3], type=int, help='This is the setup of the 4 different epochs separated by tau_s. The ith position of the vector specifies how many breakpoints the ith position should have.')

parser.add_argument('--fix_params', nargs='+', default=[], type=int, help="the index of the parameters who will be fixed to their starting value throughout simulations. For now it only works when adap=1.")
parser.add_argument('--fix_parameters_to_be_equal', type=str, default="", help="A comma and colon separated string. commas separate within group and colons separate between groups. If a startWithGuessElaborate is specified this will use the relationsships between the parameters in that as fixed. ")
parser.add_argument('--parallels', type=int, default=1, help="If more than 1, the optimiser will run in parallel if possible(It is possible for Particle-swarm).")

options = parser.parse_args()
if len(options.alignments1A) < 1:
    parser.error("Input alignment for the 1A system not provided!")
if len(options.alignments1B) < 1:
    parser.error("Input alignment for the 1B system not provided!")
if len(options.alignments2A) < 1:
    parser.error("Input alignment for the 2A system not provided!")
if len(options.alignments2B) < 1:
    parser.error("Input alignment for the 2B system not provided!")

init_parameters = (
    options.tau_1,
    options.tau_2,
    options.tau_3,
    options.theta_12,
    options.theta_1,
    options.theta_2,
    options.theta_3,
    options.theta_23,
    options.theta_123,
    options.p,
    options.rho
)

names=['admixture_time', 'split_time_1','split_time_2', 'theta_admixed_pop', 'theta_long_way_admixed','theta_short_way_admixed','theta_nonadmixed','theta_short_ancestral', 'theta_ancestral','admix_prop', 'rho']


model1A = Admixture3HModel(Admixture3HModel.INITIAL_12, options.intervals[0], options.intervals[1], options.intervals[2], options.intervals[3])
model1B = Admixture3HModel(Admixture3HModel.INITIAL_12, options.intervals[0], options.intervals[1], options.intervals[2], options.intervals[3])
model2A = Admixture3HModel(Admixture3HModel.INITIAL_21, options.intervals[0], options.intervals[1], options.intervals[2], options.intervals[3])
model2B = Admixture3HModel(Admixture3HModel.INITIAL_21, options.intervals[0], options.intervals[1], options.intervals[2], options.intervals[3])
forwarders1A = [Forwarder.fromDirectory(arg) for arg in options.alignments1A]
forwarders1B = [Forwarder.fromDirectory(arg) for arg in options.alignments1B]
forwarders2A = [Forwarder.fromDirectory(arg) for arg in options.alignments2A]
forwarders2B = [Forwarder.fromDirectory(arg) for arg in options.alignments2B]
log_likelihood1A = Likelihood(model1A, forwarders1A)
log_likelihood1B = Likelihood(model1B, forwarders1B)
log_likelihood2A = Likelihood(model2A, forwarders2A)
log_likelihood2B = Likelihood(model2B, forwarders2B)

def eval_log_likelihood(params):
    return log_likelihood1A(params)+log_likelihood1B(params)+log_likelihood2A(params)+log_likelihood2B(params)

changers=[]
parm_scale_dictionary={}
if options.fix_parameters_to_be_equal:
    groups=options.fix_parameters_to_be_equal.split(":")
    for group in groups:
        members=map(int,group.split(","))
        leader=members[0]
        for i in members[1:]:
            changers.append(i)
            parm_scale_dictionary[i]=(leader, float(init_parameters[i])/float(init_parameters[leader]) )
startVal=[s for n,s in enumerate(init_parameters) if n not in changers and n not in options.fix_params]
print init_parameters
print startVal
print changers
print parm_scale_dictionary


     
def fullparams(parameters):
    fullparm=[] #the parameters to feed log likelihood with
    count=0
    for i in xrange(len(init_parameters)):#running through all the non-time-scaling parameters
        if i in options.fix_params: #
            fullparm.append(NeverChangeParam[i])
        elif i in changers:
            fullparm.append(-1)
        else:
            fullparm.append(parameters[count])
            count+=1
    fullparm=[f if f is not -1 else fullparm[parm_scale_dictionary[n][0]]*parm_scale_dictionary[n][1] for n,f in enumerate(fullparm)]
    fullparm.extend(parameters[count:])
    print fullparm
    return fullparm
test=fullparams(startVal)
#making a wrapper to take care of fixed parameters and scaling parameters
def lwrap(parameters):#parameters is the vector of only variable parameters
    val=eval_log_likelihood(array(fullparams(parameters)))
    if math.isnan(val):
        val = float('-inf')
    print val
    return val

if options.optimizer=="Particle-Swarm":
    if options.parallels>1:
        op=OptimiserParallel()
        mle_parameters = \
            op.maximise(lwrap, len(startVal), processes=7)
    else:
        op=Optimiser()
        mle_parameters = \
            op.maximise(lwrap, len(startVal))
else:
    print "no valid maximization scheme stated"
max_log_likelihood = lwrap(mle_parameters)        
with open(options.outfile, 'w') as outfile:
    print >> outfile, '\t'.join(names)
    print >> outfile, '\t'.join(map(str, fullparams(mle_parameters) + (max_log_likelihood,)))
