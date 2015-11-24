from argparse import ArgumentParser
from IMCoalHMM.ILS import ILSModel
from IMCoalHMM.likelihood import Likelihood
from ParticleSwarm import OptimiserParallel, Optimiser
from pyZipHMM import Forwarder
from numpy import array
import math
from math import isnan,log,exp


"""
Run the main script. This makes the model
             123
              |
              |____tau_2
             / \
   tau_1____/  |
           /|  |
          / |  |
         /  |  |
        1   2  3


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
    ('tau_1', '', 1e6 / 1e9),
    ('tau_2', '', 2e6/1e9),
    ('coal_1','', 1e3),
    ('coal_2','', 1e3),
    ('coal_3','', 1e3),
    ('coal_12', '', 1e3),
    ('coal_123', '', 1e3),
    ('rho', 'recombination rate in substitutions', 0.4)#,
    #('outgroup', 'total height of tree with outgroup', 1e6 / 1e9)
]

parser.add_argument('-a', '--alignments', nargs='+',
                    help='3-sequence alignment')



for parameter_name, description, default in optimized_params:
    parser.add_argument("--%s" % parameter_name,
                        type=float,
                        default=default,
                        help="Initial guess at the %s (%g)" % (description, default))
    
parser.add_argument('--intervals', nargs='+', default=[5,5], type=int, help='This is the setup of the 4 different epochs separated by tau_s. The ith position of the vector specifies how many breakpoints the ith position should have.')

parser.add_argument('--fix_params', nargs='+', default=[], type=int, help="the index of the parameters who will be fixed to their starting value throughout simulations.")
parser.add_argument('--fix_parameters_to_be_equal', type=str, default="", help="A comma and colon separated string. commas separate within group and colons separate between groups. The ratios between all pairs of variables in one group is kept constant. If not all the parameters of one group are equal, this could cause problems because of the log transformations.")
parser.add_argument('--parallels', type=int, default=1, help="If more than 1, the optimiser will run in parallel if possible(It is possible for Particle-swarm).")

options = parser.parse_args()
if len(options.alignments) < 1:
    parser.error("Input alignment for the alignment system not provided!")


init_parameters = (
    options.tau_1,
    options.tau_2,
    options.coal_1,
    options.coal_2,
    options.coal_3,
    options.coal_12,
    options.coal_123,
    options.rho
)

names=['admixture_time', 'split_time_1','split_time_2', 'theta_admixed_pop', 'theta_long_way_admixed','theta_short_way_admixed','theta_nonadmixed','theta_short_ancestral', 'theta_ancestral','admix_prop', 'rho']


model = ILSModel(options.intervals[0], options.intervals[1])
forwarders = [Forwarder.fromDirectory(arg) for arg in options.alignments]
log_likelihood = Likelihood(model, forwarders)



#For this part we will

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
    
## Not used function but is here because it has great explanation value
def transform_to_optimise_space(fullparam):
    tau_1,tau_2,coal_12,coal_123,rho= fullparam
    def log_transformfunc(fro,to):
        def transform(num):
            return (log(num)-log(fro))/log(to/fro)
        return transform
    coal_trans=log_transformfunc(100.0, 10000.0)
    def time_trans(num):
        num*100.0
    return [time_trans(tau_1),time_trans(tau_2),coal_trans(coal_12),coal_trans(coal_123),rho]

## Not used function but is here because it has great explanation value      
def transform_from_optimise_space(fullparm):
    tau_1t,tau_2t,coal_12t,coal_123t,rhot= fullparam
    def log_transformfunc(fro,to):
        def transform(num):
            return exp(num*log(to/fro)+log(fro))
        return transform
    def time_trans(num):
        return num*0.01
    coal_trans=log_transformfunc(100.0, 10000.0)
    res=[time_trans(tau_1t),time_trans(tau_2t),coal_trans(coal_12t),coal_trans(coal_123t),rhot]
    return res    
    
def log_transformfunc(fro,to):
    def transform(num):
        return exp(num*log(to/fro)+log(fro))
    return transform
def linear_transformfunc(scale,offset=0):    
    def transform(num):
        return num*scale+offset
    return transform
    
listOfTransforms=[linear_transformfunc(0.01),           #tau_1
                  linear_transformfunc(0.01),           #tau_2
                  log_transformfunc(10.0,10000.0),      #coal_1
                  log_transformfunc(10.0,10000.0),      #coal_2
                  log_transformfunc(10.0,10000.0),      #coal_3
                  log_transformfunc(10.0,10000.0),      #coal_12
                  log_transformfunc(10.0,10000.0),      #coal_123
                  linear_transformfunc(1.0)]            #rho(identity function)




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
            big_params.append(listOfTransforms[lik_param](small_params[var_param])*value)
    return big_params

def lwrap(small_parameters):#small_parameters is the vector of only variable parameters
    likelihood_parms=from_maxvar_to_likpar(small_parameters)
    val=log_likelihood(array(likelihood_parms))
    print val,"========", likelihood_parms
    if math.isnan(val):
        val = float('-inf')
    return val

if options.optimizer=="Particle-Swarm":
    if options.parallels>1:
        op=OptimiserParallel()
        result = \
            op.maximise(lwrap, no_params, processes=options.parallels)
    else:
        op=Optimiser()
        result = \
            op.maximise(lwrap, no_params)
    mle_parameters=result.best.positions
    print mle_parameters
else:
    print "no valid maximization scheme stated"
max_log_likelihood = lwrap(mle_parameters)        
with open(options.outfile, 'w') as outfile:
    print >> outfile, '\t'.join(names)
    print >> outfile, '\t'.join(map(str, from_maxvar_to_likpar(mle_parameters) + [max_log_likelihood]))



