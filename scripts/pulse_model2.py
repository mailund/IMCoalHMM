#!/usr/bin/env python
'''
Created on Apr 20, 2015

@author: svendvn
'''
from IMCoalHMM.transitions import CTMCSystem, compute_upto, compute_between, projection_matrix, compute_transition_probabilities
from IMCoalHMM.model import Model
from IMCoalHMM.statespace_generator import CoalSystem
# from IMCoalHMM.state_spaces import make_rates_table_single
from IMCoalHMM.CTMC import make_ctmc
# from IMCoalHMM.break_points import uniform_break_points, exp_break_points
from break_points2 import gamma_break_points
from IMCoalHMM.admixture import outer_product, powerset, complement, population_lineages
# 
from numpy import zeros, matrix, identity, ix_, exp, diff, cumsum, array, ndarray,concatenate
from numpy import sum as matrixsum
from numpy import dot as matrixdot
# from numpy.testing import assert_almost_equal
# import numpy
from pyZipHMM import Matrix
# from copy import deepcopy
# numpy.set_printoptions(threshold=numpy.nan)
#from pympler import tracker
from bisect import bisect
from emissions2 import emission_matrix7
from sympy.polys.polytools import intervals



def printPyZipHMM(Matrix):
    finalString=""
    for i in range(Matrix.getHeight()):
        for j in range(Matrix.getWidth()):
            finalString=finalString+" "+str(Matrix[i,j])
        finalString=finalString+"\n"
    return finalString


def admixture_state_space_map(from_space, to_space, p, q):
    """Constructs the mapping matrix from the 'from_space' state space to the 'to_space' state space
    assuming an admixture event where lineages in population 0 moves to population 1 with probability p
    and lineages in population 1 moves to population 0 with probability q."""
#     if q==0:
#         if p==0:
#             return matrix(identity(len(from_space)))  #This is a little bit cheating because from_space=to_space
#         elif p==1:
#             def state_map(state):
#                 return frozenset([(2, nucleotides) if i == 2 or i==3 else (1,nucleotides) for (i, nucleotides) in state])
#         return admixture_state_space_map_one_way(from_space, to_space, p)
    destination_map = to_space.state_numbers
    map_matrix = matrix(zeros((len(from_space.states), len(to_space.states))))

    for state, from_index in from_space.state_numbers.items():
        population_1 = population_lineages(1, state)
        population_2 = population_lineages(2, state)

        # <debug>
        #print pretty_state(state)
        # </debug>
        total_prob = 0.0

        for x, y in outer_product(powerset(population_1), powerset(population_2)):
            cx = complement(population_1, x)
            cy = complement(population_2, y)

            ## Keep x and y in their respective population but move the other two...
            cx = frozenset((2, lin) for (p, lin) in cx)
            cy = frozenset((1, lin) for (p, lin) in cy)

            destination_state = frozenset(x).union(cx).union(y).union(cy)
            change_probability = p**len(cx) * (1.0 - p)**len(x) * q**len(cy) * (1.0 - q)**len(y)
            to_index = destination_map[destination_state]

            # <debug>
            #print '->', pretty_state(destination_state),
            #print "p^{} (1-p)^{} q^{} (1-q)^{}".format(len(cx), len(x), len(cy), len(y))
            #print from_index, '->', to_index, '[{}]'.format(change_probability)
            # </debug>

            map_matrix[from_index, to_index] = change_probability
            total_prob += change_probability

        # <debug>
        #print
        #print total_prob
        # </debug>

        # We want to move to another state with exactly probability 1.0
        assert abs(total_prob - 1.0) < 1e-10

    return map_matrix



class PulseStateSpace(CoalSystem):

    def __init__(self,starting_position):
        """Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are). This is the time interval after(in model, 
        we measure time backwards, so we say 'after' but in normal life we would say 'before') admixture. It is before divergence of  2 and 3"""

        super(PulseStateSpace, self).__init__()

        self.transitions = [[('R', self.recombination)], [('C', self.coalesce)]]
        self.state_type = dict()

        left_1 = [frozenset([(1, (frozenset([1]), frozenset([])))]), frozenset([(2, (frozenset([1]), frozenset([])))])]
        right_1 = [frozenset([(1, (frozenset([]), frozenset([1])))]), frozenset([(2, (frozenset([]), frozenset([1])))])]
        left_2 = [frozenset([(1, (frozenset([2]), frozenset([])))]), frozenset([(2, (frozenset([2]), frozenset([])))])]
        right_2 = [frozenset([(1, (frozenset([]), frozenset([2])))]), frozenset([(2, (frozenset([]), frozenset([2])))])]
            
        self.init = [l1 | r1 | l2 | r2  for l1 in left_1 for r1 in right_1 for l2 in left_2 for r2 in right_2]

        self.compute_state_space()
        
        i11_state = frozenset([(1, (frozenset([sample]), frozenset([sample])))
                               for sample in [1, 2]])
        i22_state = frozenset([(2, (frozenset([sample]), frozenset([sample])))
                               for sample in [1, 2]])
        i12_state = frozenset([(sample, (frozenset([sample]), frozenset([sample])))
                               for sample in [1, 2]])

        self.i11_index = self.states[i11_state]
        self.i12_index = self.states[i12_state]
        self.i22_index = self.states[i22_state]
            
            
def make_rates_table(coal_rate_1,coal_rate_2, recombination_rate):
    table = dict()
    table[('C', 1, 1)] = coal_rate_1
    table[('C', 2, 2)] = coal_rate_2
    table[('R', 1, 1)] = recombination_rate
    table[('R', 2, 2)] = recombination_rate
    return table
    

class PulseCTMCSystem(CTMCSystem):
    
    def __init__(self, model, ctmcs, break_points, index_of_pulses, alphas, betas):
        '''
        Makes a Pulse-CTMCsystem. 
        input ctmcs: list of ctmcs of length n
        input break_points: list of break_points ogf length m, m>=n
        index_of_pulses: list of indexes in the range 1,...,m. The length should be n-1.
        '''
        
        super(PulseCTMCSystem, self).__init__(no_hmm_states=len(break_points), initial_ctmc_state=model.initial_state)  #what to do with initial state?
        
        #saving for later
        self.index_of_pulses=index_of_pulses
        self.ctmcs = ctmcs
        self.break_points = break_points
        self.model=model
        
        self.through_=[]
        
#         print "ctmcs",ctmcs
#         print "alphas",alphas
#         print "betas",betas
#         print "break_points", len(break_points)
#         print "index_of_pulses", index_of_pulses
#         print "len(ctmcs)", len(ctmcs)
        index_of_pulses0=[0]+index_of_pulses.tolist()
        for i in range(len(index_of_pulses0)-1):
            for j in range(index_of_pulses0[i],index_of_pulses0[i+1]):
                self.through_.append(   ctmcs[i].probability_matrix(break_points[j+1] - break_points[j])  )

        
        for j in range(index_of_pulses0[-1],len(break_points)-1):
            self.through_.append(ctmcs[-1].probability_matrix(break_points[j+1] - break_points[j]))
            
        for n,i in enumerate(index_of_pulses):
            projection=admixture_state_space_map(ctmcs[n].state_space, ctmcs[n+1].state_space, alphas[n],betas[n])
            self.through_[i-1]= self.through_[i-1]*projection
            
        pseudo_through = matrix(zeros((len(self.ctmcs[-1].state_space.states), len(self.ctmcs[-1].state_space.states))))
        pseudo_through[:, self.ctmcs[-1].state_space.end_states[0]] = 1.0
        self.through_.append(pseudo_through)
        
            
        #constructing other lists of matrices
        upto0 = matrix(identity(len(ctmcs[0].state_space.states)))
        self.upto_ = compute_upto(upto0, self.through_)
        self.between_ = compute_between(self.through_)
    
    def index_to_ctmc(self, index):
        return bisect()
    
    def get_state_space(self, i):
        """Return the state space for interval i."""
        return self.ctmcs[0].state_space
        
        
    def through(self, i):
        return self.through_[i]

    def up_to(self, i):
        return self.upto_[i]

    def between(self, i, j):
        return self.between_[(i, j)]

        
        
        

def make_rates_table_admixture(coal_rate_1, coal_rate_2, recomb_rate):
    """Builds the rates table from the CTMC for the two-samples system
    for an isolation period."""
    table = dict()
    table[('C', 1, 1)] = coal_rate_1
    table[('C', 2, 2)] = coal_rate_2
    table[('R', 1, 1)] = recomb_rate
    table[('R', 2, 2)] = recomb_rate
    return table


## Class that can construct HMMs ######################################
class PulseModel(Model):
    """Class wrapping the code that generates an isolation model HMM
        with variable coalescence rates in the different intervals."""

    # Determines which initial state to start the CTMCs in
    INITIAL_11 = 0
    INITIAL_12 = 1
    INITIAL_22 = 2

    def __init__(self, initial_configuration, no_intervals, index_of_pulses, breaktimes, breaktail=0, time_modifier=None):
        """Construct the model.

        This builds the state spaces for the CTMCs but the matrices for the
        HMM since those will depend on the rate parameters."""
        super(PulseModel, self).__init__()

        assert initial_configuration in (PulseModel.INITIAL_11, PulseModel.INITIAL_12, PulseModel.INITIAL_22)
        
        self.initial_configuration=initial_configuration
        self.index_of_pulses=index_of_pulses
        self.no_epochs=len(index_of_pulses)+1
        self.intervals=diff([0]+self.index_of_pulses.tolist()+[no_intervals])
        print [0]+self.index_of_pulses+[no_intervals]
        print "self.intervals",self.intervals,"self.index_of_pulses",self.index_of_pulses

        self.pulse_state_spaces=[]
        for i in range(self.no_epochs):
            self.pulse_state_spaces.append(PulseStateSpace(self.INITIAL_12))
            
            
        if initial_configuration == self.INITIAL_11:
            self.initial_state = self.pulse_state_spaces[0].i11_index
        elif initial_configuration == self.INITIAL_12:
            self.initial_state = self.pulse_state_spaces[0].i12_index
        elif initial_configuration == self.INITIAL_22:
            self.initial_state = self.pulse_state_spaces[0].i22_index
        else:
            assert False, "We should never reach this point!"
            
        
        self.no_states = no_intervals
        self.time_modifier=time_modifier
        self.breaktimes=breaktimes
        self.breaktail=breaktail
        
    def emission_points(self, *parameters):
        return 0
    
    def valid_parameters(self, parameters):
        """Predicate testing if a given parameter point is valid for the model.
        :param parameters: Model specific parameters
        :type parameters: numpy.ndarray
        :returns: True if all parameters are valid, otherwise False
        :rtype: bool
        """
      
        assert isinstance(parameters, ndarray)
        
        if not all(parameters >= 0):  # This is the default test, useful for most models.
            return False
        #print "parameters",parameters
        coal_rates_1, coal_rates_2, alphas,betas, recomb_rate,fixed_time_points = self.unpack_parameters(parameters)
        if not all(alphas<=1):
            return False
        if not all(betas<=1):
            return False
        
        #Here we check if all fixed_time_points leave a positive gap between them
        for i,ti in fixed_time_points:
            for j,tj in fixed_time_points:
                if i>j and ti<=tj:
                    return False
                if j>i and tj<=ti:
                    return False
        return True
        
        
    def unpack_parameters(self, parameters):
        """Unpack the rate parameters for the model from the linear representation
        used in optimizations to the specific rate parameters.
        """
        no_epochs = len(self.intervals)
        coal_rates_1 = parameters[0:self.no_epochs]
        coal_rates_2 = parameters[self.no_epochs:(2*self.no_epochs)]
        mig_rates_12 = parameters[(2*self.no_epochs):(3*self.no_epochs-1)]
        mig_rates_21 = parameters[(3*self.no_epochs-1):(4*self.no_epochs-2)]
        recomb_rate = parameters[4*self.no_epochs-2]
        if self.no_epochs*4-1<len(parameters):
            assert len(parameters[(len(coal_rates_1)*4-1):])>0, "Wrong number of parameters encountered. Did you remember to specify all the parameters with startWithElaborate Guess?"
            fixed_time_points=self.time_modifier(parameters[(len(coal_rates_1)*4-1):])
        elif self.time_modifier is not None:
            fixed_time_points=self.time_modifier()
        else:
            fixed_time_points=[]
        return coal_rates_1, coal_rates_2, mig_rates_12, mig_rates_21, recomb_rate, fixed_time_points

#break_points, params,intervals,  ctmc_system,offset=0.0, ctmc_postpone=0
    def build_ctmc_system(self, *parameters):
        """Construct CTMCs and compute HMM matrices given the admixture time, split time time,
        and the rates.
        """
        coal_rates_1, coal_rates_2, alphas, betas, recomb_rate,fixed_time_points = self.unpack_parameters(parameters)

        self.coal_rates_1=coal_rates_1
        self.coal_rates_2=coal_rates_2
        ctmcs=[]
        for i in range(self.no_epochs):
            rates=make_rates_table(coal_rates_1[i], coal_rates_2[i], recomb_rate)
            ctmcs.append(make_ctmc(self.pulse_state_spaces[i],rates))
            

        break_points=gamma_break_points(self.no_states,beta1=0.001*self.breaktimes,alpha=2,
                                        beta2=0.001333333*self.breaktimes, tenthsInTheEnd=self.breaktail, fixed_time_points=fixed_time_points)

        # FIXME: depends on initial configuration option to the model...
        return PulseCTMCSystem(self, ctmcs, break_points, self.index_of_pulses, alphas, betas)
    
    def build_hidden_markov_model(self, parameters):
        ctmc_system = self.build_ctmc_system(*parameters)
        coal_rates_1, coal_rates_2, alphas, betas, recomb_rate,fixed_time_points = self.unpack_parameters(parameters)
        initial_probs, transition_probs = compute_transition_probabilities(ctmc_system) #this code might throw a runtimeerror because NaNs are produced. If they are produced, they should be fixed later.
        br=ctmc_system.break_points
        
        
        
        params=coal_rates_1.tolist()+coal_rates_2.tolist()+[0.0]*(2*len(coal_rates_1))+[0.0]
        
#         print self.intervals
#         print cumsum(array(self.intervals))
#         print len(br)
#         print params
#         
#         for j in range(len(br)):
#             print j,bisect(cumsum(array(self.intervals)), j)
        if self.initial_configuration==self.INITIAL_12:
            minimum_NonZeroEntry=-1
            for ind_number, index in enumerate(self.index_of_pulses):
                if alphas[ind_number]>0 or betas[ind_number]>0:
                    minimum_NonZeroEntry=index
                    break
            assert minimum_NonZeroEntry>=0, "There was no admixture present at all"
            #print "i am here", minimum_NonZeroEntry
            for i in xrange(minimum_NonZeroEntry):
                for j in xrange(len(br)):
                    if i>=j:
                        transition_probs[i,j]=0.0
                    else:
                        transition_probs[i,j]=float(1.0)/(len(br)-i-1)
        
        emiss_prob= emission_matrix7(break_points=br, params=params,intervals=self.intervals,  ctmc_system=ctmc_system)
        
        return initial_probs, transition_probs, emiss_prob, br
        
        
if __name__ == '__main__':
    substime_first_change=0.0005
    substime_second_change=0.0010
    substime_third_change=0.0030
    def time_modifier(t):
        print t
        return [(5,substime_first_change*t[0]),(10,substime_second_change)]
    ad=PulseModel(PulseModel.INITIAL_12, no_intervals=20, index_of_pulses=array([5,10,15]), breaktimes=1.0, breaktail=5, time_modifier=None)
    testParams2=[  8.13577869e+03 ,  1.87046487e+03 ,  3.69842619e+02  , 1.53735829e+04,
   9.41145544e+02 ,  1.41385396e+02,   6.71078803e+02 ,  4.83890546e+02,
   1.36369900e-01,   7.17341103e-01,   9.78034229e-01,   2.86343328e-01,
   7.53933794e-01,   8.21405069e-01,   5.21836511e-01]
    testParams3=array([0.000863596795993   ,    0.00310780495672    ,    0.000629449575272   ,    0.00189871279872  ,      0.00120253154653   ,
                      0.00953059649538   ,     0.00182727180054 ,       0.00122482695789 ,       0.855562449347 , 0.397245477065, 0.7859663152,
                      0.169553864268 , 0.994536120807,  0.979299188747,  1.44802078062 ,  7.13453720054 ])
    testParams=array([1000.0]*8+[0.1]*2+[1.0]+[0.2]*2+[0.0]+[0.4])
    print testParams2
    param1=[7.98440550e+07 ,  8.61542658e+04 ,  4.48464377e+03   ,5.88901751e+03,
   4.82655337e+05,   2.06577188e+09  , 1.39947388e+01  , 7.96313869e+04,
   8.49173833e-02,   1.03104476e-05 , 1.00000000e+00  , 1.97424860e-09,
   3.51565665e-01,   0.00000000e+00 ,  8.56107672e-10]
    param=array(testParams)
    i,t,e,b=ad.build_hidden_markov_model(param)
    
    print printPyZipHMM(i)
    print printPyZipHMM(t)
    print printPyZipHMM(e)
    
    if True:
        from IMCoalHMM.model import Model
        from likelihood2 import Likelihood
        from pyZipHMM import Forwarder
        
        pathToSim="/home/svendvn/IMCoalHMM-simulations.24539"
        a11s=[pathToSim + "/alignment."+ s+".ziphmm" for s in ["1.11","2.11"]]
        a12s=[pathToSim + "/alignment."+ s+".ziphmm" for s in ["1.12","2.12"]]
        a22s=[pathToSim + "/alignment."+ s+".ziphmm" for s in ["1.22","2.22"]]
        
        model11=PulseModel(PulseModel.INITIAL_11, no_intervals=20, index_of_pulses=array([5,10,15]), breaktimes=1.0, breaktail=5, time_modifier=None)
        model12=PulseModel(PulseModel.INITIAL_12, no_intervals=20, index_of_pulses=array([5,10,15]), breaktimes=1.0, breaktail=5, time_modifier=None)
        model22=PulseModel(PulseModel.INITIAL_22, no_intervals=20, index_of_pulses=array([5,10,15]), breaktimes=1.0, breaktail=5, time_modifier=None)
        forwarders11=[Forwarder.fromDirectory(a11) for a11 in a11s]
        forwarders12=[Forwarder.fromDirectory(a12) for a12 in a12s]
        forwarders22=[Forwarder.fromDirectory(a22) for a22 in a22s]
        likeli11=Likelihood(model11, forwarders11)
        likeli12=Likelihood(model12, forwarders12)
        likeli22=Likelihood(model22, forwarders22)
        
        print likeli11(param)
        print likeli12(param)
        print likeli22(param)
        
