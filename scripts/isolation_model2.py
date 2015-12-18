"""Code for constructing and optimizing the HMM for an isolation model.
"""

from numpy import zeros, matrix, array, ndarray
from numpy.testing import assert_almost_equal

from IMCoalHMM.state_spaces import Isolation, make_rates_table_isolation
from IMCoalHMM.state_spaces import Single, make_rates_table_single
from IMCoalHMM.CTMC import make_ctmc
from IMCoalHMM.transitions import CTMCSystem, projection_matrix, compute_upto, compute_between,compute_transition_probabilities
from emissions2 import coalescence_points, emission_matrix6, emission_matrix, printPyZipHMM, emission_matrix8
from IMCoalHMM.break_points import exp_break_points,uniform_break_points
from IMCoalHMM.model import Model


## Code for computing HMM transition probabilities ####################

# The way multiprocessing works means that we have to define this class for mapping in parallel
# and we have to define the processing pool after we define the class, or it won't be able to see
# it in the sub-processes. It breaks the flow of the code, but it is necessary.

class ComputeThroughInterval(object):
    def __init__(self, single, break_points):
        self.single = single
        self.break_points = break_points

    def __call__(self, i):
        return self.single.probability_matrix(self.break_points[i + 1] - self.break_points[i])


def _compute_through(single, break_points):
    """Computes the matrices for moving through an interval"""
    no_states = len(break_points)

    # Construct the transition matrices for going through each interval
    through = map(ComputeThroughInterval(single, break_points),  range(no_states-1))

    # As a hack we set up a pseudo through matrix for the last interval that
    # just puts all probability on ending in one of the end states. This
    # simplifies the HMM transition probability code as it avoids a special case
    # for the last interval.
    # noinspection PyCallingNonCallable
    pseudo_through = matrix(zeros((len(single.state_space.states),
                                   len(single.state_space.states))))
    pseudo_through[:, single.state_space.end_states[0]] = 1.0
    through.append(pseudo_through)

    return through


def _compute_upto0(isolation, single, break_points):
    """Computes the probability matrices for moving to time zero."""

    def state_map(state):
        return frozenset([(0, nucs) for (_, nucs) in state])

    projection = projection_matrix(isolation.state_space, single.state_space, state_map)
    return isolation.probability_matrix(break_points[0]) * projection


class IsolationCTMCSystem(CTMCSystem):
    """Wrapper around CTMC transition matrices for the isolation model."""

    def __init__(self, isolation_ctmc, ancestral_ctmc, break_points):
        """Construct all the matrices and cache them for the
        method calls.

        :param isolation_ctmc: CTMC for the isolation phase.
        :type isolation_ctmc: IMCoalHMM.CTMC.CTMC
        :param ancestral_ctmc: CTMC for the ancestral population.
        :type ancestral_ctmc: IMCoalHMM.CTMC.CTMC
        :param break_points: List of break points between intervals.
        :type break_points: list[int]
        """

        super(IsolationCTMCSystem, self).__init__(no_hmm_states=len(break_points),
                                                  initial_ctmc_state=isolation_ctmc.state_space.i12_index)

        self.ancestral_ctmc = ancestral_ctmc
        self.through_ = _compute_through(ancestral_ctmc, break_points)
        self.upto_ = compute_upto(_compute_upto0(isolation_ctmc, ancestral_ctmc, break_points), self.through_)
        self.between_ = compute_between(self.through_)

    def get_state_space(self, i):
        """Return the state space for interval i. In this case it is always the
        ancestral state space.

        :rtype: Single
        """
        return self.ancestral_ctmc.state_space


## Class that can construct HMMs ######################################
class IsolationModel(Model):
    """Class wrapping the code that generates an isolation model HMM."""

    # noinspection PyMethodMayBeStatic
    def valid_parameters(self, parameters):
        """Predicate testing if a given parameter point is valid for the model.
        :param parameters: Model specific parameters
        :type parameters: numpy.ndarray
        :returns: True if all parameters are valid, otherwise False
        :rtype: bool
        """
        # This works but pycharm gives a type warning... I guess it doesn't see > overloading
        assert isinstance(parameters, ndarray), "the argument parameters="+str(parameters)+ " is not an numpy.ndarray but an "+str(type(parameters))
        # noinspection PyTypeChecker
        
        
        if parameters[1]<1e-8: #checking specifically for the coalescense rate
            return False
        #checking the outgroup is larger than the split time
        if self.outgroup:
            if parameters[3]<parameters[0]:
                return False
        
        return all(parameters >= 0)

    def __init__(self, no_hmm_states, outgroup=False):
        """Construct the model.

        This builds the state spaces for the CTMCs but not the matrices for the
        HMM since those will depend on the rate parameters."""
        super(IsolationModel, self).__init__()
        self.no_hmm_states = no_hmm_states
        self.isolation_state_space = Isolation()
        self.single_state_space = Single()
        self.outgroup=outgroup

    def emission_points(self, split_time, coal_rate, _):
        """Points to emit from."""
        break_points = exp_break_points(self.no_hmm_states, coal_rate, split_time)
        if self.outgroup:
            if break_points[-1]>self.outmax: #if the break points become illegal with the outgroup, we will change the breakpoints
                break_points=uniform_break_points(self.no_hmm_states, split_time, self.outmax-(self.outmax-split_time)/20.0)
        return coalescence_points(break_points, coal_rate)

    def build_ctmc_system(self, split_time, coal_rate, recomb_rate):
        """Construct CTMC system."""
        # We assume here that the coalescence rate is the same in the two
        # separate populations as it is in the ancestral. This is not necessarily
        # true but it worked okay in simulations in Mailund et al. (2011).
        isolation_rates = make_rates_table_isolation(coal_rate, coal_rate, recomb_rate)
        isolation_ctmc = make_ctmc(self.isolation_state_space, isolation_rates)
        single_rates = make_rates_table_single(coal_rate, recomb_rate)
        single_ctmc = make_ctmc(self.single_state_space, single_rates)
        break_points = exp_break_points(self.no_hmm_states, coal_rate, split_time)
        if self.outgroup:
            if break_points[-1]>self.outmax: #if the break points become illegal with the outgroup, we will change the breakpoints
                break_points=uniform_break_points(self.no_hmm_states, split_time, self.outmax-(self.outmax-split_time)/20.0)
        return IsolationCTMCSystem(isolation_ctmc, single_ctmc, break_points)
    
    
    def build_hidden_markov_model(self, parameters):
        """Build the hidden Markov model matrices from the model-specific parameters."""
        if len(parameters)==3: #that is, no outgroup
            split_time, coal_rate, recomb_rate = parameters
        elif len(parameters)==4:
            split_time, coal_rate, recomb_rate,outgroup = parameters
            self.outmax=outgroup
        else:
            assert False, "There number of parameters was wrong"
        ctmc_system = self.build_ctmc_system(split_time, coal_rate, recomb_rate)
        #changing the break_points
        break_points=exp_break_points(self.no_hmm_states, coal_rate, 0.0)
        if self.outgroup:
            if break_points[-1]>self.outmax: #if the break points become illegal with the outgroup, we will change the breakpoints
                print "Redone breakpoints"
                break_points=uniform_break_points(self.no_hmm_states, split_time, self.outmax-(self.outmax-split_time)/20.0)
        initial_probs, transition_probs = compute_transition_probabilities(ctmc_system)
        parameters2=[coal_rate]*2+[0.0]*2+[recomb_rate]
        intervals=[self.no_hmm_states]
#         emission_probs = emission_matrix(self.emission_points(*parameters))
#         print " ------------- Emis 0 --------------"
#         print printPyZipHMM(emission_probs)
        if self.outgroup:
            emission_probs = emission_matrix8(break_points, parameters2, outgroup, intervals, ctmc_system, split_time)
        else:
            emission_probs = emission_matrix6(break_points, parameters2, intervals, ctmc_system, split_time)
        
#         emission_probs = emission_matrix3(br, parameters, self.intervals)
#           
#         def printPyZipHMM(Matrix):
#             finalString=""
#             for i in range(Matrix.getHeight()):
#                 for j in range(Matrix.getWidth()):
#                     finalString=finalString+" "+str(Matrix[i,j])
#                 finalString=finalString+"\n"
#             return finalString
#         strToWirte=str(parameters)+"\n"+str("3:")+printPyZipHMM(emission_probs)+"\n"
        
#         strToWirte=strToWirte+str("4:")+printPyZipHMM(emission_probs)+"\n"+"initial_probs: "+printPyZipHMM(initial_probs)
#         emission_probs = emission_matrix3b(br, parameters, self.intervals,ctmc_system)
#         print strToWirte+str("3b:")+printPyZipHMM(emission_probs)
#         print " ------------- Emis 6 --------------"
#         print printPyZipHMM(emission_probs)
#         
        return initial_probs, transition_probs, emission_probs


def main():
    """Test"""

    model = IsolationModel(10, outgroup=True)
    print array([0.01, 0.0000000001, 0.01])
    print model.valid_parameters(array([0.01, 0.0000001, 0.01,0.015]))
    pi, trans_probs, emis_probs = model.build_hidden_markov_model((0.0001, 100, 0.4,0.0125))

    def printPyZipHMM(Matrix):
        finalString=""
        for i in range(Matrix.getHeight()):
            for j in range(Matrix.getWidth()):
                finalString=finalString+" "+str(Matrix[i,j])
            finalString=finalString+"\n"
        return finalString

    print "--------- EMISS ---------"
    print printPyZipHMM(emis_probs)
    print "------- TRANS -----------"
    print printPyZipHMM(trans_probs)



    

#     no_states = pi.getHeight()
#     assert no_states == 4
# 
#     pi_sum = 0.0
#     for row in xrange(no_states):
#         pi_sum += pi[row, 0]
#     assert_almost_equal(pi_sum, 1.0)
# 
#     assert no_states == trans_probs.getWidth()
#     assert no_states == trans_probs.getHeight()
# 
#     trans_sum = 0.0
#     for row in xrange(no_states):
#         for col in xrange(no_states):
#             trans_sum += trans_probs[row, col]
#     assert_almost_equal(trans_sum, no_states)
# 
#     print 'Done'


if __name__ == '__main__':
    main()
