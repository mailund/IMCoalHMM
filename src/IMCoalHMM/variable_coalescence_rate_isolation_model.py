'''Code for constructing and optimizing the HMM for an isolation model.
'''

from numpy import zeros, identity, matrix
from numpy.testing import assert_almost_equal

from IMCoalHMM.isolation_model import Isolation2, make_rates_table_isolation
from IMCoalHMM.isolation_model import Single2, make_rates_table_single

from IMCoalHMM.CTMC import CTMC
from IMCoalHMM.transitions import CTMCSystem
from IMCoalHMM.transitions import compute_transition_probabilities
from IMCoalHMM.break_points import psmc_break_points
from IMCoalHMM.emissions import emission_matrix


## Code for computing HMM transition probabilities ####################
def _compute_through(ctmcs, break_points):
    '''Computes the matrices for moving through an interval'''
    no_states = len(break_points)

    # Construct the transition matrices for going through each interval
    through = [ctmcs[i].probability_matrix(break_points[i+1] - break_points[i])
               for i in xrange(no_states - 1)]

    # As a hack we set up a pseudo through matrix for the last interval that
    # just puts all probability on ending in one of the end states. This
    # simplifies the HMM transition probability code as it avoids a special case
    # for the last interval.
    pseudo_through = matrix(zeros((len(ctmcs[-1].state_space.states),
                                   len(ctmcs[-1].state_space.states))))
    pseudo_through[:, ctmcs[-1].state_space.end_states[0]] = 1.0
    through.append(pseudo_through)

    return through

def _compute_upto(isolation, ctmcs, break_points, through):
    '''Computes the probability matrices for moving from time zero up to,
    but not through, interval i.'''

    no_states = len(break_points)

    # Projection matrix needed to go from the isolation to the single
    # state spaces.
    projection = matrix(zeros((len(isolation.state_space.states),
                               len(ctmcs[0].state_space.states))))
    for state, isolation_index in isolation.state_space.states.items():
        ancestral_state = frozenset([(0, nucs) for (_, nucs) in state])
        ancestral_index = ctmcs[0].state_space.states[ancestral_state]
        projection[isolation_index, ancestral_index] = 1.0

    # We handle the first state as a special case because of the isolation
    # interval
    upto = [None] * no_states
    upto[0] = isolation.probability_matrix(break_points[0]) * projection
    for i in xrange(1, no_states):
        upto[i] = upto[i-1] * through[i-1]

    return upto

def _compute_between(ctmcs, through):
    '''Computes the matrices for moving from the end of interval i
    to the beginning of interval j.'''

    no_states = len(ctmcs)
    between = dict()
    # Transitions going from the endpoint of interval i to the entry point
    # of interval j
    for i in xrange(no_states-1):
        between[(i,i+1)] = matrix(identity(len(ctmcs[0].state_space.states)))
        for j in xrange(i+2, no_states):
            between[(i,j)] = between[(i,j-1)] * through[j-1]
    return between

class VariableCoalRateCTMCSystem(CTMCSystem):
    '''Wrapper around CTMC transition matrices for the isolation model.'''

    def __init__(self, isolation_ctmc, single_ctmcs, break_points):
        '''Construct all the matrices and cache them for the
        method calls.'''

        self.no_states_ = len(single_ctmcs)
        self.initial_ = isolation_ctmc.state_space.i12_index
        # Even though we have different CTMCs they have the same state space
        self.begin_states_ = single_ctmcs[0].state_space.begin_states
        self.left_states_ = single_ctmcs[0].state_space.left_states
        self.end_states_ = single_ctmcs[0].state_space.end_states

        self.through_ = _compute_through(single_ctmcs, break_points)
        self.upto_ = _compute_upto(isolation_ctmc, single_ctmcs,
                                   break_points, self.through_)
        self.between_ = _compute_between(single_ctmcs, self.through_)

    @property
    def no_states(self):
        "The number of states the HMM should have."
        return self.no_states_

    @property
    def initial(self):
        'The initial state index in the bottom-most matrix'
        return self.initial_

    def begin_states(self, i):
        'Begin states for interval i.'
        return self.begin_states_

    def left_states(self, i):
        'Left states for interval i.'
        return self.left_states_

    def end_states(self, i):
        'End states for interval i.'
        return self.end_states_

    def through(self, i):
        'Returns a probability matrix for going through interval i'
        return self.through_[i]

    def upto(self, i):
        '''Returns a probability matrix for going up to, but not
        through, interval i'''
        return self.upto_[i]

    def between(self, i, j):
        '''Returns a probability matrix for going from the
        end of interval i up to (but not through) interval j'''
        return self.between_[(i, j)]


## Class that can construct HMMs ######################################
from IMCoalHMM.transitions import compute_transition_probabilities
from IMCoalHMM.break_points import exp_break_points
from IMCoalHMM.emissions import emission_matrix

class VariableCoalescenceRateIsolationModel(object):
    '''Class wrapping the code that generates an isolation model HMM
    with variable coalescence rates in the different intervals.'''
    
    def __init__(self):
        '''Construct the model.
        
        This builds the state spaces for the CTMCs but the matrices for the
        HMM since those will depend on the rate parameters.'''
        super(VariableCoalescenceRateIsolationModel, self).__init__()
        self.isolation_state_space = Isolation2()
        self.single_state_space = Single2()
        

    def build_HMM(self, split_time, intervals, coal_rates, recomb_rate):
        '''Construct CTMCs and compute HMM matrices given the split time
        and the rates.
        
        The split time parameter is for setting a period where it is
        impossible for the two samples to coalesce (an isolation model).
        If it is set to 0.0 the system will work as Li & Durbin (2011)'s PSMC.
        
        The intervals list specifies how many intervals we should use for
        each coalescence rate. It is the sum over this list that will
        be the number of states.
        
        The coal_rates list should contain a coalescence rate for each interval
        in the model (except for the time up to split_time). It determins
        both the number of states and the transition probabilities.
        In optimisation it should be constrained somewhat since a free 
        rate for each interval will not be possible to estimate, but
        this is left to functionality outside the model.
        '''

        # We assume here that the coalescence rate is the same in the two
        # separate populations as in the ancestral just before teh split.
        # This is not necessarily true but it worked okay in simulations
        # in Mailund et al. (2011).
        
        isolation_rates = make_rates_table_isolation(coal_rates[0], coal_rates[0],
                                                     recomb_rate)
        isolation_ctmc = CTMC(self.isolation_state_space, isolation_rates)
        
        single_ctmcs = []
        for epoch, coal_rate in enumerate(coal_rates):
            single_rates = make_rates_table_single(coal_rate, recomb_rate)
            single_ctmc = CTMC(self.single_state_space, single_rates)
            for _ in xrange(intervals[epoch]):
                single_ctmcs.append(single_ctmc)

        no_states = len(single_ctmcs)
        break_points = psmc_break_points(n = no_states, offset = split_time)
        
        ctmc_system = VariableCoalRateCTMCSystem(isolation_ctmc, 
                                                 single_ctmcs, break_points)
        pi, T = compute_transition_probabilities(ctmc_system)
        E = emission_matrix(break_points, coal_rate)
        
        return pi, T, E


## Wrapper for maximum likelihood optimization ###############################
class MinimizeWrapper(object):
    '''Callable object wrapping the log likelihood computation for maximum
    liklihood estimation.'''
    
    def __init__(self, logL, intervals, est_split = False):
        '''Wrap the log likelihood computation with the non-variable parameter
        which is the number of states.'''
        self.logL = logL
        self.intervals = intervals
        self.est_split = est_split
        
    def __call__(self, parameters):
        '''Compute the likelihood in a paramter point. It computes -logL since
        the optimizer will minimize the function.'''
        if min(parameters) <= 0:
            return 1e18 # fixme: return infinity

        if self.est_split:
            # we are trying to estimate a split time as well
            split_time = parameters[0]
            coal_rates = parameters[1:-1]
            recomb_rate = parameters[-1]
        else:
            split_time = 0.0
            coal_rates = parameters[0:-1]
            recomb_rate = parameters[-1]

        return -self.logL(split_time, self.intervals, coal_rates, recomb_rate)




def main():
    '''Test'''

    model = VariableCoalescenceRateIsolationModel()
    split_time = 1.1
    intervals = [4] + [2]*25 + [4, 6]
    coal_rates = [1.0] * 28
    recomb_rate = 4e-4
    pi, T, E = model.build_HMM(split_time, intervals, coal_rates, recomb_rate)

    no_states = pi.getHeight()
    assert no_states == sum(intervals)

    pi_sum = 0.0
    for row in xrange(no_states):
        pi_sum += pi[row, 0]
    assert_almost_equal(pi_sum, 1.0)

    assert no_states == T.getWidth()
    assert no_states == T.getHeight()

    T_sum = 0.0
    for row in xrange(no_states):
        for col in xrange(no_states):
            T_sum += T[row, col]
    assert_almost_equal(T_sum, no_states)

    print 'Done'


if __name__ == '__main__':
    main()
