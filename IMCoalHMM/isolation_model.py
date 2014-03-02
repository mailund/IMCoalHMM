'''Code for constructing and optimizing the HMM for an isolation model.
'''

from numpy import zeros, matrix
from numpy.testing import assert_almost_equal

from IMCoalHMM.statespace_generator import Single, Isolation
from IMCoalHMM.CTMC import CTMC
from IMCoalHMM.transitions import CTMCSystem
from IMCoalHMM.transitions import compute_transition_probabilities
from IMCoalHMM.break_points import exp_break_points
from IMCoalHMM.emissions import emission_matrix

## State space code ############################################
class Isolation2(Isolation):
    '''Class for IM system with exactly two samples.'''

    def __init__(self):
        '''Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are).'''

        super(Isolation2, self).__init__([1, 2])
        self.compute_state_space()

        i12_state = frozenset([(sample,
                                (frozenset([sample]), frozenset([sample])))
                                for sample in [1, 2]])
        self.i12_index = self.states[i12_state]

def make_rates_table_isolation(C1, C2, R):
    '''Builds the rates table from the CTMC for the two-samples system.'''
    table = dict()
    table[('C', 1, 1)] = C1
    table[('C', 2, 2)] = C2
    table[('R', 1, 1)] = R
    table[('R', 2, 2)] = R
    return table


class Single2(Single):
    '''Class for a merged ancestral population'''

    def __init__(self):
        '''Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are).'''

        super(Single2, self).__init__()
        self.compute_state_space()

def make_rates_table_single(C, R):
    '''Builds the rates table from the CTMC for the two-samples system.'''
    table = dict()
    table[('C', 0, 0)] = C
    table[('R', 0, 0)] = R
    return table

## Code for computing HMM transition probabilities ####################
def _compute_through(single, break_points):
    '''Computes the matrices for moving through an interval'''
    no_states = len(break_points)

    # Construct the transition matrices for going through each interval
    through = [single.probability_matrix(break_points[i+1] - break_points[i])
               for i in xrange(no_states - 1)]

    # As a hack we set up a pseudo through matrix for the last interval that
    # just puts all probability on ending in one of the end states. This
    # simplifies the HMM transition probability code as it avoids a special case
    # for the last interval.
    pseudo_through = matrix(zeros((len(single.state_space.states),
                                   len(single.state_space.states))))
    pseudo_through[:, single.state_space.end_states[0]] = 1.0
    through.append(pseudo_through)

    return through

def _compute_upto(isolation, single, break_points, through):
    '''Computes the probability matrices for moving from time zero up to,
    but not through, interval i.'''

    no_states = len(break_points)

    # Projection matrix needed to go from the isolation to the single
    # state spaces
    projection = matrix(zeros((len(isolation.state_space.states),
                               len(single.state_space.states))))
    for state, isolation_index in isolation.state_space.states.items():
        ancestral_state = frozenset([(0, nucs) for (_, nucs) in state])
        ancestral_index = single.state_space.states[ancestral_state]
        projection[isolation_index, ancestral_index] = 1.0

    # We handle the first state as a special case because of the isolation
    # interval
    upto = [None] * no_states
    upto[0] = isolation.probability_matrix(break_points[0]) * projection
    for i in xrange(1, no_states):
        upto[i] = upto[i-1] * through[i-1]

    return upto

def _compute_between(single, break_points):
    '''Computes the matrices for moving from the end of interval i
    to the beginning of interval j.'''

    no_states = len(break_points)
    # Transitions going from the endpoint of interval i to the entry point
    # of interval j
    return dict(
        ((i, j), single.probability_matrix(break_points[j] - break_points[i+1]))
        for i in xrange(no_states - 1)
        for j in xrange(i+1, no_states)
    )


class IsolationCTMCSystem(CTMCSystem):
    '''Wrapper around CTMC transition matrices for the isolation model.'''

    def __init__(self, isolation_ctmc, single_ctmc, break_points):
        '''Construct all the matrices and cache them for the
        method calls.'''

        self.no_states_ = len(break_points)
        self.initial_ = isolation_ctmc.state_space.i12_index
        self.begin_states_ = single_ctmc.state_space.begin_states
        self.left_states_ = single_ctmc.state_space.left_states
        self.end_states_ = single_ctmc.state_space.end_states

        self.through_ = _compute_through(single_ctmc, break_points)
        self.upto_ = _compute_upto(isolation_ctmc, single_ctmc, break_points, self.through_)
        self.between_ = _compute_between(single_ctmc, break_points)

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
class IsolationModel(object):
    '''Class wrapping the code that generates an isolation model HMM.'''
    
    def __init__(self):
        '''Construct the model.
        
        This builds the state spaces for the CTMCs but the matrices for the
        HMM since those will depend on the rate parameters.'''
        super(IsolationModel, self).__init__()
        self.isolation_state_space = Isolation2()
        self.single_state_space = Single2()
        

    def build_HMM(self, no_states, split_time, coal_rate, recomb_rate):
        '''Construct CTMCs and compute HMM matrices given the split time
        and the rates.'''

        isolation_rates = make_rates_table_isolation(coal_rate, coal_rate, recomb_rate)
        isolation_ctmc = CTMC(self.isolation_state_space, isolation_rates)
        single_rates = make_rates_table_single(coal_rate, recomb_rate)
        single_ctmc = CTMC(self.single_state_space, single_rates)
        
        break_points = exp_break_points(no_states, coal_rate, split_time)

        ctmc_system = IsolationCTMCSystem(isolation_ctmc, single_ctmc, break_points)

        pi, T = compute_transition_probabilities(ctmc_system)
        E = emission_matrix(break_points, coal_rate)

        return pi, T, E


## Wrapper for maximum likelihood optimization ###############################
class MinimizeWrapper(object):
    '''Callable object wrapping the log likelihood computation for maximum
    liklihood estimation.'''
    
    def __init__(self, logL, no_states):
        '''Wrap the log likelihood computation with the non-variable parameter
        which is the number of states.'''
        self.logL = logL
        self.no_states = no_states
        
    def __call__(self, parameters):
        '''Compute the likelihood in a paramter point. It computes -logL since
        the optimizer will minimize the function.'''
        if min(parameters) <= 0:
            return 1e18 # fixme: return infinity
        return -self.logL(self.no_states, *parameters)




def main():
    '''Test'''

    isolation_ctmc = CTMC(Isolation2(), make_rates_table_isolation(1, 0.5, 4e-4))
    single_ctmc = CTMC(Single2(), make_rates_table_single(1.5, 4e-4))
    ctmc_system = IsolationCTMCSystem(isolation_ctmc, single_ctmc, [1, 2, 3, 4])
    pi, T = compute_transition_probabilities(ctmc_system)

    no_states = pi.getHeight()
    assert no_states == 4

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
