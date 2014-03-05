'''Code for constructing and optimizing the HMM for an isolation model.
'''

from numpy import zeros, matrix, identity
from numpy.testing import assert_almost_equal


from IMCoalHMM.statespace_generator import Migration
from IMCoalHMM.CTMC import CTMC
from IMCoalHMM.transitions import CTMCSystem
from IMCoalHMM.transitions import compute_transition_probabilities
from IMCoalHMM.break_points import exp_break_points, uniform_break_points
from IMCoalHMM.emissions import emission_matrix

from IMCoalHMM.isolation_model import Isolation2, make_rates_table_isolation
from IMCoalHMM.isolation_model import Single2, make_rates_table_single


## State space code ############################################
class Migration2(Migration):
    '''Class for IM system with exactly two samples.'''

    def __init__(self):
        '''Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are).

        Also collects the indices in the state space for the three
        (realistic) initial states, with both chromosomes in population 1
        or in 2 or one from each.'''

        super(Migration2, self).__init__([1, 2])
        

        self.compute_state_space()

        i11_state = frozenset([(1,
                                (frozenset([sample]), frozenset([sample])))
                                for sample in [1, 2]])
        i22_state = frozenset([(2,
                                (frozenset([sample]), frozenset([sample])))
                                for sample in [1, 2]])
        i12_state = frozenset([(sample,
                                (frozenset([sample]), frozenset([sample])))
                                for sample in [1, 2]])

        self.i11_index = self.states[i11_state]
        self.i12_index = self.states[i12_state]
        self.i22_index = self.states[i22_state]


def make_rates_table_migration(C1, C2, R, M12, M21):
    '''Builds the rates table from the CTMC for the two-samples system.'''
    table = dict()
    table[('C', 1, 1)] = C1
    table[('C', 2, 2)] = C2
    table[('R', 1, 1)] = R
    table[('R', 2, 2)] = R
    table[('M', 1, 2)] = M12
    table[('M', 2, 1)] = M21
    return table


## Code for computing HMM transition probabilities ####################
def _compute_through(migration, migration_break_points, 
                     ancestral, ancestral_break_points):
    '''Computes the matrices for moving through an interval'''
    
    # Projection matrix needed to go from the migration to the single
    # state spaces
    projection = matrix(zeros((len(migration.state_space.states),
                               len(ancestral.state_space.states))))
    for state, isolation_index in migration.state_space.states.items():
        ancestral_state = frozenset([(0, nucs) for (_, nucs) in state])
        ancestral_index = ancestral.state_space.states[ancestral_state]
        projection[isolation_index, ancestral_index] = 1.0
    
    no_migration_states = len(migration_break_points)
    no_ancestral_states = len(ancestral_break_points)

    # Construct the transition matrices for going through each interval in
    # the migration phase
    migration_through = [migration.probability_matrix(migration_break_points[i+1] - migration_break_points[i])
                         for i in xrange(no_migration_states - 1)]
    last_migration = migration.probability_matrix(ancestral_break_points[0] - migration_break_points[-1]) \
                   * projection
    migration_through.append(last_migration)
    
    
    ancestral_through = [ancestral.probability_matrix(ancestral_break_points[i+1] - ancestral_break_points[i])
                         for i in xrange(no_ancestral_states - 1)]

    # As a hack we set up a pseudo through matrix for the last interval that
    # just puts all probability on ending in one of the end states. This
    # simplifies the HMM transition probability code as it avoids a special case
    # for the last interval.
    pseudo_through = matrix(zeros((len(ancestral.state_space.states),
                                   len(ancestral.state_space.states))))
    pseudo_through[:, ancestral.state_space.end_states[0]] = 1.0
    ancestral_through.append(pseudo_through)

    return migration_through + ancestral_through

def _compute_upto(isolation, migration, ancestral, break_points, through):
    '''Computes the probability matrices for moving from time zero up to,
    but not through, interval i.'''

    no_states = len(break_points)

    # Projection matrix needed to go from the isolation to the migration
    # state spaces
    projection = matrix(zeros((len(isolation.state_space.states),
                               len(migration.state_space.states))))
    for state, isolation_index in isolation.state_space.states.items():
        migration_index = migration.state_space.states[state]
        projection[isolation_index, migration_index] = 1.0

    # We handle the first state as a special case because of the isolation
    # interval
    upto = [None] * no_states
    upto[0] = isolation.probability_matrix(break_points[0]) * projection
    for i in xrange(1, no_states):
        upto[i] = upto[i-1] * through[i-1]

    return upto

def _compute_between(through):
    '''Computes the matrices for moving from the end of interval i
    to the beginning of interval j.'''

    no_states = len(through)
    between = dict()
    # Transitions going from the endpoint of interval i to the entry point
    # of interval j
    for i in xrange(no_states-1):
        between[(i,i+1)] = matrix(identity(through[i].shape[1]))
        for j in xrange(i+2, no_states):
            between[(i,j)] = between[(i,j-1)] * through[j-1]
    return between


class IsolationMigrationCTMCSystem(CTMCSystem):
    '''Wrapper around CTMC transition matrices for the isolation model.'''

    def __init__(self, isolation_ctmc, migration_ctmc, ancestral_ctmc,
                 migration_break_points, ancestral_break_points):
        '''Construct all the matrices and cache them for the
        method calls.'''

        self.no_migration_states = len(migration_break_points)
        self.no_ancestral_states = len(ancestral_break_points)
        self.no_states_ = self.no_migration_states + self.no_ancestral_states

        self.initial_ = isolation_ctmc.state_space.i12_index
        
        self.begin_states_ = [migration_ctmc.state_space.begin_states,
                              ancestral_ctmc.state_space.begin_states]
        self.left_states_  = [migration_ctmc.state_space.left_states,
                              ancestral_ctmc.state_space.left_states]
        self.end_states_   = [migration_ctmc.state_space.end_states,
                              ancestral_ctmc.state_space.end_states]

        break_points = list(migration_break_points) + list(ancestral_break_points)

        self.through_ = _compute_through(migration_ctmc, migration_break_points,
                                         ancestral_ctmc, ancestral_break_points)
        self.upto_ = _compute_upto(isolation_ctmc, migration_ctmc, ancestral_ctmc,
                                   break_points, self.through_)
        self.between_ = _compute_between(self.through_)

    def _is_ancestral(self, i):
        '''Is index i in the ancestral populations?'''
        return self.no_migration_states <= i

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
        return self.begin_states_[self._is_ancestral(i)]

    def left_states(self, i):
        'Left states for interval i.'
        return self.left_states_[self._is_ancestral(i)]

    def end_states(self, i):
        'End states for interval i.'
        return self.end_states_[self._is_ancestral(i)]

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
class IsolationMigrationModel(object):
    '''Class wrapping the code that generates an isolation model HMM.'''
    
    def __init__(self):
        '''Construct the model.
        
        This builds the state spaces for the CTMCs but not the matrices for the
        HMM since those will depend on the rate parameters.'''
        super(IsolationMigrationModel, self).__init__()
        self.isolation_state_space = Isolation2()
        self.migration_state_space = Migration2()
        self.single_state_space = Single2()
        

    def build_HMM(self, no_mig_states, no_ancestral_states,
                  isolation_time, migration_time, 
                  coal_rate, recomb_rate, mig_rate):
        '''Construct CTMCs and compute HMM matrices given the split times
        and the rates.'''

        # We assume here that the coalescence rate is the same in the two
        # separate populations as it is in the ancestral. This is not necessarily
        # true but it worked okay in simulations in Mailund et al. (2012).
        
        isolation_rates = make_rates_table_isolation(coal_rate, coal_rate, recomb_rate)
        isolation_ctmc = CTMC(self.isolation_state_space, isolation_rates)
        
        migration_rates = make_rates_table_migration(coal_rate, coal_rate, recomb_rate,
                                                     mig_rate, mig_rate)
        migration_ctmc = CTMC(self.migration_state_space, migration_rates)
        
        single_rates = make_rates_table_single(coal_rate, recomb_rate)
        single_ctmc = CTMC(self.single_state_space, single_rates)
        
        tau1 = isolation_time
        tau2 = isolation_time + migration_time
        migration_break_points = uniform_break_points(no_mig_states, tau1, tau2)
        ancestral_break_points = exp_break_points(no_ancestral_states, coal_rate, tau2)
        break_points = list(migration_break_points) + list(ancestral_break_points)
        
        ctmc_system = IsolationMigrationCTMCSystem(isolation_ctmc, migration_ctmc, single_ctmc,
                                                   migration_break_points, ancestral_break_points)

        pi, T = compute_transition_probabilities(ctmc_system)
        E = emission_matrix(break_points, coal_rate)

        return pi, T, E


## Wrapper for maximum likelihood optimization ###############################
class MinimizeWrapper(object):
    '''Callable object wrapping the log likelihood computation for maximum
    liklihood estimation.'''
    
    def __init__(self, logL, no_migration_states, no_ancestral_states):
        '''Wrap the log likelihood computation with the non-variable parameter
        which is the number of states.'''
        self.logL = logL
        self.no_migration_states = no_migration_states
        self.no_ancestral_states = no_ancestral_states
        
    def __call__(self, parameters):
        '''Compute the likelihood in a paramter point. It computes -logL since
        the optimizer will minimize the function.'''
        if min(parameters) <= 0:
            return 1e18 # fixme: return infinity
        return -self.logL(self.no_migration_states, self.no_ancestral_states, *parameters)




def main():
    '''Test'''

    no_mig_states = 4
    no_ancestral_states = 4
    isolation_time = 0.5
    migration_time = 1.0
    coal_rate = 1
    recomb_rate = 0.4
    mig_rate = 0.1

    model = IsolationMigrationModel()
    pi, T, E = model.build_HMM(no_mig_states, no_ancestral_states,
                               isolation_time, migration_time, 
                               coal_rate, recomb_rate, mig_rate)

    no_states = pi.getHeight()
    assert no_states == no_mig_states + no_ancestral_states

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
