'''
Calculations of HMM transition probabilities for an isolation model.

'''

from numpy import zeros
from scipy import matrix
from numpy.testing import assert_almost_equal

from pyZipHMM import Matrix

def setup_CTMC_matrices(isolation, single, break_points):
    '''Build the matrices we need for constructing the HMM transition matrix
    and store them in a table using dynamic programming.'''

    # We have the same number of states as we have break points in this model
    # since the first break point corresponds to the speciation time.
    no_states = len(break_points)

    # Projection matrix needed to go from the isolation to the single
    # state spaces
    projection = matrix(zeros((len(isolation.state_space.states),
                               len(single.state_space.states))))
    for state, isolation_index in isolation.state_space.states.items():
        ancestral_state = frozenset([(0, nucs) for (_, nucs) in state])
        ancestral_index = single.state_space.states[ancestral_state]
        projection[isolation_index, ancestral_index] = 1.0

    # Construct the transition matrices for going through each interval
    through = [single.probability_matrix(break_points[i+1] - break_points[i])
               for i in xrange(no_states - 1)]

    # As a hack we set up a pseudo through matrix for the last interval that
    # just puts all probability on ending in one of the end states. This
    # simplifies the HMM transition probability code as it avoids a special case
    # for the last interval.
    pseudo_through = matrix(zeros((len(single.state_space.states),
                                   len(single.state_space.states))))
    pseudo_through[:, single.state_space.E_states[0]] = 1.0
    through.append(pseudo_through)

    # Transition matrices for going up to (but not through) interval i.
    # We handle the first state as a special case because of the isolation
    # interval
    upto = [None] * no_states
    upto[0] = isolation.probability_matrix(break_points[0]) * projection
    for i in xrange(1, no_states):
        upto[i] = upto[i-1] * through[i-1]

    # Transitions going from the endpoint of interval i to the entry point
    # of interval j
    between = dict(
        ((i, j), single.probability_matrix(break_points[j] - break_points[i+1]))
        for i in xrange(no_states - 1)
        for j in xrange(i+1, no_states)
    )

    return through, upto, between

def compute_transition_probabilities(isolation_ctmc,
                                     single_ctmc,
                                     break_points):
    '''Calculate the HMM transition probabilities from the CTMCs.

    The isolation_ctmc covers the initial phase were it is not possible
    to coalesce and the single_ctmc models the panmictic ancestral population.

    The time break points are the times between intervals except that
    the first break point is the speciation event and nothing coalesces
    before this point, so there is one state per breakpoint, corresponding
    to the time intervals just after the break points.

    Returns both the stationary probability pi together with the transition
    probability T, since pi is automatically calculated as part of
    the algorithm.
    '''

    no_states = len(break_points)
    initial = isolation_ctmc.state_space.i12_index
    B_states = single_ctmc.state_space.B_states
    L_states = single_ctmc.state_space.L_states
    E_states = single_ctmc.state_space.E_states

    through, upto, between = setup_CTMC_matrices(isolation_ctmc, single_ctmc, break_points)

    J = matrix(zeros((no_states, no_states)))

    # -- Filling in the diagonal for the J matrix ------------------------
    J[0, 0] = upto[1][initial, E_states].sum()
    for i in xrange(1, no_states-1):
        J[i, i] = sum(upto[i][initial, b] * through[i][b, e]
                      for b in B_states for e in E_states)
    J[no_states-1, no_states-1] = upto[no_states-1][initial, B_states].sum()


    # -- handle i < j (and j < i by symmetri) ---------------------------
    for i in xrange(no_states-1):
        # 0 < j < no_states - 1
        for j in xrange(i+1, no_states):
            prob = 0.0
            for b in B_states:
                for l1 in L_states:
                    for l2 in L_states:
                        for e in E_states:
                            prob += upto[i][initial, b] * through[i][b, l1] \
                                    * between[(i, j)][l1, l2] * through[j][l2, e]
            J[i, j] = J[j, i] = prob

    assert_almost_equal(J.sum(), 1.0)

    pi = Matrix(no_states, 1)
    T = Matrix(no_states, no_states)
    for row in xrange(no_states):
        pi[row, 0] = J[row,].sum()
        for col in xrange(no_states):
            T[row, col] = J[row, col] / pi[row, 0]

    return pi, T


def main():
    '''Test'''

    from I2 import Isolation2, make_rates_table_isolation
    from I2 import Single2, make_rates_table_single
    from CTMC import CTMC

    isolation_ctmc = CTMC(Isolation2(), make_rates_table_isolation(1, 0.5, 4e-4))
    single_ctmc = CTMC(Single2(), make_rates_table_single(1.5, 4e-4))

    pi, T = compute_transition_probabilities(isolation_ctmc, single_ctmc, [1, 2, 3, 4])

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
