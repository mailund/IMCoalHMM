'''
Calculations of HMM transition probabilities for an isolation model.

'''

from numpy import zeros
from scipy import matrix
from numpy.testing import assert_almost_equal

def compute_transition_probabilities(isolation_ctmc,
                                     projection,
                                     single_ctmc,
                                     break_points):
    '''Calculate the HMM transition probabilities from the CTMCs.

    The isolation_ctmc covers the initial phase were it is not possible
    to coalesce, then the projection matrix maps states from this CTMC
    into the single_ctmc that models the panmictic ancestral population.

    The time break points are the times between intervals except that
    the first break point is the speciation event and nothing coalesces
    before this point, so there is one state per breakpoint, corresponding
    to the time intervals just after the break points.

    Returns both the stationary probability pi together with the transition
    probability T, since pi is autumatically calculated as part of
    the algorithm.
    '''

    # FIXME: the P matrices should be tabulated and reused of course

    no_states = len(break_points)
    initial = isolation_ctmc.state_space.i12_index
    Pr = projection # This just to have a shorthand for the matrix
    B_states = single_ctmc.state_space.B_states
    L_states = single_ctmc.state_space.L_states
    E_states = single_ctmc.state_space.E_states
    
    P0 = isolation_ctmc.probability_matrix(break_points[0]) * Pr

    J = matrix(zeros((no_states, no_states)))

    # -- Filling in the diagonal for the J matrix: i == j ---------------
    P1 = single_ctmc.probability_matrix(break_points[1]-break_points[0])
    J[0, 0] = (P0 * P1)[initial, E_states].sum()

    for i in xrange(1, no_states-1):
        P1 = single_ctmc.probability_matrix(break_points[i]-break_points[0])
        P2 = single_ctmc.probability_matrix(break_points[i+1]-break_points[i])
        P01 = (P0 * P1)
        prob = 0.0
        for b in B_states:
            for e in E_states:
                prob += P01[initial, b] * P2[b, e]
        J[i, i] = prob

    P = P0 * single_ctmc.probability_matrix(break_points[-1]-break_points[0])
    J[no_states-1, no_states-1] = P[initial, B_states].sum()


    # -- handle i < j (and j < i by symmetri) ---------------------------
    # i == 0 and 0 < j < no_states-1
    P1 = single_ctmc.probability_matrix(break_points[1]-break_points[0]) # [0,i]
    P01 = P0 * P1
    for j in xrange(1, no_states-1):
        # 0 < j < no_states-1
        P2 = single_ctmc.probability_matrix(break_points[j]-break_points[1]) # ]i,j[
        P3 = single_ctmc.probability_matrix(break_points[j+1] - break_points[j]) # [j]
        prob = 0.0
        for l1 in L_states:
            for l2 in L_states:
                for e in E_states:
                    prob += P01[initial, l1] * P2[l1, l2] * P3[l2, e]
        J[0, j] = J[j, 0] = prob

    # i == 0 and j == no_states-1
    P2 = single_ctmc.probability_matrix(break_points[no_states-1]-break_points[1])
    prob = 0.0
    for l1 in L_states:
        for l2 in L_states:
            prob += P01[initial, l1] * P2[l1, l2]
    J[0, no_states-1] = J[no_states-1, 0] = prob


    # i > 0
    for i in xrange(1, no_states-1):
        P1 = single_ctmc.probability_matrix(break_points[i]-break_points[0]) # [0;i[
        P2 = single_ctmc.probability_matrix(break_points[i+1]-break_points[i]) # [i]
        P01 = P0 * P1

        # 0 < j < no_states - 1
        for j in xrange(i+1, no_states-1):
            P3 = single_ctmc.probability_matrix(break_points[j]-break_points[i+1]) # ]i;j[
            P4 = single_ctmc.probability_matrix(break_points[j+1]-break_points[j]) # [j]
            prob = 0.0
            for b in B_states:
                for l1 in L_states:
                    for l2 in L_states:
                        for e in E_states:
                            prob += P01[initial, b] * P2[b, l1] * P3[l1, l2] * P4[l2, e]
            J[i, j] = J[j, i] = prob


        # j == no_states - 1
        P3 = single_ctmc.probability_matrix(break_points[no_states-1]-break_points[i+1])
        prob = 0.0
        for b in B_states:
            for l1 in L_states:
                for l2 in L_states:
                    prob += P01[initial, b] * P2[b, l1] * P3[l1, l2]
        J[i, no_states-1] = J[no_states-1, i] = prob

    assert_almost_equal(J.sum(), 1.0)

    # FIXME: Build a zipHMM matrix instead
    T = matrix(zeros((no_states, no_states)))
    pi = zeros(no_states)
    for row in xrange(no_states):
        pi[row] = J[row,].sum()
        T[row,] = J[row,] / pi[row]

    return pi, T


def main():
    '''Test'''

    from I2 import Isolation2, make_rates_table_isolation
    from I2 import Single2,    make_rates_table_single
    from CTMC import CTMC

    isolation_state_space = Isolation2()
    isolation_rates = make_rates_table_isolation(1, 0.5, 4e-4)
    isolation_ctmc = CTMC(isolation_state_space, isolation_rates)

    single_state_space = Single2()
    single_rates = make_rates_table_single(1.5, 4e-4)
    single_ctmc = CTMC(single_state_space, single_rates)


    Pr = matrix(zeros((len(isolation_state_space.states),
                       len(single_state_space.states))))
            
    def map_tokens(token):
        pop, nucs = token
        return 0, nucs

    for state, isolation_index in isolation_state_space.states.items():
        ancestral_state = frozenset(map(map_tokens, state))
        ancestral_index = single_state_space.states[ancestral_state]
        Pr[isolation_index, ancestral_index] = 1.0


    pi, T = compute_transition_probabilities(isolation_ctmc,
                                             Pr,
                                             single_ctmc,
                                             [1,2,3,4])
    print pi
    print
    print '=>', pi.sum()
    print
    print T
    print
    print '=>', T.sum()


if __name__ == '__main__':
    main()
