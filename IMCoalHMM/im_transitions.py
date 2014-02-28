'''
Calculations of HMM transition probabilities for an isolation with
migration model.

FIXME: This is only half-done and only implements the migration phase
but does this from time zero an up... it shouldn't be too hard to fix,
though.
'''

from numpy import zeros
from scipy import matrix
from numpy.testing import assert_almost_equal

def compute_transition_probabilities(ctmc, break_points):
    '''Calculate the HMM transition probabilities from the CTMC.

    The time break points are the times between intervals, so there
    is one break point less than the number of states in the HMM.

    Returns both the stationary probability pi and the transition
    probability T, since pi is autumatically calculated as part of
    the algorithm.
    '''

    # FIXME: Here I'm only handling a single epoch and only a
    # migration system. It is a bit more work, but not that hard
    # to do, to generalize it.

    no_states = len(break_points) + 1
    initial = ctmc.state_space.i12_index # FIXME: generalize this
    B_states = ctmc.state_space.B_states
    L_states = ctmc.state_space.L_states
    E_states = ctmc.state_space.E_states

    # FIXME: the P matrices should be tabulated and reused of course
    J = matrix(zeros((no_states, no_states)))

    # == Filling in the diagonal for the J matrix: i == j ====
    P1 = ctmc.probability_matrix(break_points[0])
    J[0, 0] = P1[initial, E_states].sum()

    for i in xrange(1, no_states-1):
        P1 = ctmc.probability_matrix(break_points[i-1])
        P2 = ctmc.probability_matrix(break_points[i]-break_points[i-1])
        for b in B_states:
            for e in E_states:
                J[i, i] += P1[initial, b] * P2[b, e]

    P = ctmc.probability_matrix(break_points[-1])
    J[no_states-1, no_states-1] = P[initial, B_states].sum()

    # == handle i < j (and j < i by symmetri) =========
    # i == 0 and 0 < j < no_states-1
    P1 = ctmc.probability_matrix(break_points[0]) # [0,i]
    for j in xrange(1, no_states-1):
        # 0 < j < no_states-1
        P2 = ctmc.probability_matrix(break_points[j-1]-break_points[0]) # ]i,j[
        P3 = ctmc.probability_matrix(break_points[j] - break_points[j-1]) # [j]
        prob = 0.0
        for l1 in L_states:
            for l2 in L_states:
                for e in E_states:
                    prob += P1[initial, l1] * P2[l1, l2] * P3[l2, e]
        J[0, j] = J[j, 0] = prob

    # i == 0 and j == no_states-1
    P2 = ctmc.probability_matrix(break_points[no_states-2]-break_points[0])
    prob = 0.0
    for l1 in L_states:
        for l2 in L_states:
            prob += P1[initial, l1] * P2[l1, l2]
    J[0, no_states-1] = J[no_states-1, 0] = prob

    # i > 0
    for i in xrange(1, no_states-1):
        P1 = ctmc.probability_matrix(break_points[i-1]) # [0;i[
        P2 = ctmc.probability_matrix(break_points[i]-break_points[i-1]) # [i]
        # 0 < j < no_states - 1
        for j in xrange(i+1, no_states-1):
            P3 = ctmc.probability_matrix(break_points[j-1]-break_points[i]) # ]i;j[
            P4 = ctmc.probability_matrix(break_points[j]-break_points[j-1]) # [j]
            prob = 0.0
            for b in B_states:
                for l1 in L_states:
                    for l2 in L_states:
                        for e in E_states:
                            prob += P1[initial, b] * P2[b, l1] * P3[l1, l2] * P4[l2, e]
            J[i, j] = J[j, i] = prob

        # j == no_states - 1
        P3 = ctmc.probability_matrix(break_points[no_states-2]-break_points[i])
        prob = 0.0
        for b in B_states:
            for l1 in L_states:
                for l2 in L_states:
                    prob += P1[initial, b] * P2[b, l1] * P3[l1, l2]
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

    from IM2 import IM2, make_rates_table_migration
    from CTMC import CTMC

    state_space = IM2()
    rates = make_rates_table_migration(1, 0.5, 4e-4, 0.2, 0.2)
    coal_system = CTMC(state_space, rates)

    pi, T = compute_transition_probabilities(coal_system, [1, 2, 3])
    print pi
    print
    print '=>', pi.sum()
    print
    print T
    print
    print '=>', T.sum()


if __name__ == '__main__':
    main()
