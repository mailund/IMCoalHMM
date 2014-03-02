'''
General code for calculations of HMM transition probabilities.

'''

import exceptions
from numpy import zeros, matrix, ix_
from numpy.testing import assert_almost_equal
from pyZipHMM import Matrix


class CTMCSystem(object):
    '''Class that wraps the CTMC calculations and just presents the
    HMM transition probability calculations with the matrices and
    indices it needs, hiding away the underlying demographic model.
    '''

    @property
    def no_states(self):
        "The number of states the HMM should have."
        raise exceptions.NotImplementedError()

    @property
    def initial(self):
        'The initial state index in the bottom-most matrix'
        raise exceptions.NotImplementedError()

    def begin_states(self, i):
        'Begin states for interval i.'
        raise exceptions.NotImplementedError()

    def left_states(self, i):
        'Left states for interval i.'
        raise exceptions.NotImplementedError()

    def end_states(self, i):
        'End states for interval i.'
        raise exceptions.NotImplementedError()

    def through(self, i):
        'Returns a probability matrix for going through interval i'
        raise exceptions.NotImplementedError()

    def upto(self, i):
        '''Returns a probability matrix for going up to, but not
        through, interval i'''
        raise exceptions.NotImplementedError()

    def between(self, i, j):
        '''Returns a probability matrix for going from the
        end of interval i up to (but not through) interval j'''
        raise exceptions.NotImplementedError()


def compute_transition_probabilities(ctmc):
    '''Calculate the HMM transition probabilities from the CTMCs.

    Returns both the stationary probability pi together with the transition
    probability T, since pi is automatically calculated as part of
    the algorithm.
    '''

    no_states = ctmc.no_states

    # Joint genealogy probabilities
    J = matrix(zeros((no_states, no_states)))

    # -- Filling in the diagonal (i == j) for the J matrix ----------------
    J[0, 0] = ctmc.upto(1)[ctmc.initial, ctmc.end_states(0)].sum()
    for i in xrange(1, no_states-1):
        J[i, i] = (ctmc.upto(i)[ctmc.initial, ctmc.begin_states(i)]
                  * ctmc.through(i)[ix_(ctmc.begin_states(i), ctmc.end_states(i))]).sum()
    J[no_states-1, no_states-1] = ctmc.upto(no_states-1)[ctmc.initial,
                                                         ctmc.begin_states(no_states-1)].sum()

    # -- handle i < j (and j < i by symmetry) ---------------------------
    # FIXME: I don't know how to insert a projection if the state space
    # changes along one of the dimensions we sum over here, but something
    # might be needed unless it can be handled by careful choice of indices
    # when picking states from "ctmc"
    for i in xrange(no_states-1):
        up_through_i = ctmc.upto(i)[ctmc.initial, ctmc.begin_states(i)] * \
                       ctmc.through(i)[ix_(ctmc.begin_states(i), ctmc.left_states(i))]
        for j in xrange(i+1, no_states):
            between_i_and_j = ctmc.between(i, j)[ix_(ctmc.left_states(i), ctmc.left_states(j))]
            through_j = ctmc.through(j)[ix_(ctmc.left_states(j), ctmc.end_states(j))]
            J[i, j] = J[j, i] = (up_through_i * between_i_and_j * through_j).sum()

    assert_almost_equal(J.sum(), 1.0)

    pi = Matrix(no_states, 1)
    T = Matrix(no_states, no_states)
    for i in xrange(no_states):
        pi[i, 0] = J[i,].sum()
        for j in xrange(no_states):
            T[i, j] = J[i, j] / pi[i, 0]

    return pi, T
