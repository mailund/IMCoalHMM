"""
General code for calculations of HMM transition probabilities.

"""

import exceptions
from numpy import zeros, matrix, ix_
from numpy.testing import assert_almost_equal
from pyZipHMM import Matrix


class CTMCSystem(object):
    """Class that wraps the CTMC calculations and just presents the
    HMM transition probability calculations with the matrices and
    indices it needs, hiding away the underlying demographic model.
    """

    @property
    def no_states(self):
        """The number of states the HMM should have.

        :returns: The number of HMM states.
        :rtype: int
        """
        raise exceptions.NotImplementedError()

    @property
    def initial(self):
        """The initial state index in the bottom-most matrix.

        :returns: the state space index of the initial state.
        :rtype: int
        """
        raise exceptions.NotImplementedError()

    def begin_states(self, i):
        """Begin states for interval i.

        :param i: interval index
        :type i: int

        :returns: List of the begin states for the state space in interval i.
        :rtype: list
        """
        raise exceptions.NotImplementedError()

    def left_states(self, i):
        """Left states for interval i.

        :param i: interval index
        :type i: int

        :returns: List of the left states for the state space in interval i.
        :rtype: list
        """
        raise exceptions.NotImplementedError()

    def end_states(self, i):
        """End states for interval i.

        :param i: interval index
        :type i: int

        :returns: List of the end states for the state space in interval i.
        :rtype: list
        """
        raise exceptions.NotImplementedError()

    def through(self, i):
        """Returns a probability matrix for going through interval i.

        :param i: interval index
        :type i: int

        :returns: Probability transition matrix for moving through interval i. [i]
        :rtype: matrix
        """
        raise exceptions.NotImplementedError()

    def upto(self, i):
        """Returns a probability matrix for going up to, but not
        through, interval i.

        :param i: interval index
        :type i: int

        :returns: Probability transition matrix for moving from time 0
         up to, but not through, interval i. [0, i[
        :rtype: matrix
        """
        raise exceptions.NotImplementedError()

    def between(self, i, j):
        """Returns a probability matrix for going from the
        end of interval i up to (but not through) interval j.

        :param i: interval index
        :type i: int
        :param j: interval index. i < j
        :type i: int

        :returns: Probability transition matrix for moving from the end
         of interval i to the beginning of interval j: ]i, j[
        :rtype: matrix
        """
        raise exceptions.NotImplementedError()


def compute_transition_probabilities(ctmc):
    """Calculate the HMM transition probabilities from the CTMCs.

    :param ctmc: A CTMC system providing the transition probability matrices necessary
     for computing the HMM transition probability.
    :type ctmc: IMCoalHMM.CTMCSystem

    :returns: the stationary/beginning probability vector together with the transition
     probability matrix.
    """

    no_states = ctmc.no_states

    # Joint genealogy probabilities
    # noinspection PyCallingNonCallable
    joint = matrix(zeros((no_states, no_states)))

    # -- Filling in the diagonal (i == j) for the J matrix ----------------
    joint[0, 0] = ctmc.upto(1)[ctmc.initial, ctmc.end_states(0)].sum()
    for i in xrange(1, no_states - 1):
        joint[i, i] = (ctmc.upto(i)[ctmc.initial, ctmc.begin_states(i)]
                       * ctmc.through(i)[ix_(ctmc.begin_states(i), ctmc.end_states(i + 1))]).sum()

    joint[no_states - 1, no_states - 1] = ctmc.upto(no_states - 1)[ctmc.initial,
                                                                   ctmc.begin_states(no_states - 1)].sum()

    # -- handle i < j (and j < i by symmetry) ---------------------------
    for i in xrange(no_states - 1):
        up_through_i = ctmc.upto(i)[ctmc.initial, ctmc.begin_states(i)] \
            * ctmc.through(i)[ix_(ctmc.begin_states(i), ctmc.left_states(i + 1))]
        for j in xrange(i + 1, no_states):
            between_i_and_j = ctmc.between(i, j)[ix_(ctmc.left_states(i + 1), ctmc.left_states(j))]
            through_j = ctmc.through(j)[ix_(ctmc.left_states(j), ctmc.end_states(j + 1))]
            joint[i, j] = joint[j, i] = (up_through_i * between_i_and_j * through_j).sum()

    assert_almost_equal(joint.sum(), 1.0)

    initial_prob_vector = Matrix(no_states, 1)
    transition_matrix = Matrix(no_states, no_states)
    for i in xrange(no_states):
        initial_prob_vector[i, 0] = joint[i, ].sum()
        for j in xrange(no_states):
            transition_matrix[i, j] = joint[i, j] / initial_prob_vector[i, 0]

    return initial_prob_vector, transition_matrix
