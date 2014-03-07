"""
General code for calculations of HMM transition probabilities.

"""

from abc import ABCMeta, abstractmethod
from numpy import zeros, identity, matrix, ix_
from numpy.testing import assert_almost_equal
from pyZipHMM import Matrix


def projection_matrix(from_state_space, to_state_space, state_map):
    """
    Build a projection matrix for moving from one state space to another.

    :param from_state_space: The state space we move from.
    :type from_state_space: IMCoalHMM.CoalSystem
    :param to_state_space: The state space we move into
    :type to_state_space: IMCoalHMM.CoalSystem
    :param state_map: A function mapping states from one state space to another.

    :returns: a projection matrix
    :rtype: matrix
    """
    # noinspection PyCallingNonCallable
    projection = matrix(zeros((len(from_state_space.states),
                               len(to_state_space.states))))
    for from_state, from_index in from_state_space.states.items():
        to_state = state_map(from_state)
        to_index = to_state_space.states[to_state]
        projection[from_index, to_index] = 1.0
    return projection


def compute_upto(upto_0, through):
    """Computes the probability matrices for moving from time zero up to,
    but not through, interval i.

    :param upto_0: The probability matrix for moving up to the first break point.
        This is a basis case for the upto list that is returned.
    :type upto_0: numpy.matrix
    :param through: The probability matrices for moving through each interval.
    :type through: list[numpy.matrix]

    :returns: The list of transition probability matrices for moving up to
        each interval.
    :rtype: list[matrix]
    """
    no_states = len(through)
    upto = [None] * no_states
    upto[0] = upto_0
    for i in xrange(1, no_states):
        upto[i] = upto[i - 1] * through[i - 1]
    return upto


def compute_between(through):
    """Computes the matrices for moving from the end of interval i
    to the beginning of interval j.

    :param through: The probability matrices for moving through each interval.
    :type through: list[matrix]

    :returns: A table of transition probability matrices for moving between any two
        intervals i < j.
    :rtype: dict[(int,int), matrix]
    """
    no_states = len(through)
    between = dict()
    # Transitions going from the endpoint of interval i to the entry point
    # of interval j
    for i in xrange(no_states - 1):
        # noinspection PyCallingNonCallable
        between[(i, i + 1)] = matrix(identity(through[i].shape[1]))
        for j in xrange(i + 2, no_states):
            between[(i, j)] = between[(i, j - 1)] * through[j - 1]
    return between


class CTMCSystem(object):
    """Class that wraps the CTMC calculations and just presents the
    HMM transition probability calculations with the matrices and
    indices it needs, hiding away the underlying demographic model.
    """
    __metaclass__ = ABCMeta

    def __init__(self, no_hmm_states, initial_ctmc_state):
        """Create system.

        :param no_hmm_states: The number of states the HMM should have.
        :type no_hmm_states: int
        :param initial_ctmc_state: The index of the initial state used in the first CTMC.
        :type initial_ctmc_state: int
        """
        self.no_hmm_states = no_hmm_states
        self.initial_ctmc_state = initial_ctmc_state

        # These should be filled in by the sub-class's __init__ method
        self.through_ = []
        self.upto_ = []
        self.between_ = {}

    @abstractmethod
    def get_state_space(self, i):
        """Return the state space used in interval i.

        :param i: Interval index.
        :type i: int
        :returns: The state space for interval i.
        :rtype: IMCoalHMM.CoalSystem
        """
        return None

    @property
    def no_states(self):
        """The number of states the HMM should have.

        :returns: The number of HMM states.
        :rtype: int
        """
        return self.no_hmm_states

    @property
    def initial(self):
        """The initial state index in the bottom-most matrix.

        :returns: the state space index of the initial state.
        :rtype: int
        """
        return self.initial_ctmc_state

    def begin_states(self, i):
        """Begin states for interval i.

        :param i: interval index
        :type i: int

        :returns: List of the begin states for the state space in interval i.
        :rtype: list
        """
        return self.get_state_space(i).begin_states

    def left_states(self, i):
        """Left states for interval i.

        :param i: interval index
        :type i: int

        :returns: List of the left states for the state space in interval i.
        :rtype: list
        """
        return self.get_state_space(i).left_states

    def end_states(self, i):
        """End states for interval i.

        :param i: interval index
        :type i: int

        :returns: List of the end states for the state space in interval i.
        :rtype: list
        """
        return self.get_state_space(i).end_states

    def through(self, i):
        """Returns a probability matrix for going through interval i.

        :param i: interval index
        :type i: int

        :returns: Probability transition matrix for moving through interval i. [i]
        :rtype: matrix
        """
        return self.through_[i]

    def upto(self, i):
        """Returns a probability matrix for going up to, but not
        through, interval i.

        :param i: interval index
        :type i: int

        :returns: Probability transition matrix for moving from time 0
         up to, but not through, interval i. [0, i[
        :rtype: matrix
        """
        return self.upto_[i]

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
        return self.between_[(i, j)]


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
