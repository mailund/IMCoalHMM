
from IMCoalHMM.statespace_generator import CoalSystem
from IMCoalHMM.transitions import projection_matrix, compute_between, compute_upto
from IMCoalHMM.CTMC import make_ctmc
from numpy import zeros, matrix


class ILSSystem(CoalSystem):
    def __init__(self):
        super(ILSSystem, self).__init__()
        self.transitions = [[('R', self.recombination)], [('C', self.coalesce)]]


class Isolation3(ILSSystem):
    def __init__(self):
        super(Isolation3, self).__init__()
        self.init = frozenset([(sample, (frozenset([sample]), frozenset([sample]))) for sample in [1, 2, 3]])
        self.compute_state_space()
        self.init_index = self.states[self.init]


class Isolation2(ILSSystem):
    def __init__(self):
        super(Isolation2, self).__init__()
        self.init = frozenset([(population, (frozenset([sample]), frozenset([sample])))
                               for population, sample in zip([12, 12, 3], [1, 2, 3])])
        self.compute_state_space()


class Isolation1(ILSSystem):
    def __init__(self):
        super(Isolation1, self).__init__()
        self.init = frozenset([(population, (frozenset([sample]), frozenset([sample])))
                               for population, sample in zip([123, 123, 123], [1, 2, 3])])
        self.compute_state_space()


def make_rates_table_3(coal_rate_1, coal_rate_2, coal_rate_3, recombination_rate):
    """Builds the rates table from the CTMC for the two-samples system
    for an isolation period."""
    table = dict()
    table[('C', 1, 1)] = coal_rate_1
    table[('C', 2, 2)] = coal_rate_2
    table[('C', 3, 3)] = coal_rate_3
    table[('R', 1, 1)] = recombination_rate
    table[('R', 2, 2)] = recombination_rate
    table[('R', 3, 3)] = recombination_rate
    return table


def make_rates_table_2(coal_rate_12, coal_rate_3, recombination_rate):
    """Builds the rates table from the CTMC for the two-samples system
    for an isolation period."""
    table = dict()
    table[('C', 12, 12)] = coal_rate_12
    table[('C', 3, 3)] = coal_rate_3
    table[('R', 12, 12)] = recombination_rate
    table[('R', 3, 3)] = recombination_rate
    return table


def make_rates_table_1(coal_rate_123, recombination_rate):
    """Builds the rates table from the CTMC for the two-samples system
    for an isolation period."""
    table = dict()
    table[('C', 123, 123)] = coal_rate_123
    table[('R', 123, 123)] = recombination_rate
    return table


def compute_upto0(epoch_1, epoch_2, tau1):
    """Computes the probability matrices for moving to time zero."""

    def state_map_32(state):
        def lineage_map(lineage):
            population, nucleotides = lineage
            if population == 3:
                return 3, nucleotides
            else:
                return 12, nucleotides
        return frozenset(lineage_map(lineage) for lineage in state)

    projection_32 = projection_matrix(epoch_1.state_space, epoch_2.state_space, state_map_32)
    return epoch_1.probability_matrix(tau1) * projection_32


def compute_through(epoch_2, epoch_3, break_points_12, break_points_123):
    """Computes the matrices for moving through an interval.

    :rtype: list[matrix]
    """
    through_12 = [None] * len(break_points_12)
    through_123 = [None] * (len(break_points_123) - 1)

    def state_map_21(state):
        return frozenset([(123, nucleotides) for (_, nucleotides) in state])

    projection_21 = projection_matrix(epoch_2.state_space, epoch_3.state_space, state_map_21)

    # Through epoch 2
    for i in range(len(break_points_12) - 1):
        through_12[i] = epoch_2.probability_matrix(break_points_12[i + 1] - break_points_12[i])
    through_12[len(break_points_12)-1] = \
        epoch_2.probability_matrix(break_points_123[0] - break_points_12[-1]) * projection_21

    # Through epoch 3
    for i in range(len(break_points_123) - 1):
        through_123[i] = epoch_3.probability_matrix(break_points_123[i + 1] - break_points_123[i])

    # As a hack we set up a pseudo through matrix for the last interval that
    # just puts all probability on ending in one of the end states. This
    # simplifies the HMM transition probability code as it avoids a special case
    # for the last interval.
    # noinspection PyCallingNonCallable
    pseudo_through = matrix(zeros((len(epoch_3.state_space.states), len(epoch_3.state_space.states))))
    pseudo_through[:, epoch_3.state_space.end_states[0]] = 1.0
    through_123.append(pseudo_through)

    return through_12 + through_123


class ILSCTMCSystem(object):
    """Wrapper around CTMC transition matrices for the ILS model."""

    def __init__(self, epoch_1_ctmc, epoch_2_ctmc, epoch_3_ctmc, break_points_12, break_points_123):
        self.epoch_1 = epoch_1_ctmc
        self.epoch_2 = epoch_2_ctmc
        self.epoch_3 = epoch_3_ctmc
        self.break_points_12 = break_points_12
        self.break_points_123 = break_points_123

        self.through_ = compute_through(self.epoch_2, self.epoch_3, self.break_points_12, self.break_points_123)
        self.upto_ = compute_upto(compute_upto0(self.epoch_1, self.epoch_2, self.break_points_12[0]), self.through_)
        self.between_ = compute_between(self.through_)

    def get_state_space(self, i):
        if i < len(self.break_points_12):
            return self.epoch_2.state_space
        else:
            return self.epoch_3.state_space

    @property
    def initial(self):
        """The initial state index in the bottom-most matrix.

        :returns: the state space index of the initial state.
        :rtype: int
        """
        return self.epoch_1.init_index

    def begin_states(self, i):
        """Begin states for interval i.

        :param i: interval index
        :type i: int

        :returns: List of the begin states for the state space in interval i.
        :rtype: list
        """
        return self.get_state_space(i).begin_states

    # FIXME: replace by the states for the ILS model
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
        through, interval i. [...][i

        :param i: interval index
        :type i: int

        :returns: Probability transition matrix for moving from time 0
         up to, but not through, interval i. [...][i
        :rtype: matrix
        """
        return self.upto_[i]

    def between(self, i, j):
        """Returns a probability matrix for going from the
        end of interval i up to (but not through) interval j. i][...][j

        :param i: interval index
        :type i: int
        :param j: interval index. i < j
        :type i: int

        :returns: Probability transition matrix for moving from the end
         of interval i to the beginning of interval j: i][...][j
        :rtype: matrix
        """
        return self.between_[(i, j)]

epoch_1_ctmc = make_ctmc(Isolation3(), make_rates_table_3(1000.0, 1000.0, 1000.0, 0.4))
epoch_2_ctmc = make_ctmc(Isolation2(), make_rates_table_2(1000.0, 1000.0, 0.4))
epoch_3_ctmc = make_ctmc(Isolation1(), make_rates_table_1(1000.0, 0.4))

system = ILSCTMCSystem(epoch_1_ctmc, epoch_2_ctmc, epoch_3_ctmc, [1, 2, 3], [4, 5, 6])
