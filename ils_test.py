
from IMCoalHMM.statespace_generator import CoalSystem
from IMCoalHMM.transitions import projection_matrix, compute_between, compute_upto
from IMCoalHMM.CTMC import make_ctmc
from numpy import zeros, matrix


# For debugging...
def pretty_state(state):
    def pretty_lineage(lineage):
        return '{{{}}}'.format(','.join(str(x) for x in lineage))
    return '{{{}}}'.format(', '.join(pretty_lineage(lineage) for lineage in state))


def pretty_pair(step):
    return '{} | {}'.format(pretty_state(step[0]), pretty_state(step[1]))


def pretty_path(path):
    return ' -> '.join(pretty_pair(step) for step in path)


def pretty_time_path(path):
    steps = []
    for first, time, second in path:
        steps.append("{} [{}] {}".format(pretty_pair(first), time, pretty_pair(second)))
    return " ::: ".join(steps)


# extract left and right lineages for a state
def extract_lineages(state):
    left = frozenset(nuc[0] for pop, nuc in state if nuc[0])
    right = frozenset(nuc[1] for pop, nuc in state if nuc[1])
    return left, right


class ILSSystem(CoalSystem):
    def __init__(self):
        super(ILSSystem, self).__init__()
        self.state_type = dict()
        self.transitions = [[('R', self.recombination)], [('C', self.coalesce)]]

    def sort_states(self):
        for state, index in self.states.items():
            left, right = extract_lineages(state)
            self.state_type.setdefault((left, right), []).append(index)


class Isolation3(ILSSystem):
    def __init__(self):
        super(Isolation3, self).__init__()
        self.init = frozenset([(sample, (frozenset([sample]), frozenset([sample]))) for sample in [1, 2, 3]])
        self.compute_state_space()
        self.init_index = self.states[self.init]
        self.sort_states()


class Isolation2(ILSSystem):
    def __init__(self):
        super(Isolation2, self).__init__()
        self.init = frozenset([(population, (frozenset([sample]), frozenset([sample])))
                               for population, sample in zip([12, 12, 3], [1, 2, 3])])
        self.compute_state_space()
        self.sort_states()


class Isolation1(ILSSystem):
    def __init__(self):
        super(Isolation1, self).__init__()
        self.init = frozenset([(population, (frozenset([sample]), frozenset([sample])))
                               for population, sample in zip([123, 123, 123], [1, 2, 3])])
        self.compute_state_space()
        self.sort_states()


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
    """Builds the rates table from the CTMC for the two-samples system for an isolation period."""
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

    def get_states(self, i, state_type):
        """Extract the states of the given state_type for interval i.
        """
        state_type_table = self.get_state_space(i).state_type
        if state_type in state_type_table:
            return state_type_table[state_type]
        else:
            return None

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

    def valid_system_path(self, timed_path):
        for x, i, y in timed_path:
            if self.get_states(i, x) is None or self.get_states(i+1, y) is None:
                return False
        return True


STATE_B = frozenset([frozenset([1]), frozenset([2]), frozenset([3])])
STATE_12 = frozenset([frozenset([1, 2]), frozenset([3])])
STATE_13 = frozenset([frozenset([1, 3]), frozenset([2])])
STATE_23 = frozenset([frozenset([2, 3]), frozenset([1])])
STATE_E = frozenset([frozenset([1, 2, 3])])

ALL_STATES = [STATE_B, STATE_12, STATE_13, STATE_23, STATE_E]
MARGINAL_PATHS = [
    [STATE_B, STATE_E],
    [STATE_B, STATE_12, STATE_E],
    [STATE_B, STATE_13, STATE_E],
    [STATE_B, STATE_23, STATE_E],
]


def path_merger(left, right):
    if len(left) == 1:
        yield [(left[0], r) for r in right]
    elif len(right) == 1:
        yield [(l, right[0]) for l in left]
    else:
        for tail in path_merger(left[1:], right):
            yield [(left[0], right[0])] + tail
        for tail in path_merger(left, right[1:]):
            yield [(left[0], right[0])] + tail
        for tail in path_merger(left[1:], right[1:]):
            yield [(left[0], right[0])] + tail

JOINT_PATHS = []
for left in MARGINAL_PATHS:
    for right in MARGINAL_PATHS:
        JOINT_PATHS.extend(path_merger(left, right))


def time_path(path, x, y):
    assert len(path) > 1

    first, second = path[0], path[1]
    if len(path) == 2:
        for break_point in range(x, y):
            yield [(first, break_point, second)]
    else:
        for break_point in range(x, y):
            for continuation in time_path(path[1:], break_point+1, y):
                yield [(first, break_point, second)] + continuation







#for path in JOINT_PATHS:
#    for step in path:
#        print pretty_state(step[0]), '|', pretty_state(step[1])
#    print '###'


## sanity check: marginalize the joint paths
left_paths, right_paths = set(), set()
for path in JOINT_PATHS:
    left_full = [left for left, right in path]
    right_full = [right for left, right in path]

    left = [left_full[0]]
    for i in range(1, len(left_full)):
        if left_full[i-1] != left_full[i]:
            left.append(left_full[i])
    right = [right_full[0]]
    for i in range(1, len(right_full)):
        if right_full[i-1] != right_full[i]:
            right.append(right_full[i])

    left_paths.add(tuple(left))
    right_paths.add(tuple(right))



#for s in epoch_3_ctmc.state_space.states:
#    left, right = extract_lineages(s)
#    print pretty_state(left), "|", pretty_state(right)
#    print left == state_B



epoch_1_ctmc = make_ctmc(Isolation3(), make_rates_table_3(1000.0, 1000.0, 1000.0, 0.4))
epoch_2_ctmc = make_ctmc(Isolation2(), make_rates_table_2(1000.0, 1000.0, 0.4))
epoch_3_ctmc = make_ctmc(Isolation1(), make_rates_table_1(1000.0, 0.4))

system = ILSCTMCSystem(epoch_1_ctmc, epoch_2_ctmc, epoch_3_ctmc, [1, 2, 3], [4, 5, 6])



for path in time_path(JOINT_PATHS[1], 0, 6):
    print pretty_time_path(path)

print '-' * 60

for path in time_path(JOINT_PATHS[1], 0, 6):
    if system.valid_system_path(path):
        print pretty_time_path(path)


