from math import exp
from IMCoalHMM.statespace_generator import CoalSystem
from IMCoalHMM.transitions import projection_matrix, compute_between, compute_upto
from IMCoalHMM.CTMC import make_ctmc
from IMCoalHMM.model import Model
from IMCoalHMM.break_points import exp_break_points, trunc_exp_break_points

from numpy import zeros, matrix, ix_
from numpy.testing import assert_almost_equal
from pyZipHMM import Matrix


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


def pretty_marginal_time_path(path):
    steps = []
    for first, time, second in path:
        steps.append("{} [{}] {}".format(pretty_state(first), time, pretty_state(second)))
    return " ::: ".join(steps)


# Handling states and paths for ILS model.
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
for _left in MARGINAL_PATHS:
    for _right in MARGINAL_PATHS:
        JOINT_PATHS.extend(path_merger(_left, _right))


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
        self.init_index = self.states[self.init]
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


def compute_up_to0(epoch_1, epoch_2, tau1):
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
    pseudo_through[:, epoch_3.state_space.state_type[(STATE_E, STATE_E)][0]] = 1.0
    through_123.append(pseudo_through)

    return through_12 + through_123


class ILSCTMCSystem(object):
    """Wrapper around CTMC transition matrices for the ILS model."""

    def __init__(self, model, epoch_1_ctmc, epoch_2_ctmc, epoch_3_ctmc, break_points_12, break_points_123):
        self.model = model
        self.epoch_1 = epoch_1_ctmc
        self.epoch_2 = epoch_2_ctmc
        self.epoch_3 = epoch_3_ctmc
        self.break_points_12 = break_points_12
        self.break_points_123 = break_points_123

        self.through_ = compute_through(self.epoch_2, self.epoch_3, self.break_points_12, self.break_points_123)
        self.up_to_ = compute_upto(compute_up_to0(self.epoch_1, self.epoch_2, self.break_points_12[0]), self.through_)
        self.between_ = compute_between(self.through_)

    def through(self, i):
        return self.through_[i]

    def up_to(self, i):
        return self.up_to_[i]

    def between(self, i, j):
        return self.between_[(i, j)]

    def get_path_probability(self, path):
        x, i, y = path[0]

        up_to = self.up_to(i)[self.model.initial, self.model.get_states(i, x)]
        through = self.through(i)[ix_(self.model.get_states(i, x), self.model.get_states(i+1, y))]
        probability = up_to * through

        for x, j, y in path[1:]:
            between = self.between(i, j)[ix_(self.model.get_states(i+1, x), self.model.get_states(j, x))]
            through = self.through(j)[ix_(self.model.get_states(j, x), self.model.get_states(j+1, y))]
            probability = probability * between * through
            i = j

        return probability.sum()

    def make_joint_matrix(self):
        no_states = len(self.model.tree_map)
        joint = matrix(zeros((no_states, no_states)))
        for path in self.model.valid_paths:
            i, j = self.model.get_path_indices(path)
            joint[i, j] = self.get_path_probability(path)
        return joint

    def compute_transition_probabilities(self):
        no_states = len(self.model.tree_map)
        joint = self.make_joint_matrix()
        assert_almost_equal(joint.sum(), 1.0)

        initial_prob_vector = Matrix(no_states, 1)
        transition_matrix = Matrix(no_states, no_states)
        for i in xrange(no_states):
            initial_prob_vector[i, 0] = joint[i, ].sum()
            for j in xrange(no_states):
                transition_matrix[i, j] = joint[i, j] / initial_prob_vector[i, 0]

        return initial_prob_vector, transition_matrix


## Class that can construct HMMs ######################################
class ILSModel(Model):
    """Class wrapping the code that generates an isolation model HMM."""

    def __init__(self, no_12_intervals, no_123_intervals):
        """Construct the model.

        This builds the state spaces for the CTMCs but not the matrices for the
        HMM since those will depend on the rate parameters."""
        super(ILSModel, self).__init__()
        self.epoch_1 = Isolation3()
        self.epoch_2 = Isolation2()
        self.epoch_3 = Isolation1()

        self.no_12_intervals = no_12_intervals
        self.no_123_intervals = no_123_intervals
        self.no_intervals = no_12_intervals + no_123_intervals

        self.init_index = self.epoch_1.init_index

        self.valid_paths_ = None
        self.tree_map = None
        self.reverse_tree_map = None

        self.make_valid_paths()
        self.index_marginal_trees()

    def get_state_space(self, i):
        if i < self.no_12_intervals:
            return self.epoch_2
        else:
            return self.epoch_3

    @property
    def initial(self):
        """The initial state index in the bottom-most matrix.

        :returns: the state space index of the initial state.
        :rtype: int
        """
        return self.init_index

    def get_states(self, i, state_type):
        """Extract the states of the given state_type for interval i.
        """
        state_type_table = self.get_state_space(i).state_type
        if state_type in state_type_table:
            return state_type_table[state_type]
        else:
            return None

    def valid_system_path(self, timed_path):
        for x, i, y in timed_path:
            if self.get_states(i, x) is None:
                # This should always be an obtainable state since we start in it.
                return False
            if self.get_states(i+1, y) is None or self.get_states(i, y) is None:
                # Although we only need to index these states at point i+1 we need to know
                # that they are also at the end of interval i since otherwise they would
                # have zero probability anyway
                return False
        return True

    def make_valid_paths(self):
        no_intervals = self.no_intervals
        self.valid_paths_ = []
        for path in JOINT_PATHS:
            for timed_path in time_path(path, 0, no_intervals):
                if self.valid_system_path(timed_path):
                    self.valid_paths_.append(timed_path)

    @property
    def valid_paths(self):
        return self.valid_paths_

    @staticmethod
    def get_marginal_time_path(timed_path, margin):
        marginal_path = []
        for x, i, y in timed_path:
            xx, yy = x[margin], y[margin]
            if xx != yy:
                marginal_path.append((xx, i, yy))
        return tuple(marginal_path)

    def index_marginal_trees(self):
        self.tree_map = {}
        index = 0
        for path in self.valid_paths:
            tree = self.get_marginal_time_path(path, 0)
            if tree not in self.tree_map:
                self.tree_map[tree] = index
                index += 1

        self.reverse_tree_map = [None] * len(self.tree_map)
        for tree, i in self.tree_map.items():
            self.reverse_tree_map[i] = tree

    def get_path_indices(self, path):
        left_tree = self.get_marginal_time_path(path, 0)
        right_tree = self.get_marginal_time_path(path, 1)
        return self.tree_map[left_tree], self.tree_map[right_tree]

    def build_ctmc_system(self, tau1, tau2, coal1, coal2, coal3, coal12, coal123, recombination_rate):
        """Construct CTMC system."""
        epoch_1_ctmc = make_ctmc(Isolation3(), make_rates_table_3(coal1, coal2, coal3, recombination_rate))
        epoch_2_ctmc = make_ctmc(Isolation2(), make_rates_table_2(coal12, coal3, recombination_rate))
        epoch_3_ctmc = make_ctmc(Isolation1(), make_rates_table_1(coal123, recombination_rate))

        self.break_points_12 = trunc_exp_break_points(self.no_12_intervals, coal12, tau1 + tau2, tau1)
        self.break_points_123 = exp_break_points(self.no_123_intervals, coal123, tau1 + tau2)

        return ILSCTMCSystem(self, epoch_1_ctmc, epoch_2_ctmc, epoch_3_ctmc, self.break_points_12, self.break_points_123)

    def emission_points(self, *parameters):
        """Expected coalescence times between between tau1 and tau2"""

        try:
            (tau1, tau2, coal1, coal2, coal3, coal12, coal123, _), outgroup = parameters, None
        except ValueError:
            tau1, tau2, coal1, coal2, coal3, coal12, coal123, _, outgroup = parameters

        breaks_12 = list(self.break_points_12) + [float(tau1 + tau2)] # turn back into regular python...
        epoch_1_time_spans = [e-s for s, e in zip(breaks_12[0:-1], breaks_12[1:])]
        epoch_1_emission_points = [(1/coal12)-dt/(-1+exp(dt*coal12)) for dt in epoch_1_time_spans]

        epoch_2_time_spans = [e-s for s, e in zip(self.break_points_123[0:-1], self.break_points_123[1:])]
        epoch_2_emission_points = [(1/coal123)-dt/(-1+exp(dt*coal123)) for dt in epoch_2_time_spans]
        epoch_2_emission_points.append(self.break_points_123[-1] + 1/coal123)

        return epoch_1_emission_points + epoch_2_emission_points, outgroup

    def get_tree(self, path, column_representation, coalescence_time_points, outgroup, branch_shortening):

        def get_alignment_col(n, symbols = "ACGT", length=3): # FIXME: Hard coded sequence of nucleotides
            """Convert a number n that represent an alignment column to a list of indexes in
            the symbols string. E.g. A, C, G, T -> 0, 1, 2, 3

            E.g. the triplet ATG must be encoded like this:
            nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
            nuc_map['A'] + 4*nuc_map['T'], 16*nuc_map['G']
            """
            power = length - 1
            base = len(symbols)
            if power == 0:
                return [n]
            else:
                return get_alignment_col(n%base**power, symbols=symbols, length=power) + [n//base**power]

        # get bases of alignment column
        if outgroup:
            b1, b2, b3, b4 = get_alignment_col(column_representation, length=4)
        else:
            b1, b2, b3 = get_alignment_col(column_representation, length=3)

        s1, s2, s3 = branch_shortening

        # get topology and branch lengths
        assert 1 <= len(path) <= 2, "tree with more than two coalescence events"

        if len(path) == 1:
            # two coalescence events in the same interval is represented as a star-shape topology
            star = coalescence_time_points[path[0][1]]
            tree =  {'len': [star-s1, star-s2, star-s3], 'chld' : [{'leaf': b1}, {'leaf': b2}, {'leaf': b3}]}
            if outgroup:
                tree = {'len': [star + outgroup, outgroup], 'chld': [tree, {'leaf': b4}]}
        else:
            tree = list(sorted(path[0][2], key=len)[0])[0]
            short_external = coalescence_time_points[path[0][1]]
            long_external = coalescence_time_points[path[1][1]]
            internal = long_external - short_external
            if tree == 2:
                b1, b2, b3 = b1, b3, b2
                s1, s2, s3 = s1, s3, s2
            elif tree == 3:
                b1, b2, b3 = b2, b3, b1
                s1, s2, s3 = s2, s3, s1
            tree = {'len': [internal, long_external-s3],
                    'chld': [{'len': [short_external-s1, short_external-s2],
                              'chld' : [{'leaf': b1},
                                        {'leaf': b2}]},
                             {'leaf': b3}]}
            if outgroup:
                tree = {'len': [long_external + outgroup, outgroup], 'chld': [tree, {'leaf': b4}]}

        return tree

    def emission_matrix(self, *parameters):
        """Compute emission matrix for zipHMM"""

        def subst_model(s):
            """Jukes-Cantor-69 substitution model
            s = substitutions = m*t (mutation rate times time)"""
            x = 1/4.0 + 3/4.0 * exp(-4*s)
            y = 1/4.0 - 1/4.0 * exp(-4*s)
            a = x
            b = y
            matrixq = [[a,b,b,b],[b,a,b,b],[b,b,a,b],[b,b,b,a]]
            return matrixq

        def prob_tree(node, i, trans_mat):
            """Computes the probability of a tree assuming base i at the root"""
            if 'chld' in node:
                p = None
                for child, brlen in zip(node['chld'], node['len']):
                    mat = trans_mat(brlen)
                    x = sum(mat[i][j] * prob_tree(child, j, trans_mat) for j in range(4))
                    p = p * x if p is not None else x
                return p
            else:
                return 1 if node['leaf'] == i else 0

        coalescence_times, outgroup = self.emission_points(*parameters)

        prior = [0.25]*4 # uniform prior assumed by Jukes-Cantor

        if outgroup:
            no_alignment_columns = 4**4 + 1
        else:
            no_alignment_columns = 4**3 + 1

        no_states = len(self.tree_map)
        emission_probabilities = Matrix(no_states, no_alignment_columns, )

        branch_shortening = [0, 0, 0] # FIXME: not sure how to pass in this information from the script...

        for state in xrange(no_states):
            path = self.reverse_tree_map[state]
            likelihoods = list()
            for align_column in range(no_alignment_columns):
                if align_column == no_alignment_columns - 1:
                    likelihoods.append(1)
                else:
                    tree = self.get_tree(path, align_column, coalescence_times, outgroup, branch_shortening)
                    print tree
                    likelihoods.append(sum(prior[i] * prob_tree(tree, i, subst_model) for i in range(4)))
            print sum(likelihoods)
            for align_column, emission_prob in enumerate(x/sum(likelihoods) for x in likelihoods):
                emission_probabilities[state, align_column] = emission_prob

        return emission_probabilities

    # We override this one from the Model class because we cannot directly reuse the 2-sample code.
    def build_hidden_markov_model(self, parameters):
        """Build the hidden Markov model matrices from the model-specific parameters."""
        if len(parameters) == 9:
            # with outgroup
            ctmc_system = self.build_ctmc_system(*parameters[:-1]) # skip outgroup parameter for ctmc system
        else:
            assert len(parameters) == 8 # no outgroup
            ctmc_system = self.build_ctmc_system(*parameters)

        initial_probabilities, transition_probabilities = ctmc_system.compute_transition_probabilities()
        emission_probabilities = self.emission_matrix(*parameters)
        return initial_probabilities, transition_probabilities, emission_probabilities

