'''
Created on Apr 20, 2015

@author: svendvn
'''
from IMCoalHMM.transitions import CTMCSystem, compute_upto, compute_between, projection_matrix
from IMCoalHMM.model import Model
from IMCoalHMM.ILS import Isolation2, Isolation1, make_rates_table2,make_rates_table_1
from IMCoalHMM.statespace_generator import CoalSystem
from IMCoalHMM.state_spaces import make_rates_table_single
from IMCoalHMM.CTMC import make_ctmc
from IMCoalHMM.break_points import uniform_break_points, exp_break_points
from IMCoalHMM.admixture import admixture_state_space_map

from numpy import zeros, matrix, identity, ix_
from numpy.testing import assert_almost_equal
from pyZipHMM import Matrix





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


JOINT_PATHS = []
for _left in MARGINAL_PATHS:
    for _right in MARGINAL_PATHS:
        JOINT_PATHS.extend(path_merger(_left, _right))





# extract left and right lineages for a state
def extract_lineages(state):
    left = frozenset(nuc[0] for pop, nuc in state if nuc[0])
    right = frozenset(nuc[1] for pop, nuc in state if nuc[1])
    return left, right





class Admixture3HMiddle(CoalSystem):

    def __init__(self):
        """Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are)."""

        super(Admixture3HMiddle, self).__init__()

        self.transitions = [[('R', self.recombination)], [('C', self.coalesce)]]

        # We need various combinations of initial states to make sure we build the full reachable state space.
        left_1 = [frozenset([(12, (frozenset([1]), frozenset([])))]), frozenset([(3, (frozenset([1]), frozenset([])))])]
        right_1 = [frozenset([(12, (frozenset([]), frozenset([1])))]), frozenset([(3, (frozenset([]), frozenset([1])))])]
        left_2 = [frozenset([(12, (frozenset([2]), frozenset([])))]), frozenset([(3, (frozenset([2]), frozenset([])))])]
        right_2 = [frozenset([(12, (frozenset([]), frozenset([2])))]), frozenset([(3, (frozenset([]), frozenset([2])))])]
        left_3 = [frozenset([(12, (frozenset([3]), frozenset([])))]), frozenset([(3, (frozenset([3]), frozenset([])))])]
        right_3 = [frozenset([(12, (frozenset([]), frozenset([3])))]), frozenset([(3, (frozenset([]), frozenset([3])))])]
        self.init = [l1 | r1 | l2 | r2 | l3 | r3 for l1 in left_1 for r1 in right_1 for l2 in left_2 for r2 in right_2 for l3 in left_3 for r3 in right_3]

        self.compute_state_space()
    

class Admixture3HCTMCSystem(CTMCSystem):
    
    def __init__(self, before_admix_ctmc, after_admix_ctmc, ancestral_ctmc, break_points1, break_points2, break_points3,p,q):
        
        super(Admixture3HCTMCSystem, self).__init__(no_hmm_states=len(break_points1)+len(break_points2)+len(break_points3),
                                                  initial_ctmc_state=0)  #what to do with initial state?
        
        #saving for later
        self.before_admix_ctmc = before_admix_ctmc
        self.after_admix_ctmc = after_admix_ctmc
        self.ancestral_ctmc = ancestral_ctmc
        self.break_points_12a = break_points1
        self.break_points_12b = break_points2
        self.break_points_123 = break_points3
        self.no_middle_states=len(break_points2)
        self.no_before_states=len(break_points1)
        self.no_ancestral_states=len(break_points3)

        self.through_ = [None] * (self.no_before_states+self.no_middle_states + self.no_ancestral_states - 1) #in the end a last term is added so that it becomes one longer

        for i in xrange(self.no_before_states - 1):
            self.through_[i] = before_admix_ctmc.probability_matrix(break_points1[i+1] - break_points1[i])

        #the transition with admixture
        xx = before_admix_ctmc.probability_matrix(break_points2[0] - break_points1[-1])
        projection = admixture_state_space_map(before_admix_ctmc.state_space, after_admix_ctmc.state_space, p, q)
        self.through_[self.no_before_states - 1] = xx * projection

        #transitions after admixture
        for i in xrange(self.no_before_states, self.no_middle_states + self.no_before_states - 1):
            ii = i - self.no_before_states
            self.through_[i] = after_admix_ctmc.probability_matrix(break_points2[ii+1] - break_points2[ii])

        #transition to ancestral
        def state_map_21(state):
            return frozenset([(123, nucleotides) for (_, nucleotides) in state])
        
        xx = after_admix_ctmc.probability_matrix(break_points3[0] - break_points2[-1])
        projection = projection_matrix(after_admix_ctmc.state_space, ancestral_ctmc.state_space, state_map_21)
        self.through_[self.no_before_states+self.no_middle_states - 1] = xx * projection
        
        #transitions in ancestral
        for i in xrange(self.no_before_states+self.no_middle_states, self.no_middle_states + self.no_before_states+self.no_ancestral_states - 1):
            ii=i-(self.no_before_states+self.no_middle_states)
            self.through_[i] = ancestral_ctmc.probability_matrix(break_points3[ii + 1] - break_points3[ii])
        
        #last transition
        pseudo_through = matrix(zeros((len(ancestral_ctmc.state_space.states), len(ancestral_ctmc.state_space.states))))
        pseudo_through[:, ancestral_ctmc.state_space.end_states[0]] = 1.0
        self.through_.append(pseudo_through)
             
        #constructing other lists of matrices
        upto0 = matrix(identity(len(before_admix_ctmc.state_space.states)))
        self.upto_ = compute_upto(upto0, self.through_)
        self.between_ = compute_between(self.through_)
        
    def get_state_space(self, i):
        """Return the state space for interval i."""
        if i < self.no_before_states:
            return self.before_admix_ctmc.state_space
        elif i<self.no_before_states+self.no_middle_states:
            return self.after_admix_ctmc.state_space
        else:
            return self.ancestral_ctmc.state_space
        
        
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
        
        
        

def make_rates_table_admixture(coal_rate_1, coal_rate_2, recomb_rate):
    """Builds the rates table from the CTMC for the two-samples system
    for an isolation period."""
    table = dict()
    table[('C', 1, 1)] = coal_rate_1
    table[('C', 2, 2)] = coal_rate_2
    table[('R', 1, 1)] = recomb_rate
    table[('R', 2, 2)] = recomb_rate
    return table


class Admixture3HModel(Model):
    
    
    def __init__(self, initial_configuration, no_isolation_intervals, no_middle_intervals, no_ancestral_intervals):
        
        super(Admixture3HModel, self).__init__()
        
        self.initial_state = initial_configuration
        self.isolation_state_space = Isolation2()
        self.middle_state_space = Admixture3HMiddle()
        self.ancestral_state_space = Isolation1()
        
        
        
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
        
        
    def build_ctmc_system(self, *parameters):
        """Construct CTMCs and compute HMM matrices given the split times
        and the rates."""
        
        tau_1, tau_2, coal_11, coal_12, coal_21, coal_22, coal_last, recomb, p, q = parameters
        isolation_rates = make_rates_table2(coal_11, coal_12, recomb)
        middle_rates = make_rates_table2(coal_21, coal_22, recomb)
        ancestral_rates = make_rates_table_1(coal_last, recomb)
        
        isolation_ctmc = make_ctmc(self.isolation_state_space, isolation_rates)
        middle_ctmc = make_ctmc(self.middle_state_space, middle_rates)
        ancestral_ctmc = make_ctmc(self.ancestral_state_space, ancestral_rates)
        isolation_breakpoints=uniform_break_points(0,tau_1)
        middle_breakpoints=uniform_break_points(tau_1,tau_2)
        ancestral_breakpoints = exp_break_points(self.no_ancestral_states, coal_last, tau_2)
        
        return Admixture3HCTMCSystem(self, isolation_ctmc, middle_ctmc, ancestral_ctmc,isolation_breakpoints, middle_breakpoints, ancestral_breakpoints)
    
    
        # We override this one from the Model class because we cannot directly reuse the 2-sample code.
    def build_hidden_markov_model(self, parameters):
        """Build the hidden Markov model matrices from the model-specific parameters."""
        ctmc_system = self.build_ctmc_system(*parameters)
        initial_probabilities, transition_probabilities = ctmc_system.compute_transition_probabilities()
        emission_probabilities = self.emission_matrix()
        return initial_probabilities, transition_probabilities, emission_probabilities
        