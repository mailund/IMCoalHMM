'''
Created on Apr 20, 2015

@author: svendvn
'''
from IMCoalHMM.transitions import CTMCSystem, compute_upto, compute_between, projection_matrix
from IMCoalHMM.model import Model
from IMCoalHMM.ILS import Isolation1
from IMCoalHMM.statespace_generator import CoalSystem
from IMCoalHMM.state_spaces import make_rates_table_single
from IMCoalHMM.CTMC import make_ctmc
from IMCoalHMM.break_points import uniform_break_points, exp_break_points
from IMCoalHMM.admixture import outer_product, powerset, complement, population_lineages

from numpy import zeros, matrix, identity, ix_, exp
from numpy import sum as matrixsum
from numpy.testing import assert_almost_equal
import numpy
from pyZipHMM import Matrix
numpy.set_printoptions(threshold=numpy.nan)
from pympler import tracker


def printPyZipHMM(Matrix):
    finalString=""
    for i in range(Matrix.getWidth()):
        for j in range(Matrix.getHeight()):
            finalString=finalString+" "+str(Matrix[i,j])
        finalString=finalString+"\n"
    return finalString


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

def checkSymmetry(listOfTrees):
    pass


def admixture_state_space_map_one_way(from_space, to_space, p):
    """Constructs the mapping matrix from the 'from_space' state space to the 'to_space' state space
    assuming an admixture event where lineages in population 0 moves to population 1 with probability p
    and lineages in population 1 moves to population 0 with probability q."""
    destination_map = to_space.state_numbers
    map_matrix = matrix(zeros((len(from_space.states), len(to_space.states))))

    for state, from_index in from_space.state_numbers.items():
        population_1 = population_lineages(12, state)
        population_2 = population_lineages(3, state)
        print population_2

        # <debug>
        #print pretty_state(state)
        # </debug>
        total_prob = 0.0

        for x in powerset(population_1):
            cx = complement(population_1, x)

            ## Keep x and y in their respective population but move the other two...
            cx = frozenset((2, lin) for (p, lin) in cx)
            x = frozenset((1,lin) for (p,lin) in x)
            destination_state = frozenset(x).union(cx).union(population_2)
            change_probability = p**len(cx) * (1.0 - p)**len(x)
            to_index = destination_map[destination_state]

            # <debug>
            #print '->', pretty_state(destination_state),
            #print "p^{} (1-p)^{} q^{} (1-q)^{}".format(len(cx), len(x), len(cy), len(y))
            #print from_index, '->', to_index, '[{}]'.format(change_probability)
            # </debug>

            map_matrix[from_index, to_index] = change_probability
            total_prob += change_probability

        # <debug>
        #print
        #print total_prob
        # </debug>

        # We want to move to another state with exactly probability 1.0
        assert abs(total_prob - 1.0) < 1e-10

    return map_matrix



class Admixture3HMediumStep(CoalSystem):

    def __init__(self,starting_position):
        """Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are). This is the time interval after(in model, 
        we measure time backwards, so we say 'after' but in normal life we would say 'before') admixture. It is before divergence of  2 and 3"""

        super(Admixture3HMediumStep, self).__init__()

        self.transitions = [[('R', self.recombination)], [('C', self.coalesce)]]
        self.state_type = dict()
        # We need various combinations of initial states to make sure we build the full reachable state space.
        if starting_position==Admixture3HModel.INITIAL_21:
            left_1 = [frozenset([(1, (frozenset([1]), frozenset([])))]), frozenset([(2, (frozenset([1]), frozenset([])))])]
            right_1 = [frozenset([(1, (frozenset([]), frozenset([1])))]), frozenset([(2, (frozenset([]), frozenset([1])))])]
            left_2 = [frozenset([(1, (frozenset([2]), frozenset([])))]), frozenset([(2, (frozenset([2]), frozenset([])))])]
            right_2 = [frozenset([(1, (frozenset([]), frozenset([2])))]), frozenset([(2, (frozenset([]), frozenset([2])))])]
            left_3 = [frozenset([(3, (frozenset([3]), frozenset([])))])]
            right_3 = [frozenset([(3, (frozenset([]), frozenset([3])))])]
        else:
            left_1 = [frozenset([(1, (frozenset([1]), frozenset([])))]), frozenset([(2, (frozenset([1]), frozenset([])))])]
            right_1 = [frozenset([(1, (frozenset([]), frozenset([1])))]), frozenset([(2, (frozenset([]), frozenset([1])))])]
            left_2 = [frozenset([(3, (frozenset([2]), frozenset([])))])]
            right_2 = [frozenset([(3, (frozenset([]), frozenset([2])))])]
            left_3 = [frozenset([(3, (frozenset([3]), frozenset([])))])]
            right_3 = [frozenset([(3, (frozenset([]), frozenset([3])))])]            
        self.init = [l1 | r1 | l2 | r2 | l3 | r3 for l1 in left_1 for r1 in right_1 for l2 in left_2 for r2 in right_2 for l3 in left_3 for r3 in right_3]

        self.compute_state_space()
        self.sort_states()
        
    def sort_states(self):
        for state, index in self.states.items():
            left, right = extract_lineages(state)
            self.state_type.setdefault((left, right), []).append(index)
            
            
class Admixture3HAfterDivergence(CoalSystem):

    def __init__(self,starting_position):
        """Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are). This is the CTMC right after Admixture3HMediumStep."""

        super(Admixture3HAfterDivergence, self).__init__()

        self.transitions = [[('R', self.recombination)], [('C', self.coalesce)]]
        self.state_type = dict()
        # We need various combinations of initial states to make sure we build the full reachable state space.
        if starting_position==Admixture3HModel.INITIAL_21:
            left_1 = [frozenset([(1, (frozenset([1]), frozenset([])))]), frozenset([(23, (frozenset([1]), frozenset([])))])]
            right_1 = [frozenset([(1, (frozenset([]), frozenset([1])))]), frozenset([(23, (frozenset([]), frozenset([1])))])]
            left_2 = [frozenset([(1, (frozenset([2]), frozenset([])))]), frozenset([(23, (frozenset([2]), frozenset([])))])]
            right_2 = [frozenset([(1, (frozenset([]), frozenset([2])))]), frozenset([(23, (frozenset([]), frozenset([2])))])]
            left_3 = [frozenset([(23, (frozenset([3]), frozenset([])))])]
            right_3 = [frozenset([(23, (frozenset([]), frozenset([3])))])]
        else:
            left_1 = [frozenset([(1, (frozenset([1]), frozenset([])))]), frozenset([(23, (frozenset([1]), frozenset([])))])]
            right_1 = [frozenset([(1, (frozenset([]), frozenset([1])))]), frozenset([(23, (frozenset([]), frozenset([1])))])]
            left_2 = [frozenset([(23, (frozenset([2]), frozenset([])))])]
            right_2 = [frozenset([(23, (frozenset([]), frozenset([2])))])]
            left_3 = [frozenset([(23, (frozenset([3]), frozenset([])))])]
            right_3 = [frozenset([(23, (frozenset([]), frozenset([3])))])]
        self.init = [l1 | r1 | l2 | r2 | l3 | r3 for l1 in left_1 for r1 in right_1 for l2 in left_2 for r2 in right_2 for l3 in left_3 for r3 in right_3]

        self.compute_state_space()
        self.sort_states()es):
            path = self.reverse_tree_map[state]
            #likelihoods contains the probability of each alignment
            likelihoods = list()
            for align_column in range(no_alignment_columns):
                if align_column == no_alignment_columns - 1:
                    likelihoods.append(1) #this should be the sum of all the others
                else:
                    tree = self.get_tree
        
    def sort_states(self):
        for state, index in self.states.items():
            left, right = extract_lineages(state)
            self.state_type.setdefault((left, right), []).append(index)


def make_rates_table_4(coal_rate_123, recombination_rate):
    """Builds the rates table from the CTMC for the two-samples system for an isolation period."""
    table = dict()
    table[('C', 123, 123)] = coal_rate_123
    table[('R', 123, 123)] = recombination_rate
    return table  

def make_rates_table_3(coal_rate_23, coal_rate_1, recombination_rate):
    """Builds the rates table from the CTMC for the two-samples system
    for an isolation period."""
    table = dict()
    table[('C', 1, 1)] = coal_rate_1
    table[('C', 23, 23)] = coal_rate_23
    table[('R', 1, 1)] = recombination_rate
    table[('R', 23, 23)] = recombination_rate
    return table

def make_rates_table_2(coal_rate_1,coal_rate_2, coal_rate_3, recombination_rate):
    table = dict()
    table[('C', 1, 1)] = coal_rate_1
    table[('C', 2, 2)] = coal_rate_2
    table[('C', 3, 3)] = coal_rate_3
    table[('R', 1, 1)] = recombination_rate
    table[('R', 2, 2)] = recombination_rate
    table[('R', 3, 3)] = recombination_rate
    return table

def make_rates_table_1(coal_rate_12,coal_rate_3, recombination_rate):
    table = dict()
    table[('C', 12, 12)] = coal_rate_12
    table[('C', 3, 3)] = coal_rate_3
    table[('R', 12, 12)] = recombination_rate
    table[('R', 3, 3)] = recombination_rate
    return table
  

class ILSSystem(CoalSystem):
    def __init__(self):
        super(ILSSystem, self).__init__()
        self.state_type = dict()
        self.transitions = [[('R', self.recombination)], [('C', self.coalesce)]]

    def sort_states(self):
        for state, index in self.states.items():
            left, right = extract_lineages(state)
            self.state_type.setdefault((left, right), []).append(index)
            
class Isolation2(ILSSystem):
    def __init__(self, population_of_lineage_2):
        super(Isolation2, self).__init__()
        self.init = frozenset([(population, (frozenset([sample]), frozenset([sample])))
                               for population, sample in zip([12, population_of_lineage_2, 3], [1, 2, 3])])
        self.compute_state_space()
        self.init_index = self.states[self.init]
        self.sort_states()
    

class Admixture3HMediumCTMCSystem(CTMCSystem):
    
    def __init__(self, model,isolation_ctmc, medium_ctmc, afterdivergence_ctmc, ancestral_ctmc, break_points1, break_points2, break_points3, break_points4,p):
        
        super(Admixture3HMediumCTMCSystem, self).__init__(no_hmm_states=len(break_points1)+len(break_points2)+len(break_points3)+len(break_points4),
                                                  initial_ctmc_state=model.init_index)  #what to do with initial state?
        
        #saving for later
        self.isolation_ctmc = isolation_ctmc
        self.medium_ctmc = medium_ctmc
        self.afterdivergence_ctmc = afterdivergence_ctmc
        self.ancestral_ctmc = ancestral_ctmc
        self.break_points_1 = break_points1
        self.break_points_2 = break_points2
        self.break_points_3 = break_points3
        self.break_points_4 = break_points4
        self.no_medium_states=len(break_points2)
        self.no_afterdivergence_states=len(break_points3)
        self.no_isolation_states=len(break_points1)
        self.no_ancestral_states=len(break_points4)
        self.model=model
        print "making CTMC!"
        self.through_ = [None] * (self.no_isolation_states+self.no_medium_states + self.no_afterdivergence_states+self.no_ancestral_states - 1) #in the end a last term is added so that it becomes one longer

        for i in xrange(self.no_isolation_states - 1):
            self.through_[i] = isolation_ctmc.probability_matrix(break_points1[i+1] - break_points1[i])

        #the transition with admixture
        xx = isolation_ctmc.probability_matrix(break_points2[0] - break_points1[-1])
        projection = admixture_state_space_map_one_way(isolation_ctmc.state_space, medium_ctmc.state_space, p)
        print "her"
        self.through_[self.no_isolation_states - 1] = xx * projection
        print "bor"

        #transitions after admixture
        for i in xrange(self.no_isolation_states, self.no_medium_states + self.no_isolation_states - 1):
            ii = i - self.no_isolation_states
            self.through_[i] = medium_ctmc.probability_matrix(break_points2[ii+1] - break_points2[ii])
        print "trolden"
        #transition to almost ancestral
        def state_map_23_1(state):
            return frozenset([(23, nucleotides) if i == 2 or i==3 else (1,nucleotides) for (i, nucleotides) in state])
        
        xx = medium_ctmc.probability_matrix(break_points3[0] - break_points2[-1])
        projection = projection_matrix(medium_ctmc.state_space, afterdivergence_ctmc.state_space, state_map_23_1)
        self.through_[self.no_isolation_states+self.no_medium_states - 1] = xx * projection
        print "i"
        
        #transitions in afterdivence
        for i in xrange(self.no_isolation_states+self.no_medium_states, self.no_medium_states + self.no_isolation_states+self.no_afterdivergence_states - 1):
            ii=i-(self.no_isolation_states+self.no_medium_states)
            self.through_[i] = afterdivergence_ctmc.probability_matrix(break_points3[ii + 1] - break_points3[ii])
        
        print "sin"
        #transition to ancestral
        def state_map_123(state):
            return frozenset([(123, nucleotides) for (_, nucleotides) in state])        
        
        xx = afterdivergence_ctmc.probability_matrix(break_points4[0] - break_points3[-1])
        projection = projection_matrix(afterdivergence_ctmc.state_space, ancestral_ctmc.state_space, state_map_123)
        self.through_[self.no_isolation_states+self.no_medium_states+self.no_afterdivergence_states - 1] = xx * projection
        
        print "hule,"
        #transitions in ancestral
        for i in xrange(self.no_isolation_states+self.no_medium_states+self.no_afterdivergence_states, 
                        self.no_medium_states + self.no_isolation_states+self.no_afterdivergence_states+self.no_ancestral_states - 1):
            ii=i-(self.no_isolation_states+self.no_medium_states+self.no_afterdivergence_states)
            self.through_[i] = ancestral_ctmc.probability_matrix(break_points4[ii + 1] - break_points4[ii])
        print "som"
        #last transition
        pseudo_through = matrix(zeros((len(ancestral_ctmc.state_space.states), len(ancestral_ctmc.state_space.states))))
        pseudo_through[:, self.ancestral_ctmc.state_space.state_type[(STATE_E, STATE_E)][0]] = 1.0
        self.through_.append(pseudo_through)
             
        #constructing other lists of matrices
        upto0 = matrix(identity(len(isolation_ctmc.state_space.states)))
        self.upto_ = compute_upto(upto0, self.through_)
        self.between_ = compute_between(self.through_)
        print "er gul"
        
    def get_state_space(self, i):
        """Return the state space for interval i."""
        if i < self.no_isolation_states:
            return self.isolation_ctmc.state_space
        elif i<self.no_isolation_states+self.no_medium_states:
            return self.medium_ctmc.state_space
        elif i< self.no_isolation_states+self.no_medium_states+self.no_afterdivergence_states:
            return self.afterdivergence_ctmc.state_space
        else:
            return self.ancestral_ctmc.state_space
        
        
    def through(self, i):
        return self.through_[i]

    def up_to(self, i):
        return self.upto_[i]

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
        #print "check if joint has the same length as valid paths"
        #print str(no_states**2) + "=?=" + str(len(self.model.valid_paths))
        for path in self.model.valid_paths:
            i, j = self.model.get_path_indices(path)
            joint[i, j] = self.get_path_probability(path)
        return joint

    def compute_transition_probabilities(self):
        no_states = len(self.model.tree_map)

        joint = self.make_joint_matrix()
        for tree,num in self.model.tree_map.items():
#            print str(tree)
            outofdiagonal=range(0,num)+range(num+1,no_states)
#            print "prob from: "+str(sum([joint[num,j] for j in outofdiagonal]))+" prob stay:" + str(joint[num,num]) + " prob to"+ str(sum([joint[j,num] for j in outofdiagonal])) 
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
    
    INITIAL_12=0 #meaning that we have 1 individual in the admixed population and 2 in the other
    INITIAL_21=1 #meaning that we have 2 individuals in the admixed population and 2 in the other. 
        
    def __init__(self, initial_configuration, no_isolation_intervals, no_medium_intervals, no_afterdivergence_intervals, no_ancestral_intervals):
        
        super(Admixture3HModel, self).__init__()
        
        self.initial_configuration=initial_configuration
        if self.initial_configuration==Admixture3HModel.INITIAL_21:
            self.isolation_state_space = Isolation2(12)#the 12 means that lineage 2 should belong to populaiton 12.
        else:
            self.isolation_state_space = Isolation2(3)
        self.medium_state_space = Admixture3HMediumStep(self.initial_configuration)
        self.afterdivergence_state_space=Admixture3HAfterDivergence(self.initial_configuration)
        self.ancestral_state_space = Isolation1()
        
        self.init_index=self.isolation_state_space.init_index
        
        self.no_isolation_intervals=no_isolation_intervals
        self.no_medium_intervals=no_medium_intervals
        self.no_afterdivergence_intervals=no_afterdivergence_intervals
        self.no_ancestral_intervals=no_ancestral_intervals
        
        self.no_intervals=self.no_ancestral_intervals+self.no_isolation_intervals+self.no_medium_intervals+self.no_afterdivergence_intervals
        self.valid_paths_ = None
        self.tree_map = None
        self.reverse_tree_map = None

        self.make_valid_paths()
        self.index_marginal_trees()
        
    @property
    def initial(self):
        """The initial state index in the bottom-most matrix.

        :returns: the state space index of the initial state.
        :rtype: int
        """
        return self.init_index

    def get_state_space(self, i):
        """Return the state space for interval i."""
        if i < self.no_isolation_intervals:
            return self.isolation_state_space
        elif i<self.no_isolation_intervals + self.no_medium_intervals:
            return self.medium_state_space
        elif i<self.no_isolation_intervals+self.no_medium_intervals+self.no_afterdivergence_intervals:
            return self.afterdivergence_state_space
        else:
            return self.ancestral_state_space


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
        
#         print "parameters"
#         print len(parameters)
#         print parameters
        tau_1, tau_2, tau_3, coal_12, coal_3, coal_1, coal_2, coal_23, coal_last, recomb, p = parameters
        isolation_rates = make_rates_table_1(coal_rate_12=coal_12, coal_rate_3=coal_3, recombination_rate=recomb)
        medium_rates = make_rates_table_2(coal_rate_1=coal_1, coal_rate_2=coal_2,coal_rate_3=coal_3, recombination_rate=recomb)
        afterdivergence_rates = make_rates_table_3(coal_rate_1=coal_1, coal_rate_23=coal_23, recombination_rate=recomb)
        ancestral_rates = make_rates_table_4(coal_rate_123=coal_last, recombination_rate=recomb)
        
        isolation_ctmc = make_ctmc(self.isolation_state_space, isolation_rates)
        medium_ctmc = make_ctmc(self.medium_state_space, medium_rates)
        afterdivergence_ctmc = make_ctmc(self.afterdivergence_state_space, afterdivergence_rates)
        ancestral_ctmc = make_ctmc(self.ancestral_state_space, ancestral_rates)
        self.isolation_breakpoints=uniform_break_points(self.no_isolation_intervals,0,tau_1)
        self.medium_breakpoints=uniform_break_points(self.no_medium_intervals, tau_1,tau_1+tau_2)
        self.afterdivergence_breakpoints=uniform_break_points(self.no_ancestral_intervals, tau_1+tau_2, tau_3)
        self.ancestral_breakpoints = exp_break_points(self.no_ancestral_intervals, coal_last, tau_1+tau_2+tau_3)
        #print self.isolation_breakpoints
        #print self.middle_breakpoints
        #print self.ancestral_breakpoints
        
        return Admixture3HMediumCTMCSystem(self,isolation_ctmc, medium_ctmc,afterdivergence_ctmc, ancestral_ctmc, self.isolation_breakpoints, self.medium_breakpoints, self.afterdivergence_breakpoints, self.ancestral_breakpoints,p)
    
    def emission_points(self, *parameters):
        """Expected coalescence times between between tau1 and tau2"""
        try:
            (tau1, tau2, tau3, coal_12, coal_1, coal_2, coal_3, coal_23, coal_last, recomb, p), outgroup = parameters, None
        except ValueError:
            try:
                tau1, tau2, tau3, coal_12, coal_1, coal_2, coal_3, coal_23, coal_last, recomb, p, outgroup = parameters
            except ValueError:
                tau1, tau2, tau3, coal_12, coal_1, coal_2, coal_3, coal_23, coal_last, recomb, p = parameters
                outgroup=0
        print self.isolation_breakpoints
        print self.medium_breakpoints
        print self.afterdivergence_breakpoints
        print self.ancestral_breakpoints
        
        breaks_12 = list(self.isolation_breakpoints)+list(self.medium_breakpoints) + [float(tau2+tau1)] # turn back into regular python...
        mean_coal_1=(coal_12+coal_3)/2.0
        epoch_1_emission_points = [ ((mean_coal_1*s+1)*exp(-s*mean_coal_1)-(t*mean_coal_1+1)*exp(-t*mean_coal_1))/mean_coal_1/(exp(-mean_coal_1*s)-exp(-mean_coal_1*t)) for s, t in zip(self.isolation_breakpoints[0:-1], self.isolation_breakpoints[1:])]
        s,t=self.isolation_breakpoints[-1],self.medium_breakpoints[0]
        epoch_1_emission_points.append(((mean_coal_1*s+1)*exp(-s*mean_coal_1)-(t*mean_coal_1+1)*exp(-t*mean_coal_1))/mean_coal_1/(exp(-mean_coal_1*s)-exp(-mean_coal_1*t)))
        
        mean_coal_2=(coal_1+coal_2+coal_3)/3.0
        epoch_2_emission_points = [((mean_coal_2*s+1)*exp(-s*mean_coal_2)-(t*mean_coal_2+1)*exp(-t*mean_coal_2))/mean_coal_2/(exp(-mean_coal_2*s)-exp(-mean_coal_2*t)) for s, t in zip(self.medium_breakpoints[0:-1], self.medium_breakpoints[1:])]
        s,t=self.medium_breakpoints[-1],self.afterdivergence_breakpoints[0]
        epoch_2_emission_points.append(((mean_coal_1*s+1)*exp(-s*mean_coal_1)-(t*mean_coal_1+1)*exp(-t*mean_coal_1))/mean_coal_1/(exp(-mean_coal_1*s)-exp(-mean_coal_1*t)))

        mean_coal_3=(coal_1+coal_23)/2.0
        epoch_3_emission_points = [((mean_coal_3*s+1)*exp(-s*mean_coal_3)-(t*mean_coal_3+1)*exp(-t*mean_coal_3))/mean_coal_3/(exp(-mean_coal_3*s)-exp(-mean_coal_3*t)) for s, t in zip(self.afterdivergence_breakpoints[0:-1], self.afterdivergence_breakpoints[1:])]
        s,t=self.afterdivergence_breakpoints[-1],self.ancestral_breakpoints[0]
        epoch_3_emission_points.append(((mean_coal_1*s+1)*exp(-s*mean_coal_1)-(t*mean_coal_1+1)*exp(-t*mean_coal_1))/mean_coal_1/(exp(-mean_coal_1*s)-exp(-mean_coal_1*t)))

        mean_coal_4=coal_last
        epoch_4_emission_points = [((mean_coal_4*s+1)*exp(-s*mean_coal_4)-(t*mean_coal_4+1)*exp(-t*mean_coal_4))/mean_coal_4/(exp(-mean_coal_4*s)-exp(-mean_coal_4*t)) for s, t in zip(self.ancestral_breakpoints[0:-1], self.ancestral_breakpoints[1:])]
        
        epoch_4_emission_points.append(self.ancestral_breakpoints[-1] + 1/coal_last)

        #print "list og" + str(epoch_1_emission_points + epoch_2_emission_points)
        print epoch_1_emission_points
        print epoch_2_emission_points
        print epoch_3_emission_points
        print epoch_4_emission_points
        print len(epoch_1_emission_points+epoch_2_emission_points+epoch_3_emission_points+epoch_4_emission_points)
        
        return epoch_1_emission_points + epoch_2_emission_points+epoch_3_emission_points+epoch_4_emission_points, outgroup

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
            print path
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
            """Computes the probability of a tree assuming base i at the root. Every history which results in the tree is summed over"""
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
        #print "no states"+str(no_states)
        emission_probabilities = Matrix(no_states, no_alignment_columns, )

        branch_shortening = [0, 0, 0] # FIXME: not sure how to pass in this information from the script...

        for state in xrange(no_states):
            path = self.reverse_tree_map[state]
            #likelihoods contains the probability of each alignment
            likelihoods = list()
            for align_column in range(no_alignment_columns):
                if align_column == no_alignment_columns - 1:
                    likelihoods.append(1) #this should be the sum of all the others
                else:
                    tree = self.get_tree(path, align_column, coalescence_times, outgroup, branch_shortening)
                    likelihoods.append(sum(prior[i] * prob_tree(tree, i, subst_model) for i in range(4)))
            #print sum(likelihoods) #should be 2, because 1 is inserted in last entry
            for align_column, emission_prob in enumerate(x/sum(likelihoods) for x in likelihoods):
                emission_probabilities[state, align_column] = emission_prob

        return emission_probabilities

    
    
        # We override this one from the Model class because we cannot directly reuse the 2-sample code.
    def build_hidden_markov_model(self, parameters):
        """Build the hidden Markov model matrices from the model-specific parameters."""
        #print "Building a HMM"
        ctmc_system = self.build_ctmc_system(*parameters)
        initial_probabilities, transition_probabilities = ctmc_system.compute_transition_probabilities()
        emission_probabilities = self.emission_matrix(*parameters)
        return initial_probabilities, transition_probabilities, emission_probabilities, ctmc_system.break_points_1+ctmc_system.break_points_2+ctmc_system.break_points_3+ctmc_system.break_points_4
    
    
    
tra = tracker.SummaryTracker()
tra.print_diff()
ad=Admixture3HModel(Admixture3HModel.INITIAL_21,3,3,3,3)
tra.print_diff()
tr=ad.build_hidden_markov_model([0.001,0.004,0.009, 1000,1000,1000,1000,1000,1000,0.4,0.1])
tra.print_diff()
tr=ad.build_hidden_markov_model([0.001,0.004,0.009, 1000,1000,1000,1000,1000,1000,0.4,0.9])

tra.print_diff()


