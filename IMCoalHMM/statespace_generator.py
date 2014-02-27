'''Code for constructing the state space of a coalescent system.
'''

import exceptions

class CoalSystem(object):
    '''Abstract class for the two nucleotide coalescence system.

    Implements the basic state space exploration functionality, but
    leaves it to sub-classes to specify the actual system.

    The transitions in the system are specified in self.transitions
    that is a list of lists of transitions, where self.transitions[i]
    contains the transitions that take i+1 tokens as input.  The
    transitions consists of pairs, where the first element is a
    transition label (like "R" for recombination) and the second is
    the function that translates the pre-set into the post-set.

    For convenience, the pre-set is give as just the tokens (so not a
    set but a normal list of arguments).

    The post-set should always be a set or a list of sets, since it is
    otherwise hard to figure out if the transition produces single
    tokens, multiple tokens or no tokens.

    If a set is returned, it is interpreted as the post set of firing
    the transition.  If a list (of sets) is returned, it is
    interpreted as if the transition can fire and
    non-deterministically produce different post sets; the lists
    should contain all those posts sets.

    The one exeception is if the transition returns None.  This is
    interpreted as a guard violation and the transition update is
    aborted.

    If the states are provided in the constructor (by default they are
    not) this is interpreted as the fixed state space and only
    transitions will be computed.  If the state space exploration
    encounters a state not in the provided list, it is considered an
    error and the state space exploration will be aborted.
    '''

    def __init__(self):
        self.transitions = []
        self.state_numbers = None

    def successors(self, state):
        '''Calculate all successors of "state".

        Generates all successors of "state" and lets you iterate over the
        edges from "state" to tis successors.  Each generated value is a
        pair of transition type and state, where transition type is either
        "C" for a coalescence event or "R" for a recombination event.
        '''

        tokens = list(state)

        for ttype, tfunc in self.transitions[0]:
            for token in tokens:
                pre = frozenset([token])
                tproduct = tfunc(token)
                for pop_a, pop_b, post in tfunc(token):
                    new_state = state.difference(pre).union(post)
                    yield ttype, pop_a, pop_b, new_state

        for ttype, tfunc in self.transitions[1]:
            for i in xrange(len(tokens)):
                for j in xrange(i):

                    pre = frozenset([tokens[i], tokens[j]])
                    pop_a, pop_b, tproduct = tfunc(tokens[i], tokens[j])
                    if tproduct == None:
                        continue
                    post = tproduct
                    new_state = state.difference(pre).union(post)
                    yield ttype, pop_a, pop_b, new_state

    def initial_state(self):
        '''Virtual function for specifying the initial state.'''
        raise exceptions.NotImplementedError()

    def compute_state_space(self):
        '''Computes the CTMC system.'''
        initial_states = self.initial_state()

        seen = set(initial_states)
        unprocessed = initial_states
        self.state_numbers = {}
        for i, state in enumerate(initial_states):
            self.state_numbers[state] = i
        edges = []

        while unprocessed:
            state = unprocessed.pop()
            state_no = self.state_numbers[state]
            for trans, pop1, pop2, dest in self.successors(state):
                assert state != dest, "We don't like self-loops!"

                if dest not in self.state_numbers:
                    self.state_numbers[dest] = len(self.state_numbers)

                if dest not in seen:
                    unprocessed.append(dest)
                    seen.add(dest)

                edges.append((state_no,
                             (trans, pop1, pop2),
                             self.state_numbers[dest]))

        remapping = {}
        mapped_state_numbers = {}
        for state_no in set(self.state_numbers.values()):
            remapping[state_no] = len(remapping)
        for state, state_no in self.state_numbers.iteritems():
            mapped_state_numbers[state] = remapping[state_no]
        self.state_numbers = mapped_state_numbers
        edges = [(remapping[src], (trans, pop1, pop2), remapping[dest]) \
                    for src, (trans, pop1, pop2), dest in edges]

        return self.state_numbers, edges



class BasicCoalSystem(CoalSystem):
    '''The basic two-nucleotide coalescence system.'''

    def recombination(self, token):
        '''Compute the tokens we get from a recombination on "token".

        Returns None if the recombination would just return
        "token" again, so we avoid returning "empty" tokens.
        '''
        _, nucs = token
        left, right = nucs
        if not (left and right):
            return []
        return [(0, 0, frozenset([(0, (left, frozenset())),
                             (0, (frozenset(), right))]))]

    def coalesce(self, token1, token2):
        '''Construct a new token by coalescening "token1" and "token2".'''
        _, nuc1 = token1
        _, nuc2 = token2
        left1, right1 = nuc1
        left2, right2 = nuc2
        left, right = left1.union(left2), right1.union(right2)
        return 0, 0, frozenset([(0, (left, right))])

    def initial_state(self):
        '''Build the initial state for this system.

        This doesn't necessarily mean that there is only a single
        initial state, but that doesn't matter much since we just need
        a state in an initial connected component for this to work...
        '''
        return self.init

    def __init__(self, species):
        CoalSystem.__init__(self)
        self.transitions = [[('R', self.recombination)],
                            [('C', self.coalesce)]]
        self.init = [frozenset([(0, (frozenset([s]), frozenset([s]))) \
                            for s in species])]



class IM(CoalSystem):
    '''The basic two-nucleotide coalescence system.'''

    def recombination(self, token):
        '''Compute the tokens we get from a recombination on "token".

        Returns None if the recombination would just return
        "token" again, so we avoid returning "empty" tokens.
        '''
        pop, nucs = token
        left, right = nucs
        if not (left and right):
            return []
        return [(pop, pop, frozenset([(pop, (left, frozenset())),
                                 (pop, (frozenset(), right))]))]

    def coalesce(self, token1, token2):
        '''Construct a new token by coalescening "token1" and "token2".'''
        pop1, nuc1 = token1
        pop2, nuc2 = token2
        if pop1 != pop2:
            return -1, -1, None # abort transition...


        left1, right1 = nuc1
        left2, right2 = nuc2
        left, right = left1.union(left2), right1.union(right2)
        return pop1, pop2, frozenset([(pop1, (left, right))])

    def migrate(self, token):
        '''Move nucleotides from one population to another'''
        pop, nuc = token
        res = [(pop, pop2, frozenset([(pop2, nuc)])) \
                for pop2 in self.legal_migrations[pop]]
        return res

    def initial_state(self):
        '''Build the initial state for this system.

        This doesn't necessarily mean that there is only a single
        initial state, but that doesn't matter much since we just need
        a state in an initial connected component for this to work...
        '''
        return self.init[:]

    def __init__(self, species):
        CoalSystem.__init__(self)

        self.legal_migrations = dict()
        for sample in species:
            self.legal_migrations[sample] = \
                frozenset([other for other in species if sample != other])

        self.transitions = [[('R', self.recombination),
                             ('M', self.migrate)],
        					[('C', self.coalesce)]]
        self.init = [frozenset([(sample,
                                (frozenset([sample]),
                                 frozenset([sample])))
                        for sample in species])]


def pretty_lineage(lin):
    '''Pretty representation of a lineage.'''
    return ''.join(str(e) for e in lin)

def pretty_state(state):
    '''Pretty textual representation of a state ...'''
    return [(pop, pretty_lineage(lin)) for (pop, lin) in state]

def pretty_coal_class(state):
    '''Pretty textual representation of a state, showing not the
    populations but only which lineages have coalesced.'''
    return '/'.join(pretty_lineage(lin) for _, lin in state)


def main():
    '''Test.'''
    state_space = IM(range(2))
    states, transitions = state_space.compute_state_space()
    print len(states), 'and', len(transitions), 'transitions'

if __name__ == "__main__":
    main()
