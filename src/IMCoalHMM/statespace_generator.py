"""Code for constructing the state space of a coalescent system.
"""


def has_left_coalesced(state):
    """Predicate checking if a state is coalesced on the left."""
    for _, (left, _) in state:
        if len(left) == 2:
            return True
    return False


def has_right_coalesced(state):
    """Predicate checking if a state is coalesced on the right."""
    for _, (_, right) in state:
        if len(right) == 2:
            return True
    return False


class CoalSystem(object):
    """Abstract class for the two nucleotide coalescence system.

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

    The one exception is if the transition returns None.  This is
    interpreted as a guard violation and the transition update is
    aborted.

    If the states are provided in the constructor (by default they are
    not) this is interpreted as the fixed state space and only
    transitions will be computed.  If the state space exploration
    encounters a state not in the provided list, it is considered an
    error and the state space exploration will be aborted.
    """

    def __init__(self):
        self.transitions = []
        self.state_numbers = None
        self.states = None  # Set in compute_state_space

        self.init = None  # Should be set by a sub-class!

        # These are filled in by computer_state_space when
        # the concrete state space has been build.
        self.begin_states = []
        self.left_states = []
        self.right_states = []
        self.end_states = []

    def successors(self, state):
        """Calculate all successors of "state".

        Generates all successors of "state" and lets you iterate over the
        edges from "state" to tis successors.  Each generated value is a
        pair of transition type and state, where transition type is either
        "C" for a coalescence event or "R" for a recombination event.
        """

        tokens = list(state)

        for ttype, tfunc in self.transitions[0]:
            for token in tokens:
                pre = frozenset([token])
                for pop_a, pop_b, post in tfunc(token):
                    new_state = state.difference(pre).union(post)
                    yield ttype, pop_a, pop_b, new_state

        for ttype, tfunc in self.transitions[1]:
            for i in xrange(len(tokens)):
                for j in xrange(i):

                    pre = frozenset([tokens[i], tokens[j]])
                    pop_a, pop_b, tproduct = tfunc(tokens[i], tokens[j])
                    if tproduct is None:
                        continue
                    post = tproduct
                    new_state = state.difference(pre).union(post)
                    yield ttype, pop_a, pop_b, new_state

    def compute_state_space(self):
        """Computes the CTMC system."""
        seen = {self.init}
        unprocessed = [self.init]
        self.state_numbers = {self.init: 0}
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

        self.states = mapped_state_numbers
        self.transitions = [(remapping[src], (trans, pop1, pop2), remapping[dest])
                            for src, (trans, pop1, pop2), dest in edges]

        for state, index in self.states.items():
            has_left = has_left_coalesced(state)
            has_right = has_right_coalesced(state)
            if not has_left and not has_right:
                self.begin_states.append(index)
            elif has_left and not has_right:
                self.left_states.append(index)
            elif not has_left and has_right:
                self.right_states.append(index)
            elif has_left and has_right:
                self.end_states.append(index)
            else:
                assert False, "it should be impossible to reach this point."

    # Transitions: these will be in all our systems
    @staticmethod
    def recombination(token):
        """Compute the tokens we get from a recombination on "token".

        Returns None if the recombination would just return
        "token" again, so we avoid returning "empty" tokens.
        """
        pop, nucs = token
        left, right = nucs
        if not (left and right):
            return []

        return [(pop, pop, frozenset([(pop, (left, frozenset())),
                                      (pop, (frozenset(), right))]))]

    @staticmethod
    def coalesce(token1, token2):
        """Construct a new token by coalescening "token1" and "token2"."""
        pop1, nuc1 = token1
        pop2, nuc2 = token2
        if pop1 != pop2:
            return -1, -1, None  # abort transition...

        left1, right1 = nuc1
        left2, right2 = nuc2
        left, right = left1.union(left2), right1.union(right2)
        return pop1, pop2, frozenset([(pop1, (left, right))])
