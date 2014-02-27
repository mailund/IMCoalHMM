'''A migration system with a sample size of two.

'''

from statespace_generator import IM

def _has_left_coalesced(state):
    '''Predicate checking if a state is coalesced on the left.'''
    for _, (left, _) in state:
        if len(left) == 2:
            return True
    return False

def _has_right_coalesced(state):
    '''Predicate checking if a state is coalesced on the right.'''
    for _, (_, right) in state:
        if len(right) == 2:
            return True
    return False



class IM2(IM):
    '''Class for IM system with exactly two samples.'''

    def __init__(self):
        '''Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are).

        Also collects the indices in the state space for the three
        (realistic) initial states, with both chromosomes in population 1
        or in 2 or one from each.'''

        IM.__init__(self, [1, 2])

        self.states, self.transitions = self.compute_state_space()

        i11_state = frozenset([(1,
                                (frozenset([sample]), frozenset([sample])))
                                for sample in [1, 2]])
        i22_state = frozenset([(2,
                                (frozenset([sample]), frozenset([sample])))
                                for sample in [1, 2]])
        i12_state = frozenset([(sample,
                                (frozenset([sample]), frozenset([sample])))
                                for sample in [1, 2]])

        self.i11_index = self.states[i11_state]
        self.i12_index = self.states[i12_state]
        self.i22_index = self.states[i22_state]

        self.B_states = []
        self.L_states = []
        self.R_states = []
        self.E_states = []
        for state, index in self.states.items():
            has_left = _has_left_coalesced(state)
            has_right = _has_right_coalesced(state)
            if not has_left and not has_right:
                self.B_states.append(index)
            elif has_left and not has_right:
                self.L_states.append(index)
            elif not has_left and has_right:
                self.R_states.append(index)
            elif has_left and has_right:
                self.E_states.append(index)
            else:
                assert False, "it should be impossible to reach this point."


def make_rates_table(C1, C2, R, M12, M21):
    '''Builds the rates table from the CTMC for the two-samples system.'''
    table = dict()
    table[('C', 1, 1)] = C1
    table[('C', 2, 2)] = C2
    table[('R', 1, 1)] = R
    table[('R', 2, 2)] = R
    table[('M', 1, 2)] = M12
    table[('M', 2, 1)] = M21
    return table

def main():
    "Test"
    system = IM2()
    print len(system.B_states)
    print len(system.L_states)
    print len(system.R_states)
    print len(system.E_states)


if __name__ == '__main__':
    main()
