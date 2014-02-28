'''Code for an isolation model with two populations and two samples.

'''

from statespace_generator import Single, Isolation
from statespace_generator import has_left_coalesced, has_right_coalesced

class Isolation2(Isolation):
    '''Class for IM system with exactly two samples.'''

    def __init__(self):
        '''Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are).'''

        super(Isolation2, self).__init__([1, 2])

        self.compute_state_space()

        i12_state = frozenset([(sample,
                                (frozenset([sample]), frozenset([sample])))
                                for sample in [1, 2]])
        self.i12_index = self.states[i12_state]

def make_rates_table_isolation(C1, C2, R):
    '''Builds the rates table from the CTMC for the two-samples system.'''
    table = dict()
    table[('C', 1, 1)] = C1
    table[('C', 2, 2)] = C2
    table[('R', 1, 1)] = R
    table[('R', 2, 2)] = R
    return table

class Single2(Single):
    '''Class for a merged ancestral populatoin'''

    def __init__(self):
        '''Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are).'''

        super(Single2, self).__init__()
        self.compute_state_space()

def make_rates_table_single(C, R):
    '''Builds the rates table from the CTMC for the two-samples system.'''
    table = dict()
    table[('C', 0, 0)] = C
    table[('R', 0, 0)] = R
    return table



def main():
    "Test"
    isolation = Isolation2()
    assert len(isolation.B_states) == 4
    assert len(isolation.L_states) == 0
    assert len(isolation.R_states) == 0
    assert len(isolation.E_states) == 0

    single = Single2()
    assert len(single.B_states) == 7
    assert len(single.L_states) == 3
    assert len(single.R_states) == 3
    assert len(single.E_states) == 2


if __name__ == '__main__':
    main()
