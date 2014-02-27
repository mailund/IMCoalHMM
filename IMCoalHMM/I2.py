'''Code for an isolation model with two populations and two samples.

'''

from statespace_generator import Single, Isolation
from statespace_generator import has_left_coalesced, has_right_coalesced

class Isolation2(Isolation):
    '''Class for IM system with exactly two samples.'''

    def __init__(self):
        '''Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are).

        Also collects the indices in the state space for the three
        (realistic) initial states, with both chromosomes in population 1
        or in 2 or one from each.'''

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

def main():
    "Test"
    isolation = Isolation2()
    assert len(isolation.B_states) == 4
    assert len(isolation.L_states) == 0
    assert len(isolation.R_states) == 0
    assert len(isolation.E_states) == 0


if __name__ == '__main__':
    main()
