'''A migration system with a sample size of two.

'''

from statespace_generator import Migration

class Migration2(Migration):
    '''Class for IM system with exactly two samples.'''

    def __init__(self):
        '''Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are).

        Also collects the indices in the state space for the three
        (realistic) initial states, with both chromosomes in population 1
        or in 2 or one from each.'''

        super(Migration2, self).__init__([1, 2])
        

        self.compute_state_space()

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


def make_rates_table_migration(C1, C2, R, M12, M21):
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
    system = Migration2()
    print len(system.begin_states)
    print len(system.left_states)
    print len(system.right_states)
    print len(system.end_states)


if __name__ == '__main__':
    main()
