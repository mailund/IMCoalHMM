"""Concrete state spaces for specific demographic models.
"""

from IMCoalHMM.statespace_generator import CoalSystem


class Isolation(CoalSystem):
    """Class for IM system with exactly two samples."""

    def __init__(self):
        """Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are)."""

        super(Isolation, self).__init__()

        self.transitions = [[('R', self.recombination)],
                            [('C', self.coalesce)]]

        self.init = frozenset([(sample,
                                (frozenset([sample]),
                                 frozenset([sample])))
                               for sample in [1, 2]])

        self.compute_state_space()

        i12_state = frozenset([(sample,
                                (frozenset([sample]), frozenset([sample])))
                               for sample in [1, 2]])
        self.i12_index = self.states[i12_state]


def make_rates_table_isolation(coal_rate_1, coal_rate_2, recomb_rate):
    """Builds the rates table from the CTMC for the two-samples system
    for an isolation period."""
    table = dict()
    table[('C', 1, 1)] = coal_rate_1
    table[('C', 2, 2)] = coal_rate_2
    table[('R', 1, 1)] = recomb_rate
    table[('R', 2, 2)] = recomb_rate
    return table


class Single(CoalSystem):
    """Class for a merged ancestral population."""

    def __init__(self):
        """Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are)."""

        super(Single, self).__init__()

        self.transitions = [[('R', self.recombination)],
                            [('C', self.coalesce)]]

        samples = [1, 2]
        self.init = frozenset([(0,
                                (frozenset([sample]),
                                 frozenset([sample])))
                               for sample in samples])

        self.compute_state_space()


def make_rates_table_single(coal_rate, recomb_rate):
    """Builds the rates table from the CTMC for the two-samples system in a single population."""
    table = dict()
    table[('C', 0, 0)] = coal_rate
    table[('R', 0, 0)] = recomb_rate
    return table


class Migration(CoalSystem):
    """Class for IM system with exactly two samples."""

    def migrate(self, token):
        """Move nucleotides from one population to another"""
        pop, nuc = token
        res = [(pop, pop2, frozenset([(pop2, nuc)])) for pop2 in self.legal_migrations[pop]]
        return res

    def __init__(self):
        """Constructs the state space and collect B, L, R and E states (see the
        CoalHMM papers for what these are).

        Also collects the indices in the state space for the three
        (realistic) initial states, with both chromosomes in population 1
        or in 2 or one from each."""

        super(Migration, self).__init__()

        self.legal_migrations = dict()
        species = [1, 2]
        for sample in species:
            self.legal_migrations[sample] = \
                frozenset([other for other in species if sample != other])

        self.transitions = [[('R', self.recombination),
                             ('M', self.migrate)],
                            [('C', self.coalesce)]]
        self.init = frozenset([(sample,
                                (frozenset([sample]),
                                 frozenset([sample])))
                               for sample in species])

        self.compute_state_space()

        i11_state = frozenset([(1, (frozenset([sample]), frozenset([sample])))
                               for sample in [1, 2]])
        i22_state = frozenset([(2, (frozenset([sample]), frozenset([sample])))
                               for sample in [1, 2]])
        i12_state = frozenset([(sample, (frozenset([sample]), frozenset([sample])))
                               for sample in [1, 2]])

        self.i11_index = self.states[i11_state]
        self.i12_index = self.states[i12_state]
        self.i22_index = self.states[i22_state]


def make_rates_table_migration(coal_rate_1, coal_rate_2, recomb_rate,
                               migration_rate_12, migration_rate_21):
    """Builds the rates table from the CTMC for the two-samples system.
    """
    table = dict()
    table[('C', 1, 1)] = coal_rate_1
    table[('C', 2, 2)] = coal_rate_2
    table[('R', 1, 1)] = recomb_rate
    table[('R', 2, 2)] = recomb_rate
    table[('M', 1, 2)] = migration_rate_12
    table[('M', 2, 1)] = migration_rate_21
    return table
