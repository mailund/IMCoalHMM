
from IMCoalHMM.statespace_generator import CoalSystem
from IMCoalHMM.transitions import projection_matrix
from IMCoalHMM.CTMC import make_ctmc


class ILSSystem(CoalSystem):
    def __init__(self):
        super(ILSSystem, self).__init__()
        self.transitions = [[('R', self.recombination)], [('C', self.coalesce)]]


class Isolation3(ILSSystem):
    def __init__(self):
        super(Isolation3, self).__init__()
        self.init = frozenset([(sample, (frozenset([sample]), frozenset([sample]))) for sample in [1, 2, 3]])
        self.compute_state_space()


class Isolation2(ILSSystem):
    def __init__(self):
        super(Isolation2, self).__init__()
        self.init = frozenset([(population, (frozenset([sample]), frozenset([sample])))
                               for population, sample in zip([12, 12, 3], [1, 2, 3])])
        self.compute_state_space()


class Isolation1(ILSSystem):
    def __init__(self):
        super(Isolation1, self).__init__()
        self.init = frozenset([(population, (frozenset([sample]), frozenset([sample])))
                               for population, sample in zip([123, 123, 123], [1, 2, 3])])
        self.compute_state_space()


def make_rates_table_isolation_3(coal_rate_1, coal_rate_2, coal_rate_3, recombination_rate):
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


def make_rates_table_isolation_2(coal_rate_12, coal_rate_3, recombination_rate):
    """Builds the rates table from the CTMC for the two-samples system
    for an isolation period."""
    table = dict()
    table[('C', 12, 12)] = coal_rate_12
    table[('C', 3, 3)] = coal_rate_3
    table[('R', 12, 12)] = recombination_rate
    table[('R', 3, 3)] = recombination_rate
    return table


def make_rates_table_isolation_1(coal_rate_123, recombination_rate):
    """Builds the rates table from the CTMC for the two-samples system
    for an isolation period."""
    table = dict()
    table[('C', 123, 123)] = coal_rate_123
    table[('R', 123, 123)] = recombination_rate
    return table


def state_map_32(state):
    def lineage_map(lin):
        population, nucleotides = lin
        if population == 3:
            return 3, nucleotides
        else:
            return 12, nucleotides
    return frozenset(lineage_map(lin) for lin in state)


def state_map_21(state):
    return frozenset([(123, nucleotides) for (_, nucleotides) in state])


epoch_1 = make_ctmc(Isolation3(), make_rates_table_isolation_3(1.0, 1.0, 1.0, 0.4))
epoch_2 = make_ctmc(Isolation2(), make_rates_table_isolation_2(1.0, 1.0, 0.4))
epoch_3 = make_ctmc(Isolation1(), make_rates_table_isolation_1(1.0, 0.4))

projection_32 = projection_matrix(epoch_1.state_space, epoch_2.state_space, state_map_32)
projection_21 = projection_matrix(epoch_2.state_space, epoch_3.state_space, state_map_21)

print projection_32.shape
print projection_21.shape
