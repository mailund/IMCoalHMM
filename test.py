from numpy import matrix, zeros
from itertools import chain, combinations


## Helper functions
def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return set(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def complement(universe, subset):
    """Extract universe \ subset."""
    return set(universe).difference(subset)


def population_lineages(population, lineages):
    return set((p, lineage) for p, lineage in lineages if p == population)


def outer_product(set1, set2):
    for x in set1:
        for y in set2:
            yield x, y


## Output for debugging... makes the states easier to read
def pretty_state(state):
    """Presents a coalescence system state in an easier to read format."""
    def pretty_set(s):
        if len(s) == 0:
            return "{}"
        else:
            return "{{{}}}".format(','.join(str(x) for x in s))

    def lineage_map(lin):
        p, (l, r) = lin
        return "[{}, ({},{})]".format(p, pretty_set(l), pretty_set(r))

    return " ".join(map(lineage_map, state))


def admixture_state_space_map(from_space, to_space, p, q):
    """Constructs the mapping matrix from the 'from_space' state space to the 'to_space' state space
    assuming an admixture event where lineages in population 1 moves to population 2 with probability p
    and lineages in population 2 moves to population 1 with probability q."""
    destination_map = to_space.state_numbers

    map_matrix = matrix(zeros((len(from_space.states), len(to_space.states))))

    for state, from_index in from_space.state_numbers.items():
        population_1 = population_lineages(1, state)
        population_2 = population_lineages(2, state)

        # <debug>
        #print pretty_state(state)
        # </debug>
        total_prob = 0.0

        for x, y in outer_product(powerset(population_1), powerset(population_2)):
            cx = complement(population_1, x)
            cy = complement(population_2, y)

            ## Keep x and y in their respective population but move the other two...
            cx = frozenset((2, lin) for (p, lin) in cx)
            cy = frozenset((1, lin) for (p, lin) in cy)

            destination_state = frozenset(x).union(cx).union(y).union(cy)
            change_probability = p**len(cx) * (1.0 - p)**len(x) * q**len(cy) * (1.0 - q)**len(y)
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


## Testing...
from IMCoalHMM.state_spaces import Isolation, Migration

isolation = Isolation()
## The admixture state space is the same as the migration state space, just without migration transitions...
migration = Migration()

print admixture_state_space_map(isolation, migration, 0.0, 0.1)

