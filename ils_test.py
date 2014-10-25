

import numpy
from IMCoalHMM.ILS import ILSModel, pretty_marginal_time_path

def joint_probabilities(model, parameters):
    """Build the hidden Markov model matrices from the model-specific parameters."""
    ctmc_system = model.build_ctmc_system(*parameters)
    return ctmc_system.make_joint_matrix()


n = 2
model = ILSModel(1, n+1)

trees_13 = {}
trees_23 = {}

for tree, index in model.tree_map.items():
    if len(tree) != 2:
        continue # look only at the resolved trees...

    begin, i, first_coalescent = tree[0]
    _, j, _ = tree[1]
    x, y = first_coalescent
    if len(x) < len(y):
        x, y = y, x

    # skip species topology
    if x == frozenset([1, 2]):
        continue

    if x == frozenset([1, 3]):
        trees_13[(i-1,j-1)] = index
    elif x == frozenset([2, 3]):
        trees_23[(i-1,j-1)] = index
    else:
        assert False, "Impossible state"

#print trees_13, trees_23

def make_parameters(tau1 = 0.001, tau2 = 1e-20, coal1 = 1000.0, coal2 = 1000.0, coal3 = 10000000.0,
                    coal12 = 0.0, coal123 = 1.0, recombination_rate = 0.4):
    return numpy.array([tau1, tau2, coal1, coal2, coal3, coal12, coal123, recombination_rate])


def get_diagonals(parameters):
    joint = joint_probabilities(model, parameters)
    j13, j23 = [], []
    for i, j in trees_13:
        j13.append(joint[trees_13[(i,j)], trees_13[(i,j)]])
        j23.append(joint[trees_23[(i,j)], trees_23[(i,j)]])
    return j13, j23, [(x-y) for x, y in zip(j13, j23)]

print get_diagonals(make_parameters(coal1 = 0.0, coal2 = 1000000000.0))
print get_diagonals(make_parameters(coal1 = 0.0, coal2 = 100.0))
print get_diagonals(make_parameters(coal1 = 0.0, coal2 = 10.0))
print get_diagonals(make_parameters(coal1 = 0.0, coal2 = 0.0))
