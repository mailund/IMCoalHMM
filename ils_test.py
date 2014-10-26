

import numpy
from numpy import ix_
from IMCoalHMM.ILS import ILSModel, pretty_marginal_time_path


n = 3
model = ILSModel(1, n)

trees_12 = {}
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

    if x == frozenset([1, 2]):
        trees_12[(i, j)] = index
    elif x == frozenset([1, 3]):
        trees_13[(i, j)] = index
    elif x == frozenset([2, 3]):
        trees_23[(i, j)] = index
    else:
        assert False, "Impossible state"


keys_12 = [(i, j) for i in range(n) for j in range(i+1,n+1)]
keys_ils = [(i, j) for i in range(1, n) for j in range(i+1,n+1)]
idx_12 = [trees_12[k] for k in keys_12]
idx_13 = [trees_13[k] for k in keys_ils]
idx_23 = [trees_23[k] for k in keys_ils]

for idx in idx_12:
    print idx, ':', pretty_marginal_time_path(model.reverse_tree_map[idx])
print
for idx in idx_13:
    print idx, ':', pretty_marginal_time_path(model.reverse_tree_map[idx])
print


def make_parameters(tau1=0.002, tau2=0.001, coal1=1000.0, coal2=1000.0, coal3=1000.0,
                    coal12=1000.0, coal123=1000.0, recombination_rate=0.4):
    return numpy.array([tau1, tau2, coal1, coal2, coal3, coal12, coal123, recombination_rate])


def get_joints(parameters):
    ctmc_system = model.build_ctmc_system(*parameters)
    joint = ctmc_system.make_joint_matrix()
    j13 = joint[ix_(idx_12, idx_13)]
    j23 = joint[ix_(idx_12, idx_23)]
    return j13 - j23


print get_joints(make_parameters(coal1=0.0, coal2=1000.0))
print get_joints(make_parameters(coal1=0.0, coal2=100.0))
print get_joints(make_parameters(coal1=0.0, coal2=10.0))
print get_joints(make_parameters(coal1=0.0, coal2=0.0))
