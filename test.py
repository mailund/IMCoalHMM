
from IMCoalHMM.IM2 import IM2, make_rates_table_migration
from IMCoalHMM.CTMC import CTMC

state_space = IM2()
rates = make_rates_table_migration(1,0.5,4e-4,0.2,0.2)
coal_system = CTMC(state_space, rates)

# As a sanity check we can check that it is impossible to get from a L to a R state
for left in state_space.L_states:
	for right in state_space.R_states:
		assert left != right
		assert coal_system.Q[left,right] == 0
		assert coal_system.Q[right,left] == 0

def left_coalesced(P, initial):
	'''Marginalize to get the probability that the left nucleotide has coalesced.'''
	prob = P[initial,state_space.L_states].sum() + P[initial,state_space.E_states].sum()
	return prob

from scipy import linspace
times = linspace(0, 10, num = 50)
coal11, coal12, coal22 = [], [], []
for t in times:
	P = coal_system.probability_matrix(t)
	coal11.append(left_coalesced(P, state_space.i11_index))
	coal12.append(left_coalesced(P, state_space.i12_index))
	coal22.append(left_coalesced(P, state_space.i22_index))

from pylab import plot, show, legend
plot(times, coal11, label='11')
plot(times, coal12, label='12')
plot(times, coal22, label='22')
legend()
show()
