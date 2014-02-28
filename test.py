
from numpy import zeros
from scipy import matrix

from IMCoalHMM.I2 import Isolation2, make_rates_table_isolation
from IMCoalHMM.I2 import Single2,    make_rates_table_single
from IMCoalHMM.CTMC import CTMC
from IMCoalHMM.i_transitions import compute_transition_probabilities
from IMCoalHMM.emissions import emission_matrix

# FIXME: this should really be moved to the library
isolation_state_space = Isolation2()
isolation_rates = make_rates_table_isolation(1, 0.5, 4e-4)
isolation_ctmc = CTMC(isolation_state_space, isolation_rates)

coal_rate = 1.5
single_state_space = Single2()
single_rates = make_rates_table_single(coal_rate, 4e-4)
single_ctmc = CTMC(single_state_space, single_rates)

Pr = matrix(zeros((len(isolation_state_space.states),
                   len(single_state_space.states))))
            
def map_tokens(token):
    pop, nucs = token
    return 0, nucs

for state, isolation_index in isolation_state_space.states.items():
    ancestral_state = frozenset(map(map_tokens, state))
    ancestral_index = single_state_space.states[ancestral_state]
    Pr[isolation_index, ancestral_index] = 1.0

break_points = [1,2,3,4]
pi, T = compute_transition_probabilities(isolation_ctmc,
                                         Pr,
                                         single_ctmc,
                                         break_points)

E = emission_matrix(break_points, coal_rate)

from pyZipHMM import Forwarder

forwarder = Forwarder.fromDirectory('examples/example_data.ziphmm')
logL = forwarder.forward(pi, T, E)
print logL