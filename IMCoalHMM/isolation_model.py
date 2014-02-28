'''Code for constructing the HMM for an isolation model.
'''

from numpy import zeros
from scipy import matrix

from I2 import Isolation2, make_rates_table_isolation
from I2 import Single2,    make_rates_table_single
from CTMC import CTMC
from i_transitions import compute_transition_probabilities
from break_points import exp_break_points
from emissions import emission_matrix

class IsolationModel(object):
    '''Class wrapping the code that generates an isolation model HMM.'''
    
    def __init__(self):
        '''Construct the model.
        
        This builds the state spaces for the CTMCs but the matrices for the
        HMM since those will depend on the rate parameters.'''

        super(IsolationModel, self).__init__()
        
        self.isolation_state_space = Isolation2()
        self.single_state_space = Single2()
        
        self.Pr = matrix(zeros((len(self.isolation_state_space.states),
                                len(self.single_state_space.states))))
        def map_tokens(token):
            pop, nucs = token
            return 0, nucs
        for state, isolation_index in self.isolation_state_space.states.items():
            ancestral_state = frozenset(map(map_tokens, state))
            ancestral_index = self.single_state_space.states[ancestral_state]
            self.Pr[isolation_index, ancestral_index] = 1.0

    def build_HMM(self, no_states, split_time, coal_rate, recomb_rate):
        '''Construct CTMCs and compute HMM matrices given the split time
        and the rates.'''

        isolation_rates = make_rates_table_isolation(coal_rate, coal_rate, 
                                                     recomb_rate)
        isolation_ctmc = CTMC(self.isolation_state_space, isolation_rates)
        single_rates = make_rates_table_single(coal_rate, recomb_rate)
        single_ctmc = CTMC(self.single_state_space, single_rates)

        break_points = exp_break_points(no_states, coal_rate, split_time)

        pi, T = compute_transition_probabilities(isolation_ctmc,
                                                 self.Pr,
                                                 single_ctmc,
                                                 break_points)
        E = emission_matrix(break_points, coal_rate)

        return pi, T, E

