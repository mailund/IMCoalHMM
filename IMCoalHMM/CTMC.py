from numpy import zeros
from scipy import matrix
from scipy.linalg import expm

class CTMC(object):
	'''Class representing the CTMC for the back-in-time coalescent.'''
	
	def __init__(self, state_space, rates_table):
		'''Create the CTMC based on a state space and a mapping
		from transition labels to rates.'''
		
		# Remember these, just to decouple state space from CTMC
		# in other parts of the code...
		self.B_states = state_space.B_states
		self.L_states = state_space.L_states
		self.R_states = state_space.R_states
		self.E_states = state_space.E_states
		
		self.Q = matrix(zeros((len(state_space.states),len(state_space.states))))
		
		for src,trans,dst in state_space.transitions:
			self.Q[src,dst] = rates_table[trans]
		
		for i in xrange(len(state_space.states)):
			self.Q[i,i] = - self.Q[i,:].sum()
	
	def probability_matrix(self, delta_t):
		'''Computes the transition probability matrix for a time period of delta_t.'''
		return expm(self.Q * delta_t)


if __name__ == '__main__':
	from IM2 import IM2, make_rates_table
	
	state_space = IM2()
	rates_table = make_rates_table(1,1,4e-4,0.2,0.2)
	
	ctmc = CTMC(state_space, rates_table)
	P = ctmc.probability_matrix(1.0)
	
	print P[0,:]
	