from numpy import zeros
from scipy import matrix
from scipy.linalg import expm

class CTMC(object):
	'''Class representing the CTMC for the back-in-time coalescent.'''
	
	def __init__(self, states, transitions, rates_table):
		'''Create the CTMC based on a state space and a mapping
		from transition labels to rates.'''
		
		self.Q = matrix(zeros((len(states),len(states))))
		
		for src,trans,dst in transitions:
			self.Q[src,dst] = rates_table[trans]
			
		for i in xrange(len(states)):
			self.Q[i,i] = - self.Q[i,:].sum()
			
	def probability_matrix(self, delta_t):
		'''Computes the transition probability matrix for a time period of delta_t.'''
		return expm(self.Q * delta_t)


if __name__ == '__main__':
	from statespace_generator import IM

	species = ['H','C']
	states, trans = IM(['H','C']).compute_state_space()
	
	coal_rates = {'H':1, 'C':0.5}
	rec_rates = {'H':4e-4, 'C':4e-4}
	mig_rates = {('H','C'):1,('C','H'):1}
	rates_table = dict()
	for s in species:
		rates_table[('C',s,s)] = coal_rates[s]
		rates_table[('R',s,s)] = rec_rates[s]
		for ss in species:
			if s != ss:
				rates_table[('M',s,ss)] = mig_rates[(s,ss)]
	
	ctmc = CTMC(states, trans, rates_table)
	P = ctmc.probability_matrix(1.0)

	print P[0,:]
	