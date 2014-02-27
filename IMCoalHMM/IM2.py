from statespace_generator import IM

class IM2(IM):
	'''Class for IM system with exactly two samples.'''
	
	def _has_left_coalesced(self, state):
		for pop,(left,right) in state:
			if len(left) == 2:
				return True
		return False
	
	def _has_right_coalesced(self, state):
		for pop,(left,right) in state:
			if len(right) == 2:
				return True
		return False

	def __init__(self):
		'''Constructs the state space and collect B, L, R and E states (see the 
		CoalHMM papers for what these are).
		
		Also collects the indices in the state space for the three (realistic) initial
		states, with both chromosomes in population 1 or in two or one from each.'''
		
		IM.__init__(self,[1,2])
		
		self.states, self.transitions = self.compute_state_space()
		
		i11_state = frozenset([(1,(frozenset([s]),frozenset([s]))) for s in [1,2]])
		i22_state = frozenset([(2,(frozenset([s]),frozenset([s]))) for s in [1,2]])
		i12_state = frozenset([(s,(frozenset([s]),frozenset([s]))) for s in [1,2]])
		
		self.i11_index = self.states[i11_state]
		self.i12_index = self.states[i12_state]
		self.i22_index = self.states[i22_state]
		
		self.B_states = []
		self.L_states = []
		self.R_states = []
		self.E_states = []
		for state,index in self.states.items():
			L,R = self._has_left_coalesced(state), self._has_right_coalesced(state)
			if not L and not R:
				self.B_states.append(index)
			elif L and not R:
				self.L_states.append(index)
			elif not L and R:
				self.R_states.append(index)
			elif L and R:
				self.E_states.append(index)
			else:
				assert False, "it should be impossible to reach this point."
		
		
def make_rates_table(C1, C2, R, M12, M21):
	'''Builds the rates table from the CTMC for the two-samples system.'''
	species = [0,1]
	table = dict()
	table[('C',1,1)] = C1
	table[('C',2,2)] = C2
	table[('R',1,1)] = R
	table[('R',2,2)] = R
	table[('M',1,2)] = M12
	table[('M',2,1)] = M21
	return table
	
if __name__ == '__main__':
	system = IM2()
	print len(system.B_states)
	print len(system.L_states)
	print len(system.R_states)
	print len(system.E_states)


	