"""Code wrapping the code in the (mini-)ziphmm module to match the interface in pyZipHMM.

The code here just wrap the hmm likelihood computation from the ziphmm module in a Forwarder
class that matches the interface there was in the earlier pyZipHMM package.
"""

import ziphmm
import numpy as np

class Forwarder:

	def __init__(self, input_filename, NSYM):
		with open(input_filename) as finp:
			obs = np.array(map(int, finp.read().split()), dtype=np.int32)
			self.NSYM = NSYM
			self.new_obs, self.sym2pair, self.new_nsyms = ziphmm.preprocess_raw_observations(obs, self.NSYM)


	def forward(self, init_probs, trans_probs, emission_probs):
		return ziphmm.zip_forward(init_probs, trans_probs, emission_probs, 
			   		              self.sym2pair, self.new_obs, self.NSYM, self.new_nsyms)

