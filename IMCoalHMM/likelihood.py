'''Code for combining data with a model for computing and maximising
likelihoods.
'''

class Likelihood(object):
    '''Combining model and data.'''

    def __init__(self, model, forwarder):
        '''Bind a model to sequence data in the form of a zipHMM Forwarder.'''
        self.model = model
        self.forwarder = forwarder
    
    def __call__(self, *parameters):
        '''Compute the log-likelihood at a set of parameters.'''
        pi, T, E = self.model.build_HMM(*parameters)
        return self.forwarder.forward(pi, T, E)

import scipy.optimize
def maximum_likelihood_estimate(minimize_wrapper, initial_parameters,
                                optimizer = scipy.optimize.fmin):
    return optimizer(minimize_wrapper, initial_parameters)

