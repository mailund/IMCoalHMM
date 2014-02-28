'''Code for combining data with a model for computing and maximising
likelihoods.
'''

class Likelihood(object):
    '''Combining model and data.'''

    def __init__(self, model, forwarder):
        '''Bind a model to sequence data in the form of a zipHMM Forwarder.'''
        super(Likelihood, self).__init__()
        self.model = model
        self.forwarder = forwarder
    
    def __call__(self, *parameters):
        '''Compute the log-likelihood at a set of parameters.'''
        pi, T, E = self.model.build_HMM(*parameters)
        return self.forwarder.forward(pi, T, E)

import scipy.optimize
def maximum_likelihood_estimate(wrapper, initial_parameters,
                                optimizer = scipy.optimize.fmin):
    '''Maximum likelihood estimation.
    
    This function requires a wrapper around the likelihood computation
    that will be model specific, splitting the paramters to be optimized from those
    that should not, such as number of states, epochs etc.
    
    It also requires an initial parameter point for the optimization, and a numerical
    algorithm for optimisation (a default from scipy is fmin).
    
    The function returns the maximum likelihood parameters.
    '''
    return optimizer(wrapper, initial_parameters, disp=False)

