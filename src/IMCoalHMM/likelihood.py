"""Code for combining data with a model for computing and maximising
likelihoods.
"""


class Likelihood(object):
    """Combining model and data."""

    def __init__(self, model, forwarders):
        """Bind a model to sequence data in the form of a zipHMM Forwarder.

        :param model: Any demographic model that can build a hidden Markov model.
        :param forwarders: ZipHMM forwarder or forwarders for computing the HMM likelihood.
        :type forwarders: pyZipHMM.Forwarder | list[pyZipHMM.Forwarder]
        """
        # FIXME: have a superclass for models?

        super(Likelihood, self).__init__()
        self.model = model

        if hasattr(forwarders, '__iter__'):
            self.forwarders = forwarders
        else:
            self.forwarders = [forwarders]

    def __call__(self, *parameters):
        """Compute the log-likelihood at a set of parameters."""
        init_probs, trans_probs, emission_probs = self.model.build_hidden_markov_model(*parameters)
        return sum(forwarder.forward(init_probs, trans_probs, emission_probs) for forwarder in self.forwarders)


import scipy.optimize


def maximum_likelihood_estimate(wrapper, initial_parameters,
                                optimizer=scipy.optimize.fmin,
                                log_file=None,
                                log_param_transform=lambda x: x):
    """Maximum likelihood estimation.

    This function requires a wrapper around the likelihood computation
    that will be model specific, splitting the parameters to be optimized from those
    that should not, such as number of states, epochs etc.

    If a log file files is provided, the optimisation is logged to that file through
    a call back. The "log_param_transform" function provides a handle to transform
    the parameters to a different space for logging.

    It also requires an initial parameter point for the optimization, and a numerical
    algorithm for optimisation (a default from scipy is fmin).

    :param wrapper: The wrapper needed for computing the likelihood.
    :param initial_parameters: The initial set of parameters. Model specific.
    :param optimizer: The algorithm used for numerical optimization.
    :param log_file: Progress will be logged to this file/stream.
    :param log_param_transform: A function to map the optimization parameter space
     into a model parameter space.

    :returns: the maximum likelihood parameters.
    """
    log_callback = None
    if log_file:
        def log_callback(parameters):
            log_params = [str(param) for param in log_param_transform(parameters)]
            print >> log_file, '\t'.join(log_params)

    return optimizer(wrapper, initial_parameters, callback=log_callback, disp=False)
