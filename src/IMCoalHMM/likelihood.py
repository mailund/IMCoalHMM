"""Code for combining data with a model for computing and maximising
likelihoods.
"""

import scipy.optimize


class Likelihood(object):
    """Combining model and data."""

    def __init__(self, model, forwarders):
        """Bind a model to sequence data in the form of ZipHMM Forwarders.

        :param model: Any demographic model that can build a hidden Markov model.
        :type model: IMCoalHMM.model.Model
        :param forwarders: ZipHMM forwarder or forwarders for computing the HMM likelihood.
        :type forwarders: pyZipHMM.Forwarder | list[pyZipHMM.Forwarder]
        """
        super(Likelihood, self).__init__()
        self.model = model

        if hasattr(forwarders, '__iter__'):
            self.forwarders = forwarders
        else:
            self.forwarders = [forwarders]

    def __call__(self, *parameters):
        """Compute the log-likelihood at a set of parameters."""
        if not self.model.valid_parameters(*parameters):
            return -float('inf')

        init_probs, trans_probs, emission_probs = self.model.build_hidden_markov_model(*parameters)
        return sum(forwarder.forward(init_probs, trans_probs, emission_probs) for forwarder in self.forwarders)


def maximum_likelihood_estimate(log_likelihood, initial_parameters,
                                optimizer_method="Nelder-Mead",
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

    :param log_likelihood: The Likelihood wrapper needed for computing the likelihood.
    :type log_likelihood: Likelihood
    :param initial_parameters: The initial set of parameters. Model specific.
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
            log_file.flush()

    def minimize_wrapper(parameters):
        return -log_likelihood(parameters)

    options = {'disp': False}
    # Set optimizer specific options

    #FIXME: if I have other ways of checking valid parameters for later models, this
    # really needs to be updated as well!
    if optimizer_method in ['Anneal', 'L-BFGS-B', 'TNC', 'SLSQP']:
        bounds = [(0, None)] * len(initial_parameters)
        result = scipy.optimize.minimize(fun=minimize_wrapper, x0=initial_parameters,
                                         method=optimizer_method, bounds=bounds,
                                         callback=log_callback, options=options)
    else:
        result = scipy.optimize.minimize(fun=minimize_wrapper, x0=initial_parameters,
                                         method=optimizer_method,
                                         callback=log_callback, options=options)

    #print result
    return result.x
