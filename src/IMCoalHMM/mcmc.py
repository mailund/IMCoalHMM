"""
Module for generic MCMC code.

"""

from scipy.stats import norm, expon
from numpy.random import random, randint
from math import log, exp
from numpy import array


class LogNormPrior(object):
    """Prior and proposal distribution. The prior is a log-normal and steps are a
    random walk in log-space."""

    def __init__(self, log_mean, proposal_sd=None):
        self.log_mean = log_mean
        if proposal_sd is not None:
            self.proposal_sd = proposal_sd
        else:
            self.proposal_sd = 0.1

    def pdf(self, x):
        return norm.pdf(log(x), loc=self.log_mean)

    def sample(self):
        return exp(norm.rvs(loc=self.log_mean, size=1)[0])

    def proposal(self, x):
        log_step = norm.rvs(loc=log(x), scale=self.proposal_sd, size=1)[0]
        return exp(log_step)


class ExpLogNormPrior(object):
    """Prior and proposal distribution. The prior is an exponential and steps are a
    random walk in log-space."""

    def __init__(self, mean, proposal_sd=None):
        self.mean = mean
        if proposal_sd is not None:
            self.proposal_sd = proposal_sd
        else:
            self.proposal_sd = 0.1

    def pdf(self, x):
        return expon.pdf(x, scale=self.mean)

    def sample(self):
        return expon.rvs(scale=self.mean, size=1)[0]

    def proposal(self, x):
        log_step = norm.rvs(loc=log(x), scale=self.proposal_sd, size=1)[0]
        return exp(log_step)


class MCMC(object):
    def __init__(self, priors, log_likelihood, thinning):
        self.priors = priors
        self.log_likelihood = log_likelihood
        self.thinning = thinning

        self.current_theta = array([pi.sample() for pi in self.priors])
        self.current_posterior = self.log_prior(self.current_theta) + self.log_likelihood(self.current_theta)

    def log_prior(self, theta):
        log_prior = 0.0
        for i in xrange(len(theta)):
            log_prior += log(self.priors[i].pdf(theta[i]))
        return log_prior

    def step(self, temperature=1.0):
        new_theta = array([self.priors[i].proposal(self.current_theta[i]) for i in xrange(len(self.current_theta))])
        new_prior = self.log_prior(new_theta)
        new_log_likelihood = self.log_likelihood(new_theta)
        new_posterior = new_prior + new_log_likelihood

        if new_posterior > self.current_posterior or \
            random() < exp(new_posterior / temperature - self.current_posterior / temperature):

            self.current_theta = new_theta
            self.current_posterior = new_posterior

    def sample(self, temperature=1.0):
        for _ in xrange(self.thinning):
            self.step(temperature)
        return self.current_theta, self.current_posterior


class MC3(object):
    """A Metropolis-Coupled MCMC."""
    def __init__(self, priors, log_likelihood, no_chains, thinning, switching):
        self.no_chains = no_chains
        self.chains = [MCMC(priors, log_likelihood, thinning) for _ in xrange(no_chains)]
        self.thinning = thinning
        self.switching = switching

    def sample(self):
        '''Sample after running "thinning" steps with a proposal for switching chains at each
        "switching" step.'''

        for itr in xrange(self.thinning):
            for temperature, chain in enumerate(self.chains):
                chain.step(temperature + 1.0)  # +1 because enumerate starts from zero

            if itr % self.switching == 0:
                i = randint(0, self.no_chains)
                j = randint(0, self.no_chains)


                if i != j:
                    chain_i, chain_j = self.chains[i], self.chains[j]
                    current = chain_i.current_posterior / (i + 1) + chain_j.current_posterior / (j + 1)
                    new = chain_j.current_posterior / (i + 1) + chain_i.current_posterior / (j + 1)
                    if random() < exp(new - current):
                        self.chains[i], self.chains[j] = self.chains[j], self.chains[i]

        return self.chains[0].current_theta, self.chains[0].current_posterior