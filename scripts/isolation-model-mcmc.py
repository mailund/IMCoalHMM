#!/usr/bin/env python

"""Script for estimating parameters in an isolation model.
"""

from argparse import ArgumentParser

from IMCoalHMM.isolation_model import IsolationModel
from IMCoalHMM.likelihood import Likelihood
from pyZipHMM import Forwarder

# Status functionionality for priors
from scipy.stats import norm
from numpy.random import random
from math import log, exp
from numpy import array

class LogNormPrior(object):
    def __init__(self, log_mean, propsal_sd=None):
        self.log_mean = log_mean
        if propsal_sd is not None:
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

class MCMC(object):

    def __init__(self, priors, log_likelihood):
        self.priors = priors
        self.logL = log_likelihood
        
        self.current_theta = array([pi.sample() for pi in self.priors])
        self.current_posterior = self.log_prior(self.current_theta) + self.logL(self.current_theta)

    def log_prior(self, theta):
        log_prior = 0.0
        for i in xrange(len(theta)):
            log_prior += log(self.priors[i].pdf(theta[i]))
        return log_prior
    
    def step(self, temperature = 1.0):
        new_theta = array([self.priors[i].proposal(self.current_theta[i]) for i in xrange(len(self.current_theta))])
        new_prior = self.log_prior(new_theta)
        new_logL = self.logL(new_theta)
        new_posterior = new_prior + new_logL
        
        if new_posterior > self.current_posterior or \
            random() < exp(new_posterior / temperature - self.current_posterior / temperature):
            self.current_theta = new_theta
            self.current_posterior = new_posterior
            
    def sample(self, thinning = 1, temperature = 1.0):
        for _ in xrange(thinning):
            self.step(temperature)
        return self.current_theta, self.current_posterior

def main():
    """
    Run the main script.
    """
    usage = """%(prog)s [options] <forwarder dirs>

This program samples the posterior parameters of an isolation model with two species
and uniform coalescence and recombination rates."""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.0")

    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/dev/stdout",
                        help="Output file for the estimate (/dev/stdout)")

    parser.add_argument("--states",
                        type=int,
                        default=10,
                        help="Number of intervals used to discretize the time (10)")

    parser.add_argument("-n", "--samples",
                        type=int,
                        default=500,
                        help="Number of samples to draw (500)")

    parser.add_argument("-k", "--thinning",
                        type=int,
                        default=100,
                        help="Number of MCMC steps between samples (100)")

    meta_params = [
        ('split', 'split time in substitutions', 1e6 / 1e9),
        ('theta', 'effective population size in 4Ne substitutions', 1e6 / 1e9),
        ('rho', 'recombination rate in substitutions', 0.4),
    ]

    for parameter_name, description, default in meta_params:
        parser.add_argument("--%s" % parameter_name,
                            type=float,
                            default=default,
                            help="Meta-parameter mean of the %s (%g)" % (description, default))

    parser.add_argument('alignments', nargs='+', help='Alignments in ZipHMM format')

    options = parser.parse_args()
    if len(options.alignments) < 1:
        parser.error("Input alignment not provided!")

    # Specify priors and proposal distributions... 
    # I am sampling in log-space to make it easier to make a random walk
    split_prior = LogNormPrior(log(options.split))
    coal_prior = LogNormPrior(log(1/(options.theta/2)))
    rho_prior = LogNormPrior(log(options.rho))
    priors = [split_prior, coal_prior, rho_prior]

    # Draw initial parameters from the priors
    init_params = [pi.sample() for pi in priors]

    # Read data and provide likelihood function
    forwarders = [Forwarder.fromDirectory(arg) for arg in options.alignments]
    log_likelihood = Likelihood(IsolationModel(options.states), forwarders)

    mcmc = MCMC(priors, log_likelihood)

    def transform(params):
        split_time, coal_rate, recomb_rate = params
        return split_time, 2 / coal_rate, recomb_rate

    with open(options.outfile, 'w') as outfile:
        print >> outfile, '\t'.join(['split.time', 'theta', 'rho', 'posterior'])
        
        for _ in xrange(options.samples):
            params, post = mcmc.sample(thinning=options.thinning)
            print >> outfile, '\t'.join(map(str, transform(params) + (post,)))


if __name__ == '__main__':
    main()
