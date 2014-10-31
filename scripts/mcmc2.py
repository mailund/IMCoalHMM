"""
Module for generic MCMC code but with some extensions to the original MCMC.

"""
from pyZipHMM import Forwarder
from likelihood2 import Likelihood

from scipy.stats import norm, expon
from numpy.random import random, randint
from math import log, exp
from numpy import array

from multiprocessing import Process, Queue

def printPyZipHMM(Matrix):
    finalString=""
    for i in range(Matrix.getWidth()):
        for j in range(Matrix.getHeight()):
            finalString=finalString+" "+str(Matrix[i,j])
        finalString=finalString+"\n"
    return finalString

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
    def __init__(self, priors, log_likelihood, thinning, transferminator=None, mixtureWithScew=0 , mixtureWithSwitch=0, switcher=None, startVal=None):
        self.priors = priors
        self.log_likelihood = log_likelihood
        self.thinning = thinning
        self.transform = transferminator
        self.adapParam='none'    
        if startVal is None:
            self.current_theta = array([pi.sample() for pi in self.priors])
        else:
            self.current_theta=array(startVal)
            
        self.rejectedSwitches={}
        self.acceptedSwitches={}
        
        self.current_prior = self.log_prior(self.current_theta)
        forget,forget2,self.current_likelihood = self.log_likelihood(self.current_theta)
        self.current_posterior = self.current_prior + self.current_likelihood
        self.mixtureWithScew=mixtureWithScew
        self.mixtureWithSwitch=mixtureWithSwitch
        self.switcher=switcher
        self.rejections=0
        self.accepts=0
        self.current_transitionMatrix=forget
        self.current_initialDistribution=forget2
            

    def log_prior(self, theta):
        log_prior = 0.0
        for i in xrange(len(theta)):
            pdf = self.priors[i].pdf(theta[i])
            # FIXME: We shouldn't ever get a non-positive pdf so I should find out how it happens.
            if pdf <= 0.0: return -float("inf")
            log_prior += log(pdf)
        return log_prior

    def step(self, temperature=1.0):
        propPar=self.current_theta
        new_theta = array([self.priors[i].proposal(propPar[i]) for i in xrange(len(self.current_theta))])
        new_prior = self.log_prior(new_theta)
        new_transitionMatrix, new_initialDistribution, new_log_likelihood = self.log_likelihood(new_theta)
        new_posterior = new_prior + new_log_likelihood

        
        if new_posterior > self.current_posterior or \
                        random() < exp(new_posterior / temperature - self.current_posterior / temperature):
            self.current_theta = new_theta
            self.current_prior = new_prior
            self.current_transitionMatrix, self.current_initialDistribution, self.current_likelihood = new_transitionMatrix, new_initialDistribution, new_log_likelihood
            self.current_posterior = new_posterior
            self.accepts+=1
        else:
            self.rejections+=1
        
            
    def ScewStep(self, temperature=1.0):
        print self.current_theta[0]
        propPar=self.transform.first_transform(self.current_theta)
        new_thetaTmp = array([self.priors[i].proposal(propPar[i]) for i in xrange(len(self.current_theta))])
        new_theta= array(self.transform.after_transform(new_thetaTmp))
        new_prior = self.log_prior(new_theta)
        new_transitionMatrix, new_initialDistribution, new_log_likelihood = self.log_likelihood(new_theta)
        new_posterior = new_prior + new_log_likelihood

        if new_posterior > self.current_posterior:
            alpha=1
        else:
            alpha=min(1, exp(new_posterior / temperature - self.current_posterior / temperature))
        if new_posterior > self.current_posterior or \
                        random() < exp(new_posterior / temperature - self.current_posterior / temperature):
            self.current_theta = new_theta
            self.current_prior = new_prior
            self.current_transitionMatrix, self.current_initialDistribution, self.current_likelihood = new_transitionMatrix, new_initialDistribution, new_log_likelihood
            self.current_posterior = new_posterior
            self.accepts+=1
            self.adapParam=self.transform.update_alpha(True, alpha)
        else: 
            self.rejections+=1
            self.adapParam=self.transform.update_alpha(False,alpha)
            
    def switchStep(self,temperature):
        new_theta,whatSwitch=self.switcher(self.current_theta)
        new_prior = self.log_prior(new_theta)
        new_transitionMatrix, new_initialDistribution, new_log_likelihood = self.log_likelihood(new_theta)
        new_posterior = new_prior + new_log_likelihood
        
        if new_posterior > self.current_posterior or \
                        random() < exp(new_posterior / temperature - self.current_posterior / temperature):
            self.current_theta = new_theta
            self.current_prior = new_prior
            self.current_transitionMatrix, self.current_initialDistribution, self.current_likelihood = new_transitionMatrix, new_initialDistribution, new_log_likelihood
            self.current_posterior = new_posterior
            self.accepts+=1
            if whatSwitch in self.acceptedSwitches:
                self.acceptedSwitches[whatSwitch]+=1
            else:
                self.acceptedSwitches[whatSwitch]=1
        else: 
            self.rejections+=1
            if whatSwitch in self.rejectedSwitches:
                self.rejectedSwitches[whatSwitch]+=1
            else:
                self.rejectedSwitches[whatSwitch]=1
    
    def getSwitchStatistics(self):
        return self.acceptedSwitches, self.rejectedSwitches

    def sample(self, temperature=1.0):
        self.accepts=0
        self.rejections=0
        for _ in xrange(self.thinning):
            if(randint(0,self.mixtureWithSwitch+1)==1):
                self.switchStep(temperature)
            elif(self.mixtureWithScew>0):
                self.ScewStep(temperature)
            else:
                self.step(temperature)
        return self.current_theta, self.current_prior, self.current_likelihood, self.current_posterior, self.accepts, self.rejections, self.adapParam
    
    def transformToIdef(self, inarray):
        return inarray
    
    def transformFromIdef(self,inarray):
        return inarray


class RemoteMCMC(object):
    """ MCMC that is designed to run in another process for parallel execution.
    """

    def __init__(self, priors, input_files, model, thinning):
        self.priors = priors
        self.input_files = input_files
        self.model = model
        self.thinning = thinning
        self.chain = None
        self.task_queue = Queue()
        self.response_queue = Queue()

    def _set_chain(self):
        forwarders = [Forwarder.fromDirectory(arg) for arg in self.input_files]
        log_likelihood = Likelihood(self.model, forwarders)
        self.chain = MCMC(priors=self.priors, log_likelihood=log_likelihood, thinning=self.thinning)

    def __call__(self):
        self._set_chain()
        while True:
            temperature = self.task_queue.get()
            self.response_queue.put(self.chain.sample(temperature))


class RemoteMCMCProxy(object):
    """Local handle to a remote MCMC object."""

    def __init__(self, priors, input_files, model, thinning):
        self.remote_chain = RemoteMCMC(priors, input_files, model, thinning)
        self.remote_process = Process(target=self.remote_chain)
        self.current_theta = None
        self.current_prior = None
        self.current_likelihood = None
        self.current_posterior = None

        self.remote_process.start()

    def remote_start(self, temperature):
        self.remote_chain.task_queue.put(temperature)

    def remote_complete(self):
        self.current_theta, self.current_prior, self.current_likelihood, self.current_posterior = \
            self.remote_chain.response_queue.get()

    def remote_terminate(self):
        self.remote_process.terminate()


class MC3(object):
    """A Metropolis-Coupled MCMC."""

    def __init__(self, priors, input_files, model, no_chains, thinning, switching, temperature_scale):

        self.no_chains = no_chains
        self.chains = [RemoteMCMCProxy(priors, input_files, model, switching) for _ in xrange(no_chains)]
        self.thinning = thinning
        self.switching = switching
        self.temperature_scale = temperature_scale

    def chain_temperature(self, chain_no):
        if chain_no == 0:
            return 1.0
        else:
            return chain_no * self.temperature_scale

    def sample(self):
        """Sample after running "thinning" steps with a proposal for switching chains at each
        "switching" step."""

        for _ in xrange(int(float(self.thinning) / self.switching)):

            for chain_no, chain in enumerate(self.chains):
                chain.remote_start(self.chain_temperature(chain_no))
            for chain in self.chains:
                chain.remote_complete()

            i = randint(0, self.no_chains)
            j = randint(0, self.no_chains)

            if i != j:
                temperature_i = self.chain_temperature(i)
                temperature_j = self.chain_temperature(j)
                chain_i, chain_j = self.chains[i], self.chains[j]
                current = chain_i.current_posterior / temperature_i + chain_j.current_posterior / temperature_j
                new = chain_j.current_posterior / temperature_i + chain_i.current_posterior / temperature_j
                if new > current or random() < exp(new - current):
                    self.chains[i], self.chains[j] = self.chains[j], self.chains[i]

        return self.chains[0].current_theta, self.chains[0].current_prior, \
               self.chains[0].current_likelihood, self.chains[0].current_posterior

    def terminate(self):
        for chain in self.chains:
            chain.remote_terminate()
