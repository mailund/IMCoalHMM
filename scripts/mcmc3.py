"""
Module for generic MCMC code but with some extensions to the original MCMC.

"""
from pyZipHMM import Forwarder
from likelihood2 import Likelihood

from scipy.stats import norm, expon
from numpy.random import random, randint,seed
from math import log, exp,sqrt
from numpy import array, sum,prod
from break_points2 import gamma_break_points
from copy import deepcopy
import operator

from multiprocessing import Process, Queue,Pool

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
    
    def log_proposal_step(self):
        return norm.rvs(loc=0, scale=self.proposal_sd, size=1)[0]
        


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
    
    def log_proposal_step(self):
        return norm.rvs(loc=0, scale=self.proposal_sd, size=1)[0]
        


class MCMC(object):
    def __init__(self, priors, log_likelihood, thinning, transferminator=None, mixtureWithScew=0 , mixtureWithSwitch=0, switcher=None, startVal=None, multiple_try=False):
        self.priors = priors
        self.log_likelihood = log_likelihood
        self.thinning = thinning
        self.transform = transferminator
        self.latest_squaredJumpSize=0.0    
        if startVal is None:
            self.current_theta = array([pi.sample() for pi in self.priors])
        else:
            self.current_theta=array(startVal)
            
        self.rejectedSwitches={}
        self.acceptedSwitches={}
        self.multiple_try=multiple_try
        
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
        self.latest_initialDistribution=forget2
        self.latest_suggestedTheta=[0]*len(self.current_theta)
        self.swapAdapParam=[1]
        self.nonSwapAdapParam=[1]
            

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
        self.latest_squaredJumpSize=self.calcSquaredJump(self.current_theta, new_theta)
        new_prior = self.log_prior(new_theta)
        new_transitionMatrix, self.latest_initialDistribution, new_log_likelihood = self.log_likelihood(new_theta)
        new_posterior = new_prior + new_log_likelihood

        
        if new_posterior > self.current_posterior or \
                        random() < exp(new_posterior / temperature - self.current_posterior / temperature):
            self.current_theta = new_theta
            self.current_prior = new_prior
            self.current_transitionMatrix, self.current_initialDistribution, self.current_likelihood = new_transitionMatrix, self.latest_initialDistribution, new_log_likelihood
            self.current_posterior = new_posterior
            self.accepts+=1
        else:
            self.rejections+=1
        self.latest_suggestedTheta=new_theta
            
            
    def calcSquaredJump(self, thBefore, thAfter):
        tij=gamma_break_points(20,beta1=0.001,alpha=2,beta2=0.001333333)
        tij.append(0.0)
        def calcForOne(th):
            prob11=0
            prob22=0
            res=[0]*20
            for k in range(0,20):
                res[k]=th[int(k/5)]*prob11+th[4+int(k/5)]*prob22
                mt21=(tij[k+1]-tij[k])*th[int(k/5)+3*4]
                mt12=(tij[k+1]-tij[k])*th[int(k/5)+2*4]
                prob11=mt21+prob11*(1-mt21)
                prob22=mt12+prob22*(1-mt12)
            return res
        a=calcForOne(thBefore)
        b=calcForOne(thAfter)
        return sum([(i-j)**2 for i,j in zip(a,b)])
            
            
            
        
            
    def ScewStep(self, temperature=1.0):
        self.transform.first_transform(self.current_theta)
        new_thetaTmp = array([self.priors[i].log_proposal_step() for i in xrange(len(self.current_theta))])
        new_theta= array(self.transform.after_transform(new_thetaTmp))
        new_prior = self.log_prior(new_theta)
        try:
            new_transitionMatrix, self.latest_initialDistribution, new_log_likelihood = self.log_likelihood(new_theta)    
        except AssertionError as e:
            print "The model has tried to move outside of its stabile values at temperature "+str(temperature)+ " "+str(e)[0:10]+"..."
            return
        new_posterior = new_prior + new_log_likelihood
        self.latest_squaredJumpSize=self.calcSquaredJump(self.current_theta, new_theta)

        if new_posterior > self.current_posterior:
            alpha=1
        else:
            alpha=min(1, exp(new_posterior / temperature - self.current_posterior / temperature))
        if new_posterior > self.current_posterior or \
                        random() < exp(new_posterior / temperature - self.current_posterior / temperature):
            self.current_theta = new_theta
            self.current_prior = new_prior
            self.current_transitionMatrix, self.current_initialDistribution, self.current_likelihood = new_transitionMatrix, self.latest_initialDistribution, new_log_likelihood
            self.current_posterior = new_posterior
            self.accepts+=1
            self.nonSwapAdapParam,self.swapAdapParam=self.transform.update_alpha(True, alpha)
        else: 
            self.rejections+=1
            self.nonSwapAdapParam,self.swapAdapParam=self.transform.update_alpha(False,alpha)
        self.latest_suggestedTheta=new_theta
            
    def switchStep(self,temperature):
        new_theta,whatSwitch=self.switcher(self.current_theta)
        new_prior = self.log_prior(new_theta)
        new_transitionMatrix, self.latest_initialDistribution, new_log_likelihood = self.log_likelihood(new_theta)
        new_posterior = new_prior + new_log_likelihood
        self.latest_squaredJumpSize=self.calcSquaredJump(self.current_theta, new_theta)
        
        if new_posterior > self.current_posterior or \
                        random() < exp(new_posterior / temperature - self.current_posterior / temperature):
            self.current_theta = new_theta
            self.current_prior = new_prior
            self.current_transitionMatrix, self.current_initialDistribution, self.current_likelihood = new_transitionMatrix, self.latest_initialDistribution, new_log_likelihood
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
        
    
    def PropstepScew(self,bundleOfInfo):
        self.transform.setAdapParam(bundleOfInfo[1])
        if len(bundleOfInfo)>=3:
            self.setSample(bundleOfInfo[2],bundleOfInfo[3])
            
        self.transform.first_transform(self.current_theta)
        new_thetaTmp = array([self.priors[i].log_proposal_step() for i in xrange(len(self.current_theta))])
        new_theta= array(self.transform.after_transform(new_thetaTmp))
        
        new_prior = self.log_prior(new_theta)
        _, self.latest_initialDistribution, new_log_likelihood = self.log_likelihood(new_theta)
        new_posterior=new_log_likelihood+new_prior
        jumps=array(self.transform.getStandardizedLogJumps())
        return new_theta,new_prior,new_log_likelihood, new_posterior,jumps,0, "doesnt matter",'doesnt matter',0
    
    def __call__(self):
        propPar=self.current_theta
        new_theta = array([self.priors[i].proposal(propPar[i]) for i in xrange(len(self.current_theta))])
        new_prior = self.log_prior(new_theta)
        _, self.latest_initialDistribution, new_log_likelihood = self.log_likelihood(new_theta)
        new_posterior=new_log_likelihood+new_prior
        return new_posterior, new_theta
    
    def setSample(self,post,thet):
        if post is not None: #the first time we run this we don't have a posterior, so we should not 
            self.current_posterior=post
            self.current_theta=thet
    
    
    def getSwitchStatistics(self):
        return self.acceptedSwitches, self.rejectedSwitches

    def sample(self, temperature=1.0):
        if self.multiple_try:
            bundleOfInfo=temperature
            if self.transform is not None:
                return self.PropstepScew(bundleOfInfo)
            else:
                return self.Propstep(bundleOfInfo) #doesnot exist
        elif not isinstance(temperature,float):#in some schemes we pass on more information
            bundleOfInfo=temperature
            if temperature[1] is not None and self.transform is not None:
                self.transform.setAdapParam(temperature[1])
            temperature=temperature[0]
        self.accepts=0
        self.rejections=0
        for _ in xrange(self.thinning):
            if(randint(0,self.mixtureWithSwitch+1)==1):
                self.switchStep(temperature)
            elif(self.mixtureWithScew>0):
                self.ScewStep(temperature)
            else:
                self.step(temperature)
        return self.current_theta, self.current_prior, self.current_likelihood, self.current_posterior, self.accepts, self.rejections, self.nonSwapAdapParam,self.swapAdapParam, self.latest_squaredJumpSize
    
    
    def sampleRecordInitialDistributionJumps(self, temperature=1.0):
        self.accepts=0
        self.rejections=0
        for _ in xrange(self.thinning):
            if(randint(0,self.mixtureWithSwitch+1)==1):
                self.switchStep(temperature)
            elif(self.mixtureWithScew>0):
                self.ScewStep(temperature)
            else:
                self.step(temperature)
        return self.current_theta, self.current_prior, self.current_likelihood, self.current_posterior, self.accepts, self.rejections, self.nonSwapAdapParam,self.swapAdapParam, self.latest_squaredJumpSize, self.latest_suggestedTheta,self.latest_initialDistribution
    
    
    
    def orderSample(self, params):
        new_prior = self.log_prior(params)
        self.current_transitionMatrix, self.current_initialDistribution, new_log_likelihood = self.log_likelihood(params)
        new_posterior = new_prior + new_log_likelihood
        acc=random() < exp(new_posterior-self.current_posterior)
        return params, new_prior, new_log_likelihood, new_posterior,acc,1-acc, self.nonSwapAdapParam,self.swapAdapParam 
        
    def transformToIdef(self, inarray):
        return inarray
    
    def transformFromIdef(self,inarray):
        return inarray
    
    def getNonSwapAdapParam(self):
        return self.nonSwapAdapParam

    def setNonSwapAdapParam(self, value):
        self.transform.setAdapParam(value)


class RemoteMCMC(object):
    """ MCMC that is designed to run in another process for parallel execution.
    """

    def __init__(self, priors, likelihood, thinning, **kwargs):
        seed()
        self.transferminator=kwargs.get("transferminator",None)
        self.mixtureWithScew=kwargs.get("mixtureWithScew",0)
        self.mixtureWithSwitch=kwargs.get("mixtureWithSwitch",0)
        self.switcher=kwargs.get("switcher",None)
        self.startVal=kwargs.get("startVal",None)
        self.multiple_try=kwargs.get("multiple_try",False)
        print self.startVal
        self.priors = priors
        self.log_likelihood=likelihood
        self.thinning = thinning
        self.chain = None
        self.task_queue = Queue()
        self.response_queue = Queue()
        print "made remote mcmc"


    def _set_chain(self):

        self.chain = MCMC(priors=self.priors, log_likelihood=self.log_likelihood, thinning=self.thinning,
                           transferminator=self.transferminator, mixtureWithScew=self.mixtureWithScew,
                           mixtureWithSwitch=self.mixtureWithSwitch, switcher=self.switcher,
                           startVal=self.startVal, multiple_try=self.multiple_try)

    def __call__(self):
        print "called remote mcmc"
        self._set_chain()
        while True:
            temperature = self.task_queue.get()
            self.response_queue.put(self.chain.sample(temperature))

class RemoteMCMCProxy(object):
    """Local handle to a remote MCMC object."""

    def __init__(self, priors, likelihood, thinning, **kwargs):
        self.remote_chain = RemoteMCMC(priors, likelihood, thinning, **kwargs)
        self.remote_process = Process(target=self.remote_chain)
        #self.remote_process=Process(target=remoteMCMC, args=(priors, models, input_files, thinning))
        self.current_theta = None
        self.current_prior = None
        self.current_likelihood = None
        self.current_posterior = None

        self.remote_process.start()

    def remote_start(self, bundleOfInfo):
        self.remote_chain.task_queue.put(bundleOfInfo)
    
    def setNonSwapAdapParam(self,val):
        self.remote_chain.setNonSwapAdapParam(val)

    def remote_complete(self):
        self.current_theta, self.current_prior, self.current_likelihood, self.current_posterior, self.accepts, self.rejections, self.nonSwapAdapParam,self.swapAdapParam, self.latest_squaredJumpSize = \
            self.remote_chain.response_queue.get()

    def remote_terminate(self):
        self.remote_process.terminate()
       


class MC3(object):
    """A Metropolis-Coupled MCMC."""

    def __init__(self, priors, likelihood, no_chains, thinning, switching, temperature_scale=1, **kwargs):
        if not "transferminator" in kwargs:
            kwargs["transferminator"]=[None]*no_chains
        if not "mixtureWithScew" in kwargs:
            kwargs["mixtureWithScew"]=0
        if not "mixtureWithSwitch" in kwargs:
            kwargs["mixtureWithSwitch"]=0
        if not "switcher" in kwargs:
            kwargs["switcher"]=None
        if not "startVal" in kwargs:
            kwargs["startVal"]=None
        self.no_chains = no_chains
        print kwargs
        self.chains = [RemoteMCMCProxy(priors, likelihood, switching, transferminator=kwargs["transferminator"][i], 
                                       mixtureWithScew=kwargs["mixtureWithScew"], mixtureWithSwitch=kwargs["mixtureWithSwitch"], 
                                       switcher=kwargs["switcher"], startVal=kwargs["startVal"]) for i in xrange(no_chains)]
        self.thinning = thinning
        self.switching = switching
        self.temperature_scale = [1.0+temperature_scale*n for n in range(no_chains)]
        print self.temperature_scale
        self.count=1
        self.alpha=0.5
        self.orgChains=self.no_chains
        
        #this is for storage of the nonswap-adaption parameters
        self.nsap=[None]*self.no_chains

    def chain_temperature(self, chain_no):
        return self.temperature_scale[chain_no]
        
    def chainValues(self, temp):
        return self.chains[temp].current_theta, self.chains[temp].current_prior, self.chains[temp].current_likelihood, self.chains[temp].current_posterior, self.chains[temp].accepts,\
    self.chains[temp].rejections, self.nsap[temp], self.chains[temp].swapAdapParam, self.chains[temp].latest_squaredJumpSize

    def updateTemperature(self, index, acceptProb):
        self.count+=1
        gamma=0.9/self.count**self.alpha
        tempChange = exp(gamma*(acceptProb-0.234))
        diffs=[j-i for i,j in zip(self.temperature_scale[:-1],self.temperature_scale[1:])]
        diffs[index]*=tempChange
        self.temperature_scale=[1.0]*self.no_chains
        for n in range(self.no_chains-1):
            self.temperature_scale[n+1]=self.temperature_scale[n]+diffs[n]
            
        #The very hot chains produces very unlikely events, which 
        #sometimes will throw an AssertionError deeper into the code. 
        #The assertion will produce a warning and the step ignored, but
        #we don't want too many of those in order to keep ergodicity. 
        if self.temperature_scale[self.no_chains-1]>2000:
            self.no_chains-=1
            print "\n"+"Dropped temperature "+str(self.temperature_scale[self.no_chains])+" for good." +"\n"
    
    def sample(self):
        """Sample after running "thinning" steps with a proposal for switching chains at each
        "switching" step."""

        flips=""
        for index in xrange(int(float(self.thinning) / self.switching)):
            
            for chain_no in range(self.no_chains):
                self.chains[chain_no].remote_start((self.chain_temperature(chain_no), self.nsap[chain_no]))
            for chain_no in range(self.no_chains):
                self.chains[chain_no].remote_complete()
                self.nsap[chain_no]=deepcopy(self.chains[chain_no].nonSwapAdapParam)
                

            i = randint(0, self.no_chains-1)
            j = i+1

            temperature_i = self.chain_temperature(i)
            temperature_j = self.chain_temperature(j)
            chain_i, chain_j = self.chains[i], self.chains[j]
            current = chain_i.current_posterior / temperature_i + chain_j.current_posterior / temperature_j
            new = chain_j.current_posterior / temperature_i + chain_i.current_posterior / temperature_j
            if new > current:
                acceptProb=1.0
            else:
                acceptProb=exp(new - current)
            if random()<acceptProb:
                self.chains[i], self.chains[j] = self.chains[j], self.chains[i]
                flips+=str(index)+":"+str(i)+"-"+str(j)+","
            self.updateTemperature(i,acceptProb)
            if i==0:
                print self.temperature_scale
                    
            
            
            
        return tuple( self.chainValues(t) for t in range(self.orgChains))+(flips,)

    def terminate(self):
        for chain in self.chains:
            chain.remote_terminate()
    
    
class MCG(object):
    
    def __init__(self, priors, likelihood, thinning=1,startVal=None, probs=1,transferminator=None):
        self.priors = priors
        self.thinning = thinning
        if startVal is None:
            self.current_theta = array([pi.sample() for pi in self.priors])
        else:
            self.current_theta=array(startVal)
        self.transferminator=transferminator
        self.chains = [RemoteMCMCProxy(priors, likelihood,thinning, transferminator=transferminator, startVal=self.current_theta,multiple_try=True) for _ in xrange(probs)]
        self.temp=1.0
        self.probs=probs
        self.glob_scale=transferminator.getAdapParam(all=True)
        self.current_theta=None
        self.current_posterior=None
        self.current_prior=0
        self.current_likelihood=0
        self.pool=Pool(processes=self.probs)


    
    def sample(self):
        posteriors=[0]*self.probs
        thetas=[0]*self.probs
        standardizedLogJumps=[0]*self.probs
        for chain_no in range(self.probs):
            self.chains[chain_no].remote_start((1.0, self.glob_scale,self.current_posterior,self.current_theta))
        for chain_no in range(self.probs):
            self.chains[chain_no].remote_complete()
            posteriors[chain_no]=self.chains[chain_no].current_posterior
            thetas[chain_no]=self.chains[chain_no].current_theta
            
            ###This is not the best notation. accepts is
            #not useful in MCG and we need the standardized log jumps anyway.
            standardizedLogJumps[chain_no]=self.chains[chain_no].accepts.tolist()
        if self.current_posterior is None: #in the beginnning we just assign this to the most probable beginning state
            max_index, max_value = max(enumerate(posteriors), key=operator.itemgetter(1))
            self.current_posterior=max_value
            self.current_theta=thetas[max_index]
        posteriors.append(self.current_posterior)
        thetas.append(self.current_theta)
        pies=[0]*(self.probs+1)
        for i in range(self.probs):
            pies[i]=exp(posteriors[i]-max(posteriors))
        pies[self.probs]=exp(self.current_posterior-max(posteriors))
        
        #We now calculate K, so that we make a good suggestion.
        Ks=self.pool.map(Kcalculator, zip(*(range(self.probs+1),[standardizedLogJumps]*(self.probs+1))))
        stationary=[k*p for k,p in zip(Ks,pies)]
        if sum(stationary)==0:
            print "The sum of all choices was 0."
            print "Ks="+str(Ks)
            print "pies="+str(pies)
            return self.current_theta, self.current_prior,self.current_likelihood,self.current_posterior,\
                0, 1,self.glob_scale[0],self.glob_scale[1],0
        stationary=[s/sum(stationary) for s in stationary]
        
        #we now make a PathChoice with probabilities just calculated
        PathChoice=random_distr(zip(range(self.probs+1),stationary))
        
        #we translate so we always start at 0.
        self.transferminator.first=[0]*len(self.current_theta)
        
        if PathChoice<self.probs:
            self.current_theta=thetas[PathChoice]
            self.current_posterior=posteriors[PathChoice]
            self.current_likelihood=self.chains[PathChoice].current_prior
            self.current_prior=self.chains[PathChoice].current_prior
        
        #some adaption schemes uses the before and after step, and in case we use the one with the highest probability which is not the previous state.
        max_index, _= max(enumerate(stationary[:-1]), key=operator.itemgetter(1))
        self.transferminator.jumps=standardizedLogJumps[max_index]
        
        #making the adaption
        self.transferminator.setAdapParam(self.glob_scale)
        self.glob_scale=self.transferminator.update_alpha(PathChoice<self.probs ,1-stationary[-1])
        
        return self.current_theta, self.current_prior,self.current_likelihood,self.current_posterior,\
                int(PathChoice<self.probs), int(PathChoice==self.probs),self.glob_scale[0],self.glob_scale[1],0
    
    
    def terminate(self):
        for chain in self.chains:
            chain.remote_terminate()
    

def Kcalculator(listOfStandardizedLogsAndIndex):
    """listOfStandardized logs doesn't contain
    the point we are from, which is always [0,...,0]"""
    listOfStandardizedLogs=listOfStandardizedLogsAndIndex[1]
    index=listOfStandardizedLogsAndIndex[0]
    listOfStandardizedLogs.append([0]*len(listOfStandardizedLogs[0]))
    ans=1
    for n,lis in enumerate(listOfStandardizedLogs):
        if not n==index:
            ans=ans*prod(norm.pdf(array(lis)+array(listOfStandardizedLogs[index]),scale=sqrt(0.1)))
    return ans

def random_distr(l):
    r = random()
    s = 0
    for item, prob in l:
        s += prob
        if s >= r:
            return item
    print "rounding error leading to last element being sent or an incorrect distribution has been passed on"
    print "r="+str(r)
    return l[-1][0]