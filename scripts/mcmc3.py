"""
Module for generic MCMC code but with some extensions to the original MCMC.

"""
from pyZipHMM import Forwarder
from likelihood2 import Likelihood

from scipy.stats import norm, expon
from numpy.random import random, randint,seed
from math import log, exp,sqrt
from numpy import array, sum, prod
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
        
class UniformPrior(object):
    
    def __init__(self, init, until, proposal_sd=None, a=0.0):
        self.until=until
        self.init=init
        self.a=a
        if proposal_sd is not None:
            self.proposal_sd = proposal_sd
        else:
            self.proposal_sd = 0.1
            
    def pdf(self, x):
        if x<=self.until and x>=self.a:
            return 1.0/(self.until-self.a)
        else:
            return 0

    def sample(self):
        return self.a+random()*(self.until-self.a)

    def proposal(self, x):
        log_step = norm.rvs(loc=log(x), scale=self.proposal_sd, size=1)[0]
        return exp(log_step)
    
    def log_proposal_step(self):
        return norm.rvs(loc=0, scale=self.proposal_sd, size=1)[0]
    
class UniformPriorWithGoalPosts(object):
    
    def __init__(self, init, until, proposal_sd=None, a=0.0):
        self.until=until
        self.init=init
        self.a=a
        if proposal_sd is not None:
            self.proposal_sd = proposal_sd
        else:
            self.proposal_sd = 0.1
            
    def pdf(self, x):
        if x<self.until and x>self.a:
            return 1.0/(self.until-self.a)*0.5
        elif x==self.until or x==self.a:
            return 0.25
        else:
            return 0

    def sample(self):
        proposal=(1.3333*random()-0.3333)*(self.until-self.a)
        if proposal<self.a:
            return self.a
        elif proposal>self.until:
            return self.until
        return proposal

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
    def __init__(self, priors, log_likelihood, thinning, transferminator=None, mixtureWithScew=0 , mixtureWithSwitch=0, switcher=None, startVal=None, multiple_try=False, printFrequency=0):
        self.priors = priors
        self.log_likelihood = log_likelihood
        self.thinning = thinning
        self.transform = transferminator
        self.latest_squaredJumpSize=0.0    
        if startVal is None:
            self.current_theta = array([pi.sample() for pi in self.priors])
            if self.transform is not None:
                self.transform.first_transform(self.current_theta)
                self.current_theta=array(self.transform.after_transform([0]*len(self.current_theta)))
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
        self.steps=0
        self.printFrequency=printFrequency
            

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
        self.latest_squaredJumpSize=3.217#self.calcSquaredJump(self.current_theta, new_theta)
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
            
            
#     def calcSquaredJump(self, thBefore, thAfter):
#         tij=gamma_break_points(20,beta1=0.001,alpha=2,beta2=0.001333333)
#         tij.append(0.0)
#         def calcForOne(th):
#             prob11=0
#             prob22=0
#             res=[0]*20
#             for k in range(0,20):
#                 res[k]=th[int(k/5)]*prob11+th[4+int(k/5)]*prob22
#                 mt21=(tij[k+1]-tij[k])*th[int(k/5)+3*4]
#                 mt12=(tij[k+1]-tij[k])*th[int(k/5)+2*4]
#                 prob11=mt21+prob11*(1-mt21)
#                 prob22=mt12+prob22*(1-mt12)
#             return res
#         a=calcForOne(thBefore)
#         b=calcForOne(thAfter)
#         return sum([(i-j)**2 for i,j in zip(a,b)])
            
            
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
        self.latest_squaredJumpSize=3.217#self.calcSquaredJump(self.current_theta, new_theta)

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
        self.latest_squaredJumpSize=3.217#self.calcSquaredJump(self.current_theta, new_theta)
        
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
        if self.printFrequency:
            if self.steps%self.printFrequency==0:
                if not isinstance(temperature,float):
                    tempt=temperature[0]
                else:
                    tempt=temperature
                if tempt==1.0:
                    print self.current_transitionMatrix
                    print "\n" + "----------temperature=1 transitionMatrix----------"
                    for mat in self.current_transitionMatrix:
                        print printPyZipHMM(mat)
                    print "\n" + "----------temperature=1 initialDistribution----------"
                    for mat in self.current_initialDistribution:
                        print printPyZipHMM(mat)
                    print "\n"
        self.steps+=1
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
        self.mc3_of_mcgs=kwargs.get("mc3_of_mcgs", 1)
        self.printFrequency=kwargs.get("printFrequency", 0)
#        self.fixed_time_pointer=kwargs.get("fixed_time_pointer", None)
        print self.startVal
        self.priors = priors
        self.log_likelihood=likelihood
        self.thinning = thinning
        self.chain = None
        self.task_queue = Queue()
        self.response_queue = Queue()
        print "made remote mcmc"


    def _set_chain(self):
        if self.mc3_of_mcgs>1:
            self.chain=MCG(priors=self.priors, probs=self.mc3_of_mcgs, log_likelihood=self.log_likelihood, thinning=self.thinning,
                           transferminator=self.transferminator, mixtureWithScew=self.mixtureWithScew,
                           mixtureWithSwitch=self.mixtureWithSwitch, switcher=self.switcher,
                           startVal=self.startVal, multiple_try=self.multiple_try, printFrequency=self.printFrequency)
        else:
            self.chain = MCMC(priors=self.priors, log_likelihood=self.log_likelihood, thinning=self.thinning,
                           transferminator=self.transferminator, mixtureWithScew=self.mixtureWithScew,
                           mixtureWithSwitch=self.mixtureWithSwitch, switcher=self.switcher,
                           startVal=self.startVal, multiple_try=self.multiple_try, printFrequency=self.printFrequency)

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

    def __init__(self, priors, log_likelihood, accept_jump, flip_suggestions, sort, chain_structure, thinning, switching, temperature_scale=1,fixedMax=None,covariance_matrix_sharing=None, **kwargs):
        no_chains=len(chain_structure)
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
        if not "printFrequency" in kwargs:
            kwargs["printFrequency"]=0
#        if not "fixed_time_pointer" in kwargs:
#            kwargs["fixed_timer_pointer"]=None
        self.no_chains = no_chains
        self.accept_jump=accept_jump
        self.flip_suggestions=flip_suggestions
        self.sort=sort
        self.covariance_matrix_sharing=covariance_matrix_sharing
        print kwargs
        self.chains = [RemoteMCMCProxy(priors, log_likelihood, switching, transferminator=kwargs["transferminator"][n], 
                                       mixtureWithScew=kwargs["mixtureWithScew"], mixtureWithSwitch=kwargs["mixtureWithSwitch"], 
                                       switcher=kwargs["switcher"], startVal=kwargs["startVal"], mc3_of_mcgs=x,
                                       printFrequency=kwargs["printFrequency"]) for n,x in enumerate(chain_structure)]
        self.thinning = thinning
        self.switching = switching
        if fixedMax is None:
            self.temperature_scale = [1.0+temperature_scale*n for n in range(no_chains)]
            self.fixedMax=False
            self.tempUpdater=self.updateTemperature
        else:
            self.initializeFixedMax(fixedMax)
            self.fixedMax=fixedMax
            self.tempUpdater=self.uniformlyUpdateTemperature
        print self.temperature_scale
        self.count=1
        self.countTotal=1
        self.alpha=0.5
        self.noSwitchInRow=0
        self.orgChains=self.no_chains
        
        #this is for storage of the nonswap-adaption parameters
        self.nsap=[None]*self.no_chains

    def chain_temperature(self, chain_no):
        return self.temperature_scale[chain_no]
        
    def chainValues(self, temp):
        return self.chains[temp].current_theta, self.chains[temp].current_prior, self.chains[temp].current_likelihood, self.chains[temp].current_posterior, self.chains[temp].accepts,\
    self.chains[temp].rejections, self.nsap[temp], self.chains[temp].swapAdapParam, self.chains[temp].latest_squaredJumpSize

    def updateTemperature(self, index, acceptProb):
        gamma=0.9/self.count**self.alpha
        tempChange = exp(gamma*(acceptProb-self.accept_jump))
        diffs=[j-i for i,j in zip(self.temperature_scale[:-1],self.temperature_scale[1:])]
        diffs[index]*=tempChange
        self.temperature_scale=[1.0]*self.no_chains
        for n in range(self.no_chains-1):
            self.temperature_scale[n+1]=self.temperature_scale[n]+diffs[n]
        #The very hot chains produces very unlikely events, which 
        #sometimes will throw an AssertionError deeper into the code. 
        #The assertion will produce a warning and the step ignored, but
        #we don't want too many of those in order to keep ergodicity. 
        if self.temperature_scale[self.no_chains-1]>2000 and self.no_chains-2==index:
            self.no_chains-=1
            print "\n"+"Dropped temperature "+str(self.temperature_scale[self.no_chains])+" for good." +"\n"
    
    def initializeFixedMax(self,fixedMax):
        #choosing a geometric as prior.
        geom_param=fixedMax**(1.0/(self.no_chains-1))
        print geom_param
        self.temperature_scale=[geom_param**i for i in range(self.no_chains)]
        self.sumAll=1
             
    def uniformlyUpdateTemperature(self, index,acceptProb):
        self.sumAll+=acceptProb
        self.countTotal+=1
        gamma=0.9/self.count**self.alpha
        tempChange = exp(gamma*(acceptProb-float(self.sumAll)/self.countTotal))
        diffs=[j-i for i,j in zip(self.temperature_scale[:-1],self.temperature_scale[1:])]
        diffs[index]*=tempChange
        normalizer=self.fixedMax/(sum(diffs)+1.0)
        self.temperature_scale=[1.0]*self.no_chains
        for n in range(self.no_chains-1):
            self.temperature_scale[n+1]=self.temperature_scale[n]+diffs[n]*normalizer
        if(random()<0.01):
            print "temperatureUpdate probability "+ str(float(self.sumAll)/self.countTotal)
        
    
    def sample(self):
        """Sample after running "thinning" steps with a proposal for switching chains at each
        "switching" step."""

        flips=""
        for index in xrange(self.thinning):
            
            for chain_no in range(self.no_chains):
                self.chains[chain_no].remote_start((self.chain_temperature(chain_no), self.nsap[chain_no]))
            for chain_no in range(self.no_chains):
                self.chains[chain_no].remote_complete()
                self.nsap[chain_no]=deepcopy(self.chains[chain_no].nonSwapAdapParam)
            if self.covariance_matrix_sharing is not None: ##This is when we have algorithm 4 scaling. So we take a weighted mean for each chain. We do this so that the probability of "killing" parameters is small
                for chain_no in range(self.no_chains):
                    matrixsum=sum(self.nsap[n][1]*max(0,1-self.covariance_matrix_sharing*abs(n-chain_no)) for n in xrange(self.no_chains))
                    matrixdivider=sum(max(0,1-self.covariance_matrix_sharing*abs(n-chain_no)) for n in xrange(self.no_chains))
                    self.nsap[chain_no][1]=matrixsum/matrixdivider               
            
            if self.sort:
                orderAndSort=sorted(enumerate(self.chains), key=lambda ch: -ch[1].current_posterior)
                order,self.chains=map(list, zip(*orderAndSort))
                #sorting all chains to make them very close to the mode of the posterior.
                
                
            for k in range(self.flip_suggestions):
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
                    if self.sort:
                        order[i],order[j]=order[j],order[i] #checking if a change has been made
                    flips+=str(index)+":"+str(i)+"-"+str(j)+","
                if k==0 and not self.sort: #when you accept a transition that has probability less than 1, the backswitch has probability more than one. Therefore, we do this in order not to explode the temperature adaption.
                    self.count+=1
                    self.tempUpdater(i,acceptProb)
                    if i==0:
                        print "acceptance probability of jump between chains "+ str(acceptProb)
            if i==0:
                print self.temperature_scale
            if self.sort:
                if all(x<y for x, y in zip(order, order[1:])):
                    self.noSwitchInRow+=1
                else:
#                     if random()<0.01:
                    print str(self.noSwitchInRow)+" ids and now "+ str(order)
                    self.noSwitchInRow=0
                self.count+=1   
                for ro in range(self.no_chains-1):
                    if max(order[:ro+1])>ro or min(order[(ro+1):])<ro+1: # if a passage has happened.
                        self.tempUpdater(ro, 1.0)
                    else: 
                        self.tempUpdater(ro, 0)
                

        return tuple( self.chainValues(t) for t in range(self.orgChains))+(flips,)

    def terminate(self):
        for chain in self.chains:
            chain.remote_terminate()
    
    
class MCG(object):
    
    def __init__(self, priors, log_likelihood, thinning=1,probs=1, transferminator=None, mixtureWithScew=0 , mixtureWithSwitch=0, switcher=None, startVal=None, multiple_try=False, mcg_flip_suggestions=40, printFrequency=0, fixed_time_pointer=None):
        self.mcg_flip_suggestions=mcg_flip_suggestions
        self.priors = priors
        self.thinning = thinning  #thinning is obsolete
        if startVal is None:
            self.current_theta = array([pi.sample() for pi in self.priors])
        else:
            self.current_theta=array(startVal)
        self.transferminator=transferminator
        self.chains = [RemoteMCMCProxy(priors, log_likelihood,1, transferminator=transferminator, startVal=self.current_theta,multiple_try=True, printFrequency=printFrequency) for _ in xrange(probs)]
        self.temp=1.0
        self.probs=probs #number of proposals.
        self.glob_scale=transferminator.getAdapParam(all=True)
        self.current_theta=None
        self.current_posterior=None
        self.current_prior=0
        self.current_likelihood=0
        self.pool=Pool(processes=self.probs)


    
    def sample(self,temperatureBundle=(1.0,'keep')):
        if not temperatureBundle[1]=='keep' and temperatureBundle[1] is not None:
            self.glob_scale=(temperatureBundle[1],self.glob_scale[1])
        temperature=temperatureBundle[0]

        posteriors=[0]*self.probs
        thetas=[0]*self.probs
        standardizedLogJumps=[0]*self.probs
        for chain_no in range(self.probs):
            self.chains[chain_no].remote_start((temperature, self.glob_scale,self.current_posterior,self.current_theta))
        for chain_no in range(self.probs):
            self.chains[chain_no].remote_complete()
            posteriors[chain_no]=self.chains[chain_no].current_posterior
            thetas[chain_no]=self.chains[chain_no].current_theta
            
            ###This is not the best notation. accepts variable doesn't contain accept probability
            #accept variable contains the logjumps
            if self.transferminator.stationaryPossible():
                standardizedLogJumps[chain_no]=self.chains[chain_no].accepts.tolist()
        if self.current_posterior is None: #in the beginnning we just assign this to the most probable beginning state
            max_index, max_value = max(enumerate(posteriors), key=operator.itemgetter(1))
            self.current_posterior=max_value
            self.current_theta=thetas[max_index]
        
        posteriors.append(self.current_posterior)
        thetas.append(self.current_theta)
        pies=[0]*(self.probs+1)
        for i in range(self.probs):
            pies[i]=exp(posteriors[i]/temperature-max(posteriors)/temperature)
        pies[self.probs]=exp(self.current_posterior/temperature-max(posteriors)/temperature)
        
        
        #We now calculate K if the transformation method allows for it, so that we make a good suggestion.
        if self.transferminator.stationaryPossible(): 
            Ks=self.pool.map(Kcalculator, zip(*(range(self.probs+1),[standardizedLogJumps]*(self.probs+1))))
            stationary=[k*p for k,p in zip(Ks,pies)]
            if sum(stationary)==0:
                print "The sum of all choices was 0."
                print "Ks="+str(Ks)
                print "pies="+str(pies)
                return self.current_theta, self.current_prior,self.current_likelihood,self.current_posterior,\
                    0, 1,self.glob_scale[0],self.glob_scale[1],0
            stationary=[s/sum(stationary) for s in stationary]
            PathChoice=random_distr(zip(range(self.probs+1),stationary))
        else:
            currentPath=self.probs #this is the index of the previous state.
            averageAlpha=0
            for i in range(self.mcg_flip_suggestions):
                suggst=currentPath
                while suggst==currentPath:
                    suggst=randint(0,self.probs)
                if posteriors[suggst]>posteriors[currentPath]:
                    alpha=1.0
                else:
                    alpha=exp(posteriors[suggst]/temperature-posteriors[currentPath]/temperature)
                if random()< alpha:
                    currentPath=suggst
                averageAlpha+=alpha
            averageAlpha/=self.mcg_flip_suggestions
            PathChoice=currentPath
        print "averageAlpha="+str(averageAlpha)
        #we translate so we always start at 0.
        

        if PathChoice<self.probs:
            self.current_theta=thetas[PathChoice]
            self.current_posterior=posteriors[PathChoice]
            self.current_likelihood=self.chains[PathChoice].current_likelihood
            self.current_prior=self.chains[PathChoice].current_prior
        

        #making the adaption
        self.transferminator.setAdapParam(self.glob_scale)
        if self.transferminator.stationaryPossible():
            self.transferminator.first=[0]*len(self.current_theta)
            max_index, _= max(enumerate(stationary[:-1]), key=operator.itemgetter(1))
            self.transferminator.jumps=standardizedLogJumps[max_index]
            self.glob_scale=self.transferminator.update_alpha(PathChoice<self.probs ,1-stationary[-1])
        else:
            self.transferminator.first=thetas[self.probs]
            self.transferminator.second=thetas[PathChoice]
            print "glob_scale after="+str((self.glob_scale[0],self.glob_scale[1][:3]))
            print "steps="+str(self.transferminator.count)+"/"+str(self.transferminator.timeTillStart)
            print "proportions="+str(self.transferminator.proportions)
            self.glob_scale=self.transferminator.update_alpha(PathChoice<self.probs, averageAlpha)
            print "glob_scale after="+str((self.glob_scale[0],self.glob_scale[1][:3]))
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
