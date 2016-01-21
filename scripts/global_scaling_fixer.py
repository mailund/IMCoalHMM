'''
Created on Oct 22, 2014

@author: svendvn
'''

from math import log, exp,sqrt

class Global_scaling_fixp(object):
    '''
    classdocs
    '''

    NONSWAP_PARAM=("theta",)
    SWAP_PARAM=()
    
    
    def __init__(self,theta=1.0, params=[0.5,10], alphaDesired=0.234, small_parameters_function=None, full_parameters_function=None):
        '''
        Constructor. theta is the factor that is multiplied on all proposals. It is updated throughout so input to the constructor is only a 
        starting value. 
        alpha is the power in the updating rule theta_{count+1}=max(0.0001,theta_{count}+1/count^alpha*(1_{accept}(true)*2-1))
        '''
        self.theta=theta
        self.full_parameters_function=full_parameters_function
        self.small_parameters_function=small_parameters_function
        self.count=1
        self.alpha=params[0]
        self.multip=params[1]
        self.alphaDesired=alphaDesired
    
    def setAdapParam(self, val):
        if len(val)>1: #in someone wants to set both swapAdapParam and nonSwapAdapParam
            self.theta=val[0][0]
        else:
            self.theta=val[0]
        
    def getAdapParam(self,all=True):
        if all:
            return [self.theta],[0]
        return [self.theta]
    
    def getStandardizedLogJumps(self):
        return self.jumps
    
    def stationaryPossible(self):
        return True
    
    def first_transform(self, params):
        '''
        this takes a vector and transforms it into the scaled space
        '''
        sre=self.small_parameters_function(params)
        self.indices0 = [ i for i, x in enumerate(sre) if x == 0 ]
        self.first=map(log,[s if s>0 else 1 for n,s in enumerate(sre)])
        #self.first=[log(x) if x>0 else 3.217 for x in params] #the 3.217 should be erased later. 
    
     
    def after_transform(self, adds):   
        '''
        this takes a vector from scaled space and transforms it back
        '''
        adds=self.small_parameters_function(adds)
        self.second=[f+a*sqrt(self.theta) if n not in self.indices0 else a*sqrt(self.theta) for n,(f,a) in enumerate(zip(self.first,adds))]
        self.jumps=adds
#         res=[exp(s) for s in self.second]
#         tmp=[self.fixes.get(n,r) if n not in self.fixEqual else -1 for n,r in enumerate(res)]
        

        return self.full_parameters_function([exp(s) if n not in self.indices0 else 0 for n,s in enumerate(self.second)])
        
    def update_alpha(self, accept,alphaXY):
        '''
        This updates alpha based on whether or not 'accept' is true or false. 
        We want: accept=True <=> the proposal based on alpha was accepted. 
        
        Other algorithms return two values, that's why we ignore one input. 
        '''
        #0.234 has some special meaning in mcmc acceptance probabilities.
        gamma=self.multip/self.count**self.alpha
        self.theta *= exp(gamma*(alphaXY-self.alphaDesired))
        self.theta=min(10000,self.theta)
        self.count+=1
        return [self.theta],[0]
        
        