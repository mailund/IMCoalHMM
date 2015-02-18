'''
Created on Oct 22, 2014

@author: svendvn
'''

from math import log, exp,sqrt

class Global_scaling(object):
    '''
    classdocs
    '''

    NONSWAP_PARAM=("theta",)
    SWAP_PARAM=()
    
    
    def __init__(self,theta=1.0, params=[0.5,10], alphaDesired=0.234):
        '''
        Constructor. theta is the factor that is multiplied on all proposals. It is updated throughout so input to the constructor is only a 
        starting value. 
        alpha is the power in the updating rule theta_{count+1}=max(0.0001,theta_{count}+1/count^alpha*(1_{accept}(true)*2-1))
        '''
        self.theta=theta
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
        self.first=[log(x) for x in params]
    
     
    def after_transform(self, adds):   
        '''
        this takes a vector from scaled space and transforms it back
        '''
        self.second=[f+a*sqrt(self.theta) for f,a in zip(self.first,adds)]
        self.jumps=adds
        return [exp(s) for s in self.second]
        
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
        
        