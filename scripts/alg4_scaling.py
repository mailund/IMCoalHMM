'''
Created on Oct 27, 2014

@author: svendvn
'''

from math import log, exp, sqrt
from numpy.random import multivariate_normal, normal
from numpy import array,matrix,identity, outer
from numpy.linalg import eig
from random import randrange

class AM4_scaling(object):
    '''
    classdocs
    '''


    def __init__(self,startVal=[1.0]*17, params=[0.5,1], sigmaStart=None, alphaDesired=0.234):
        '''
        Constructor. theta is the factor that is multiplied on all proposals. It is updated throughout so input to the constructor is only a 
        starting value. 
        alpha is the power in the updating rule theta_{count+1}=max(0.0001,theta_{count}+1/count^alpha*(1_{accept}(true)*2-1))
        '''
        self.theta=startVal[0]
        size=len(startVal)
        if sigmaStart is None:
            self.sigma=matrix(identity(size))*0.1
        else:
            self.sigma=sigmaStart
        self.count=1
        self.alpha=params[0]
        self.multip=params[1]
        self.mean=array([0]*size)
        self.alphaDesired=alphaDesired
        
    
    def setAdapParam(self, val):
        if len(val)>1:
            self.sigma=val[1]
            self.theta=val[0][0]
        else:
            self.theta=val[0]
        
    def getAdapParam(self,all=False):
        if all:
            return [self.theta], self.sigma
        return [self.theta]
    
    def getStandardizedLogJumps(self):
        return None #this vector is not well defined here/useful here. 
    
    def first_transform(self, params): 
        '''
        We record the mean. 
        '''
        self.first=map(log,params)
    
     
    def after_transform(self, adds):   
        '''
        Here we don't use the parameters, _, already simulated, we make new ones. 
        This doesn't change the prior
        '''
        if self.count>100:
            self.second=multivariate_normal(self.first, self.theta*self.sigma)
        else:
            self.second=[(f+a*sqrt(self.theta)) for f,a in zip(self.first,adds)]
        print self.second
        return map(exp,self.second)
        
    def update_alpha(self, accept, alphaXY):
        '''
        This updates alpha based on whether or not 'accept' is true or false. 
        We want: accept=True <=> the proposal based on alpha was accepted. 
        '''
        if accept:
            x=self.second
        else:
            x=self.first
        
        gamma=self.multip/self.count**self.alpha

        self.sigma += gamma*(outer(x-self.mean,x-self.mean)-self.sigma)
        self.mean += gamma*(x-self.mean)
        self.theta *= exp(gamma*(alphaXY-self.alphaDesired))
        self.count += 1
        if self.count==101:#we try to normalize self.theta
            normalizer=len(self.first)*0.1/sum(eig(self.sigma)[0])
            self.sigma*=normalizer
        if self.count%20==0:
            print self.sigma
            print self.theta
        
        
        return [self.theta],[0]
        
        