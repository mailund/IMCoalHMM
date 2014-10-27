'''
Created on Oct 27, 2014

@author: svendvn
'''

from math import log, exp,sqrt
from numpy.random import multivariate_normal
from numpy import array,matrix,identity, outer

class AM4_scaling(object):
    '''
    classdocs
    '''


    def __init__(self,size , theta=1.0, params=[0.5,1], sigmaStart=None, alphaDesired=0.234):
        '''
        Constructor. theta is the factor that is multiplied on all proposals. It is updated throughout so input to the constructor is only a 
        starting value. 
        alpha is the power in the updating rule theta_{count+1}=max(0.0001,theta_{count}+1/count^alpha*(1_{accept}(true)*2-1))
        '''
        self.theta=theta
        if sigmaStart is None:
            self.sigma=matrix(identity(size))*0.1
        else:
            self.sigma=sigmaStart
        self.count=1
        self.alpha=params[0]
        self.multip=params[1]
        self.mean=array([0]*size)
        self.alphaDesired=alphaDesired
        
        
    
    def first_transform(self, params): 
        '''
        We record the mean. 
        '''
        self.formerX=map(log,params)
        return params
    
     
    def after_transform(self, _):   
        '''
        Here we don't use the parameters, _, already simulated, we make new ones. 
        This doesn't change the prior
        '''
        self.latterX=multivariate_normal(self.formerX, self.theta*self.sigma)
        return map(exp,self.latterX)
        
    def update_alpha(self, accept, alphaXY):
        '''
        This updates alpha based on whether or not 'accept' is true or false. 
        We want: accept=True <=> the proposal based on alpha was accepted. 
        '''
        if accept:
            x=self.latterX
        else:
            x=self.formerX
        
        gamma=self.multip/self.count**self.alpha

        self.sigma += gamma*(outer(x-self.mean,x-self.mean)-self.sigma)
        self.mean += gamma*(x-self.mean)
        self.theta *= exp(gamma*(alphaXY-self.alphaDesired))
        self.count += 1
        
        
        return self.theta
        
        