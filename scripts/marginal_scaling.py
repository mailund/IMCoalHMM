'''
Created on Oct 22, 2014

@author: svendvn
'''


from scipy.stats import norm
from math import log, exp,sqrt
from random import random

class MarginalScaler(object):
    '''
    classdocs
    '''


    def __init__(self,startVal=[1.0]*17, params=[0.5,10], alphaDesired=0.234):
        '''
        Constructor. theta is the factor that is multiplied on all proposals. It is updated throughout so input to the constructor is only a 
        starting value. 
        alpha is the power in the updating rule theta_{count+1}=max(0.0001,theta_{count}+1/count^alpha*(1_{accept}(true)*2-1))
        '''
        self.thetas=startVal
        self.count=1
        self.alpha=params[0]
        self.multip=params[1]
        self.alphaDesired=alphaDesired
        self.marginal=-1
        
        
    
    def first_transform(self, params):
        '''
        this takes a vector and transforms it into the scaled space
        '''
        self.first=[log(x)/sqrt(t) for x,t in zip(params,self.thetas)]
        return [exp(j) for j in self.first]
    
     
    def after_transform(self, params):   
        '''
        this takes a vector from scaled space and transforms it back
        '''
        self.second=[log(x) for x in params]
        if random()<0.5:
            print k
            self.marginal=k
            return [exp(x*sqrt(t)) if j==k else exp(f*sqrt(t)) for x,t,f,j in zip(self.second,self.thetas,self.first,range(len(self.thetas)))]
        else:
            self.marginal=-1
            return [exp(x*sqrt(t)) for x,t in zip(self.second,self.thetas)]
    
    def setTheta(self, new_theta):
        self.theta=new_theta
        
    def update_alpha(self, accept,alphaXY):
        '''
        This updates alpha based on whether or not 'accept' is true or false. 
        We want: accept=True <=> the proposal based on alpha was accepted. 
        
        Other algorithms return two values, that's why we ignore one input. 
        '''
        #extremity is the probability of observing something less extreme than what is observed. If that probability is high, we want that sample to mean a lot.
        gamma=self.multip/self.count**self.alpha
        print self.marginal
        if self.marginal>=0:
            self.thetas[self.marginal]=max(self.thetas[self.marginal]*exp(gamma*(alphaXY-self.alphaDesired)),0.001)
        else:
            self.thetas=[max(t*exp(gamma*(alphaXY-self.alphaDesired)),0.001) for t in self.thetas]
        
        
        print "thetas="+str(self.thetas)
        self.count+=1
        return self.thetas
        
        