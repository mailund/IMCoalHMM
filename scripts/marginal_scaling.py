'''
Created on Oct 22, 2014

@author: svendvn
'''


from scipy.stats import norm
from math import log, exp,sqrt
from random import random,randint

class MarginalScaler(object):
    '''
    classdocs
    '''


    def __init__(self,startVal=[0.1]*17, params=[0.5,10], alphaDesired=0.234):
        '''
        Constructor. theta is the factor that is multiplied on all proposals. It is updated throughout so input to the constructor is only a 
        starting value. 
        alpha is the power in the updating rule theta_{count+1}=max(0.0001,theta_{count}+1/count^alpha*(1_{accept}(true)*2-1))
        '''
        self.theta=sum(startVal)
        self.thetas=[s/sum(startVal) for s in startVal]
        self.count=1
        self.alpha=params[0]
        self.multip=params[1]
        self.alphaDesired=alphaDesired
        self.marginals=[0]*len(startVal)
        self.power=0.1
    
    def setAdapParam(self,val):
        if len(val)>1:
            self.theta=val[0][0]
            self.thetas=val[1]
        else:
            self.theta=val[0][0]
        
    def getAdapParam(self,all=False):
        if all:
            return [self.theta],self.thetas
        return [self.theta]
    
    def getStandardizedLogJumps(self):
        return [(f-s) for f,s in zip(self.first,self.second)]
        
    
    def first_transform(self, params):
        '''
        this takes a vector and transforms it into the scaled space
        '''
        self.first=[log(x)/sqrt(t*self.theta) for x,t in zip(params,self.thetas)]
        print [exp(j) for j in self.first]
        return [exp(j) for j in self.first]
    
     
    def after_transform(self, params):   
        '''
        this takes a vector from scaled space and transforms it back
        '''
        self.second=[log(x) for x in params]
        print [exp(x*sqrt(t)*self.theta) for x,t in zip(self.second,self.thetas)]
        return [exp(x*sqrt(t)*self.theta) for x,t in zip(self.second,self.thetas)]

    
        
    def update_alpha(self, accept,alphaXY):
        '''
        This updates alpha based on whether or not 'accept' is true or false. 
        We want: accept=True <=> the proposal based on alpha was accepted. 
        
        Other algorithms return two values, that's why we ignore one input. 
        '''
        #extremity is the probability of observing something less extreme than what is observed. If that probability is high, we want that sample to mean a lot.
        gamma=self.multip/self.count**self.alpha
        
        #jumpsCorrection is approximately the acceptance probability normalized so that it has mean one in a one dimensional case if we assume a normal target density.
        jumpsCorrection=[exp(-((i-j)**2)/self.power)*sqrt(1+2*0.01/self.power) for i,j in zip(self.first,self.second)]
        for n,e in enumerate(jumpsCorrection):
            print e
            self.thetas[n]=min(100,max(self.thetas[n]*exp(gamma*(alphaXY-self.alphaDesired*e)),0.001))   #p is truncated in order to make every step evaulatable-
        self.theta*=sum(self.thetas)
        self.thetas=[s/sum(self.thetas) for s in self.thetas]
        print "alphaXY="+str(alphaXY)
        print "new theta="+str(self.theta)
        print "new thetas="+str(self.thetas)

        self.count+=1
        return [self.theta],self.thetas
        
        