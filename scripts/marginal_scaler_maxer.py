'''
Created on Oct 22, 2014

@author: svendvn
'''


from scipy.stats import norm
from math import log, exp,sqrt
from random import random,randint
import operator

class MarginalScalerMax(object):
    '''
    classdocs
    '''


    def __init__(self,startVal=[0.1]*17, params=[0.5,10,True,0.1], alphaDesired=0.234, targetProportion=0.5):
        '''
        Constructor. theta is the factor that is multiplied on all proposals. It is updated throughout so input to the constructor is only a 
        starting value. 
        alpha is the power in the updating rule theta_{count+1}=max(0.0001,theta_{count}+1/count^alpha*(1_{accept}(true)*2-1))
        '''
        self.theta=sum(startVal)
        self.thetas=[s/sum(startVal) for s in startVal]
        self.mediorizing=params[2]
        self.count=1
        self.count2=[1]*len(startVal)
        self.alpha=params[0]
        self.multip=params[1]
        self.multip2=params[3]
        self.targetProportion=targetProportion
        self.alphaDesired=alphaDesired
    
    def setAdapParam(self,val):
        if len(val)>1:
            self.theta=val[0][0]
            self.thetas=val[1]
        else:
            self.theta=val[0]
        
    def getAdapParam(self,all=False):
        if all:
            return [self.theta],self.thetas
        return [self.theta]
    
    def getStandardizedLogJumps(self):
        return self.jumps
        
    
    def first_transform(self, params):
        '''
        this takes a vector and transforms it into the scaled space
        '''
        self.first=[log(x) for x in params]
    
     
    def after_transform(self, adds):   
        '''
        this takes a vector from scaled space and transforms it back
        '''
        self.second=[f+a*sqrt(t*self.theta) for f,a,t in zip(self.first,adds,self.thetas)]
        self.jumps=adds
        return [exp(s) for s in self.second]

    def stationaryPossible(self):
        return True
        
    def update_alpha(self, accept,alphaXY):
        '''
        This updates alpha based on whether or not 'accept' is true or false. 
        We want: accept=True <=> the proposal based on alpha was accepted. 
        
        Other algorithms return two values, that's why we ignore one input. 
        '''
        #extremity is the probability of observing something less extreme than what is observed. If that probability is high, we want that sample to mean a lot.
        gamma=self.multip/self.count**self.alpha

        #jumpsCorrection is approximately the acceptance probability normalized so that it has mean one in a one dimensional case if we assume a normal target density.
        if random()<self.targetProportion:
            max_index,_ = max(enumerate([a*sqrt(t) for a,t in zip(self.jumps, self.thetas)]), key=operator.itemgetter(1))
        else:
            max_index,_ = max(enumerate(self.jumps), key=operator.itemgetter(1))
        gammaMarg=self.multip2/self.count2[max_index]**self.alpha
        self.theta*=exp(gamma*(alphaXY-self.alphaDesired))
        if self.mediorizing:
            self.thetas[max_index]=max(0.001,self.thetas[max_index]+gammaMarg*(alphaXY-self.alphaDesired)/(1000*self.thetas[max_index]))
        else:
            self.thetas[max_index]=max(0.001,self.thetas[max_index]+gammaMarg*(alphaXY-self.alphaDesired))
        squaredSum=[s*s for s in self.thetas]           
        self.thetas=[max(s/sqrt(sum(squaredSum)),0.0001) for s in self.thetas]
        #vi vil holde sqrt(self.theta) under 100 og 1/sqrt(self.theta*min(self.thetas)) under 100 for at undgaa overflows


        self.count+=1
        self.count2[max_index]+=1
        
        return [self.theta],self.thetas
        
        