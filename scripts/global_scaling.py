'''
Created on Oct 22, 2014

@author: svendvn
'''

from math import log, exp,sqrt

class Global_scaling(object):
    '''
    classdocs
    '''


    def __init__(self,theta=1.0, params=[0.5,10]):
        '''
        Constructor. theta is the factor that is multiplied on all proposals. It is updated throughout so input to the constructor is only a 
        starting value. 
        alpha is the power in the updating rule theta_{count+1}=max(0.0001,theta_{count}+1/count^alpha*(1_{accept}(true)*2-1))
        '''
        self.theta=theta
        self.count=1
        self.alpha=params[0]
        self.multip=params[1]
        
        
    
    def first_transform(self, params):
        '''
        this takes a vector and transforms it into the scaled space
        '''
        return [exp(log(x)/sqrt(self.theta)) for x in params]
    
     
    def after_transform(self, params):   
        '''
        this takes a vector from scaled space and transforms it back
        '''
        return [exp(log(x)*sqrt(self.theta)) for x in params]
        
    def update_alpha(self, accept):
        '''
        This updates alpha based on whether or not 'accept' is true or false. 
        We want: accept=True <=> the proposal based on alpha was accepted. 
        '''
        #0.234 has some special meaning in mcmc acceptance probabilities.
        self.theta = max(0.1,self.theta+self.multip/self.count**self.alpha*(float(accept)*1.0-(1.0-float(accept))/9.0))
        self.count+=1
        return self.theta
        
        