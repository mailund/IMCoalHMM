'''
Created on Oct 20, 2014

@author: svendvn
'''

from bisect import bisect
from math import log



class Coal_times_log_lik(object):
    '''
    classdocs
    '''


    def __init__(self, times,counts,model):
        self.times=times
        self.counts=counts
        self.model=model
    
    def __call__(self,*parameters):
        #emiss_probs will be ignored here. 
        if not self.model.valid_parameters(*parameters):
            return "help", "help",-float('inf')

        init_probs, trans_probs, _, break_points = self.model.build_hidden_markov_model(*parameters)   #ignored here is emission_probabilities
        
        
        t=[0]*len(self.times)
        for j in xrange(len(self.times)):
            t[j]=bisect(break_points,self.times[j])-1

        loglik=log(init_probs[0,t[0]])+self.counts[0]*log(trans_probs[t[0],t[0]])
        for i in xrange(1,len(t)):
            loglik+=log(trans_probs[t[i-1],t[i]])+log(trans_probs[t[i],t[i]])*self.counts[i]
        ans=(trans_probs,init_probs,loglik)
        return ans