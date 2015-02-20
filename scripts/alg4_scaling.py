'''
Created on Oct 27, 2014

@author: svendvn
'''

from math import log, exp, sqrt
from numpy.random import multivariate_normal, random, normal
from numpy import array,matrix,identity, outer, diagonal,real,cov
from numpy.linalg import eig
from random import randrange

class AM4_scaling(object):
    '''
    classdocs
    '''
    
    NONSWAP_PARAM=("theta",)
    SWAP_PARAM=("thetaD","thetaID","thetaIID")

    def __init__(self,startVal=1.0, params=[0.5,1,100,(0.2,0.0),0], sigmaStart=None, alphaDesired=0.234):
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
        self.timeTillStart=params[2]
        self.proportions=params[3]
        self.major=params[4]
        self.mean=array([0]*size)
        self.alphaDesired=alphaDesired
        self.adap=2
        self.firstValues=[]
        
        self.thetaDependent=1.0
        self.thetaIndependent=1.0
        self.thetaIdentical=1.0
        
    
    def setAdapParam(self, val):
        if len(val)>1:
            self.sigma=val[1][3]
            self.theta=val[0][0]
            self.thetaDependent=val[1][0]
            self.thetaIndependent=val[1][1]
            self.thetaIdentical=val[1][2]
            self.count=val[1][4]
            self.adap=val[1][5]
            self.firstValues=val[1][6]
        else:
            self.theta=val[0]
        
    def getAdapParam(self,all=False):
        if all:
            return [self.theta], [self.thetaDependent, self.thetaIndependent,self.thetaIdentical,self.sigma,self.count,self.adap, self.firstValues]
        return [self.theta]
    
    def getStandardizedLogJumps(self):
        return [0] #this vector is not well defined here/useful here. 
    
    def first_transform(self, params): 
        '''
        We record the mean. 
        '''
        self.first=map(log,params)
        
    def stationaryPossible(self):
        return False
    
     
    def after_transform(self, adds):   
        '''
        Here we don't use the parameters, _, already simulated, we make new ones. 
        This doesn't change the prior
        '''
        if self.count>self.timeTillStart and random()>self.proportions[0]:
            if random()<self.proportions[1]/(1-self.proportions[0]):
                print "count,self.theta, self.thetaID="+str((self.count,self.theta,self.thetaIndependent))
#                 print "diagsigma="+str(diagonal(self.sigma))
                self.second = normal(loc=self.first, scale=sqrt(self.theta*self.thetaIndependent)*0.1*array(map(sqrt, diagonal(self.sigma))), size=len(self.first))
#                 print "expsecond="+str(map(exp,self.second))
                self.adap=1
            else:
                print "count,self.theta, self.thetaD="+str((self.count,self.theta,self.thetaDependent))
                self.second = multivariate_normal(self.first, 0.01*self.theta*self.thetaDependent*self.sigma)
                self.adap=0
        else:
            print "count, self.theta, self.thetaIID="+str((self.count,self.theta,self.thetaIdentical))
#             print "self.first="+str(self.first)
#             print "adds="+str(adds)
            self.second=[(f+a*sqrt(self.theta)*sqrt(self.thetaIdentical)) for f,a in zip(self.first,adds)]
#             print "logsecond="+str(self.second)
            self.adap=2
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
        if self.count>self.timeTillStart:
            self.sigma += gamma*(outer(x-self.mean,x-self.mean)-self.sigma)
            self.mean += gamma*(x-self.mean)
        else:
            self.firstValues.append(x)
        
        
        
        print "adjusting adap "+str(self.adap)+" when major="+str(self.major)
        multip=max(0.001,exp(gamma*(alphaXY-self.alphaDesired)))#limiting, so it wont be too close to 0.
        print "multip="+str(multip)
        if self.adap==self.major:
            self.theta *= multip
            if self.major==0:
                self.thetaIndependent/=multip
                self.thetaIdentical/=multip
            elif self.major==1:
                self.thetaDependent/=multip
                self.thetaIdentical/=multip
            else:
                self.thetaIndependent/=multip
                self.thetaDependent/=multip
        elif self.adap==0:
            self.thetaDependent *= multip
        elif self.adap==1:
            self.thetaIndependent *= multip
        elif self.adap==2:
            self.thetaIdentical *= multip
            
            
        if self.count==self.timeTillStart:#we try to normalize self.theta, to skip some steps
            self.sigma=cov(map(list,zip(*self.firstValues)))
            self.firstValues=[]
            print self.sigma
            if self.major==2:
                pass
            elif self.major==0:
                normalizer=sum(diagonal(self.sigma))/len(self.first)
                print "normalizerD="+str(normalizer)
                self.theta=self.thetaIdentical/normalizer
            elif self.major==1:
                normalizer=real(sum(eig(self.sigma)[0]))/len(self.first)
                print "normalizerID="+str(normalizer)
                self.theta=self.thetaIdentical/normalizer
            
            
            
        self.count += 1
        if (self.count-3)%2000==0:
            print self.sigma
            print self.theta
        
        
        return [self.theta],[self.thetaDependent, self.thetaIndependent,self.thetaIdentical,self.sigma, self.count,self.adap,self.firstValues]
        
        