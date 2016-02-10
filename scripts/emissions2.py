"""Code for computing emission probabilities for pairwise sequence alignments.

The code uses a truncated exponential to get the coalescence time point
and a Jukes-Cantor for emission probabilities, with a pseudo emission
probability of 1 for missing data.
"""

from math import exp
from numpy.random import gamma
from numpy import cumsum,array, concatenate
from numpy import sum as npsum
from numpy.testing import assert_almost_equal
from pyZipHMM import Matrix
from bisect import bisect
from numpy.linalg import eig, inv,det
from operator import mul


def printPyZipHMM(Matrix):
    finalString=""
    for i in range(Matrix.getHeight()):
        for j in range(Matrix.getWidth()):
            finalString=finalString+" "+str(Matrix[i,j])
        finalString=finalString+"\n"
    return finalString


def truncated_exp_midpoint(t1, t2, rate):
    """Calculates the mean coalescence point between t1 and t2
    from a truncated exponential distribution.

    :param t1: Beginning of the interval.
    :type t1: float
    :param t2: End of the interval.
    :type t2: float
    :param rate: Coalescence rate within the interval. Used for computing the mean.
    :type rate: float

    :returns: a list of time points from which to emit the alignment.
    """
    delta_t = t2 - t1
    return t1 + 1.0 / rate - (delta_t * exp(-delta_t * rate)) / (1 - exp(-delta_t * rate))


def exp_midpoint(t, rate):
    """Calculates the mean coalescence point after t
    from an exponential distribution.

    :param t: an offset added to the mean given by the exponential distribution
     and the coalescence rate.
    :type t: float

    :param rate: the coalescence rate after time t.
    :type rate: float

    :returns: the mean coalescence rate after to.
    """
    return t + 1.0 / rate


def coalescence_points(break_points, rates):
    """Calculates the mean coalescence times (given the rate) between
    each time break point and after the last break point.

    :param break_points: Break points between the HMM states.
    :type break_points: list[float]
    :param rates: A coalescence rate or a list of rates for each interval.
    :type rates: float | list[float]

    :rtype: list[float]
    """
    if hasattr(rates, '__iter__'):
        assert len(rates) == len(break_points), \
            "You must have the same number of rates as break points."
    else:
        rates = [rates] * len(break_points)

    result = []
    for i in xrange(1, len(break_points)):
        t = truncated_exp_midpoint(break_points[i - 1], break_points[i], rates[i-1])
        result.append(t)
    result.append(exp_midpoint(break_points[-1], rates[-1]))
    return result


def jukes_cantor(a, b, dt):
    """Compute the Jukes-Cantor transition probability for switching from
    "a" to "b" in time "dt".

    :param a: The state at one end of the tree.
    :type a: object
    :param b: The state at the other end of the tree. The a and b objects
     are only used to test equality.
    :type b: object

    :param dt: the time distance between the two leaves in the tree.
    :type dt: float

    :returns: the probability of changing from a to b in time dt."""
    if a == b:
        return 0.25 + 0.75 * exp(-4.0 / 3 * dt)
    else:
        return 0.75 - 0.75 * exp(-4.0 / 3 * dt)


def emission_matrix(coal_points):
    """Compute the emission matrix given the time break points and coalescence
    rate.

    :param coal_points: List coalescence points to emit from.
    """
    emission_probabilities = Matrix(len(coal_points), 3)
    for state in xrange(len(coal_points)):
        emission_probabilities[state, 0] = jukes_cantor(0, 0, 2 * coal_points[state])
        emission_probabilities[state, 1] = jukes_cantor(0, 1, 2 * coal_points[state])
        emission_probabilities[state, 2] = 1.0  # Dummy for missing data
    return emission_probabilities

def emission_matrix2(break_points, rates):
    """Compute the emission matrix given the time break points and coalescence
    rate.

    :param coal_points: List coalescence points to emit from.
    """
    emission_probabilities = Matrix(len(break_points), 3)
    
    def integraterJC(t2,t1,rate):
        cr=8.0/3+rate
        ans=float(rate)/cr*(exp(-cr*t1)-exp(-cr*t2))/(exp(-rate*t1)-exp(-rate*t2))
        return ans
    def integraterJCinf(t1,rate):
        cr=8.0/3+rate
        ans=float(rate)/cr*(exp(-cr*t1))/exp(-rate*t1)
        return ans
    
    for state in xrange(len(break_points)-1):
        emission_probabilities[state, 0] = 0.25+0.75*integraterJC(break_points[state+1],break_points[state],rates[state])
        emission_probabilities[state, 1] = 0.75-0.75*integraterJC(break_points[state+1],break_points[state],rates[state])
        emission_probabilities[state, 2] = 1.0  # Dummy for missing data
     
    state=len(break_points)-1   
    emission_probabilities[state, 0] = 0.25+0.75*integraterJCinf(break_points[state],rates[state])
    emission_probabilities[state, 1] = 0.75-0.75*integraterJCinf(break_points[state],rates[state])
    emission_probabilities[state, 2] = 1.0  # Dummy for missing data
    
    return emission_probabilities

def emission_matrix3(break_points, params,intervals):
    """
    This calculates the emission matrix in the model variable-migration-model.
    This assumes in each time interval that the the migration and 
    coalescence rate of the time interval has been constant from that time interval to the present.
    Under this assumption, the calculations are true(so to say) following 
    'Efficient computation in the IM model, Lars Noervang Andersen, Thomas Mailund, Asger Hobolth.'
    In that article they calculate a integral f(t)*(1/4+3/4exp(-4/3t)) from 0 to infinity, but here it is 
    calculated from t_i to t_{i+1} and divided by normsum which is int_{t_i}^{t_{i+1}} f(t) dt.
    params are the parameters in the variable-migration-model. 
    The parameters are untransformed, meaning the parameters for population sizes are coalescence rates.
    intervals is a list indicating the number of intervals in each epoch(and the number of epochs specified by the length og intervals).
    break_points is the list [t_0,t_1,...,t_(sum(intervals))]. 
    
    output: emission matrix.   
    """


    no_epochs=len(intervals)
    epoch=-1
    intervals=cumsum(array(intervals))
    emission_probabilities = Matrix(len(break_points), 3)
    for j in range(len(break_points[:-1])):
        #new epoch is the epoch that corresponds to interval j
        new_epoch=bisect(intervals, j)
        if new_epoch!=epoch:
            epoch=new_epoch
            c1=params[0:no_epochs][epoch]
            c2=params[no_epochs:2*no_epochs][epoch]
            mig12 = params[(2 * no_epochs):(3 * no_epochs)][epoch]
            mig21 = params[(3 * no_epochs):(4 * no_epochs)][epoch]
            Qgamma=array([[-2*mig12-c1, 2*mig12,0],[mig21,-mig21-mig12, mig12], [0,2*mig21,-2*mig21-c2]])
            QgammaA=array([[-2*mig12-c1, 2*mig12,0,c1],[mig21,-mig21-mig12, mig12,0], [0,2*mig21,-2*mig21-c2,c2],[0,0,0,0]])
            w,V=eig(Qgamma)
            Vinv=inv(V)
            emissum=0
            normsum=0
            A=3
            for i in range(3):
                for k in range(2):
                    normsum+=V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp(break_points[j+1]*w[i]) - exp(break_points[j]*w[i]))/w[i]
                    emissum+=V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp(break_points[j+1]*w[i]) - exp(break_points[j]*w[i])) / (4.0*w[i]) +
                                                               3.0/4.0*(exp(break_points[j+1]*(w[i]-8.0/3.0)) - exp(break_points[j]*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
            
        else:
            emissum=0
            normsum=0
            A=3
            for i in range(3):
                for k in range(2):
                    normsum+=V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp(break_points[j+1]*w[i]) - exp(break_points[j]*w[i]))/w[i]
                    emissum+=V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp(break_points[j+1]*w[i]) - exp(break_points[j]*w[i])) / (4.0*w[i]) +
                                                               3.0/4.0*(exp(break_points[j+1]*(w[i]-8.0/3.0)) - exp(break_points[j]*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
        
   #     print str(normsum)+" "+str(emissum)
        if emissum<0 or normsum<0 or emissum>normsum:
            print "Error in emission matrix 3 " + "calculating row number " + str(j) +"."
            print "emissum "+str(emissum)
            print "normsum "+str(normsum)
            print "guilty parameters " + str(params)
            print "guilty break points " + str(break_points)
            print "guilty (transformed) intervals " + str(intervals) 
        emission_probabilities[j,0]=emissum/normsum
        emission_probabilities[j,1]=1-emissum/normsum
        emission_probabilities[j,2]=1.0
    emissum=0
    normsum=0
    j=len(break_points)-1
    A=3
    for i in range(3):
        for k in range(2):
            normsum+=V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp(break_points[j]*100*w[i]) - exp(break_points[j]*w[i]))/w[i]
            emissum+=V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp(break_points[j]*100*w[i]) - exp(break_points[j]*w[i])) / (4.0*w[i]) +
                                                       3.0/4.0*(exp(break_points[j]*100*(w[i]-8.0/3.0)) - exp(break_points[j]*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
    emission_probabilities[j,0]=emissum/normsum
    emission_probabilities[j,1]=1-emissum/normsum
    emission_probabilities[j,2]=1.0    
    #print printPyZipHMM(emission_probabilities)
    return emission_probabilities

def emission_matrix3b(break_points, params,intervals,ctmc_system):
    """
    This calculates the emission matrix in the model variable-migration-model.
    This assumes in each time interval that the the migration and 
    coalescence rate of the time interval has been constant from that time interval to the present.
    Under this assumption, the calculations are true(so to say) following 
    'Efficient computation in the IM model, Lars Noervang Andersen, Thomas Mailund, Asger Hobolth.'
    In that article they calculate a integral f(t)*(1/4+3/4exp(-4/3t)) from 0 to infinity, but here it is 
    calculated from t_i to t_{i+1} and divided by normsum which is int_{t_i}^{t_{i+1}} f(t) dt.
    params are the parameters in the variable-migration-model. 
    The parameters are untransformed, meaning the parameters for population sizes are coalescence rates.
    intervals is a list indicating the number of intervals in each epoch(and the number of epochs specified by the length og intervals).
    break_points is the list [t_0,t_1,...,t_(sum(intervals))]. 
    
    output: emission matrix.   
    """
    states=ctmc_system.get_state_space(None).states #all states are identical.
    leftleft=[]
    leftright=[]
    rightright=[]
    restIndexes=[]
    for state,number in states.items():
        coalesced=False 
        position1=-1
        position2=-1
        for branch, (b1,_) in state:#we only do this marginally for pop1
            if 1 in b1:
                position1=branch
                if 2 in b1:
                    coalesced=True
            if 2 in b1:
                position2=branch
        if not coalesced:
            if position1==1 and position2==1:
                leftleft.append(number)
#                 print "ll "+str(state)
            elif position1==2 and position2==2:
                rightright.append(number)
#                print "rr "+str(state)
            elif position1!=-1 and position2!=-1:
                leftright.append(number) 
                #pass#print "lr "+str(state)
        else:
            restIndexes.append(number)
            #print "non2 "+str(state)


    i=ctmc_system.initial_ctmc_state
    if i in leftleft:
        ind=0
    elif i in leftright:
        ind=1
    elif i in rightright:
        ind=2
    else:
        print "error"
    print leftleft
    print i
    print leftright
    print "INDINDIND="+str(ind)

    no_epochs=len(intervals)
    epoch=-1
    intervals=cumsum(array(intervals))
    emission_probabilities = Matrix(len(break_points), 3)
    for j in range(len(break_points[:-1])):
        #new epoch is the epoch that corresponds to interval j
        new_epoch=bisect(intervals, j)
        if new_epoch!=epoch:
            epoch=new_epoch
            c1=params[0:no_epochs][epoch]
            c2=params[no_epochs:2*no_epochs][epoch]
            mig12 = params[(2 * no_epochs):(3 * no_epochs)][epoch]
            mig21 = params[(3 * no_epochs):(4 * no_epochs)][epoch]
            #print str((c1,c2,mig12,mig21))
            Qgamma=array([[-2*mig12-c1, 2*mig12,0],[mig21,-mig21-mig12, mig12], [0,2*mig21,-2*mig21-c2]])
            QgammaA=array([[-2*mig12-c1, 2*mig12,0,c1],[mig21,-mig21-mig12, mig12,0], [0,2*mig21,-2*mig21-c2,c2],[0,0,0,0]])
            w,V=eig(Qgamma)
            Vinv=inv(V)
            emissum=0
            normsum=0
            A=3
            for i in range(3):
                for k in range(2):
                    normsum+=V[ind,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp(break_points[j+1]*w[i]) - exp(break_points[j]*w[i]))/w[i]
                    emissum+=V[ind,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp(break_points[j+1]*w[i]) - exp(break_points[j]*w[i])) / (4.0*w[i]) +
                                                               3.0/4.0*(exp(break_points[j+1]*(w[i]-8.0/3.0)) - exp(break_points[j]*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
            
        else:
            emissum=0
            normsum=0
            A=3
            for i in range(3):
                for k in range(2):
                    normsum+=V[ind,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp(break_points[j+1]*w[i]) - exp(break_points[j]*w[i]))/w[i]
                    emissum+=V[ind,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp(break_points[j+1]*w[i]) - exp(break_points[j]*w[i])) / (4.0*w[i]) +
                                                               3.0/4.0*(exp(break_points[j+1]*(w[i]-8.0/3.0)) - exp(break_points[j]*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
        
   #     print str(normsum)+" "+str(emissum)
        if emissum<0 or normsum<0 or emissum>normsum:
            print "Error in emission matrix 3 " + "calculating row number " + str(j) +"."
            print "emissum "+str(emissum)
            print "normsum "+str(normsum)
            print "guilty parameters " + str(params)
            print "guilty break points " + str(break_points)
            print "guilty (transformed) intervals " + str(intervals) 
        emission_probabilities[j,0]=emissum/normsum
        emission_probabilities[j,1]=1.-emissum/normsum
        emission_probabilities[j,2]=1.0
    emissum=0
    normsum=0
    j=len(break_points)-1
    A=3
    for i in range(3):
        for k in range(2):
            normsum+=V[ind,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp(break_points[j]*100*w[i]) - exp(break_points[j]*w[i]))/w[i]
            emissum+=V[ind,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp(break_points[j]*100*w[i]) - exp(break_points[j]*w[i])) / (4.0*w[i]) +
                                                       3.0/4.0*(exp(break_points[j]*100*(w[i]-8.0/3.0)) - exp(break_points[j]*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
    emission_probabilities[j,0]=emissum/normsum
    emission_probabilities[j,1]=1-emissum/normsum
    emission_probabilities[j,2]=1.0    
    #print printPyZipHMM(emission_probabilities)
    return emission_probabilities
            
              



def emission_matrix4(break_points, params,intervals, ctmc_system):
    """
    This calculates the emission matrix in the model variable-migration-model.
    This assumes in each time interval that the the migration and 
    coalescence rate of the time interval has been constant from that time interval to the present.
    Under this assumption, the calculations are true(so to say) following 
    'Efficient computation in the IM model, Lars Noervang Andersen, Thomas Mailund, Asger Hobolth.'
    In that article they calculate a integral f(t)*(1/4+3/4exp(-4/3t)) from 0 to infinity, but here it is 
    calculated from t_i to t_{i+1} and divided by normsum which is int_{t_i}^{t_{i+1}} f(t) dt.
    params are the parameters in the variable-migration-model. 
    The parameters are untransformed, meaning the parameters for population sizes are coalescence rates.
    intervals is a list indicating the number of intervals in each epoch(and the number of epochs specified by the length og intervals).
    break_points is the list [t_0,t_1,...,t_(sum(intervals))]. 
    
    output: emission matrix.   
    """

 #           print "non2 "+str(state)
            
    states=ctmc_system.get_state_space(0).states #all states are identical.
    leftleft=[]
    leftright=[]
    rightright=[]
    restIndexes=[]
    for state,number in states.items():
        coalesced=False 
        position1=-1
        position2=-1
        for branch, (b1,_) in state:#we only do this marginally for pop1
            if 1 in b1:
                position1=branch
                if 2 in b1:
                    coalesced=True
            if 2 in b1:
                position2=branch
        if not coalesced:
            if position1==1 and position2==1:
                leftleft.append(number)
#                 print "ll "+str(state)
            elif position1==2 and position2==2:
                rightright.append(number)
#                print "rr "+str(state)
            elif position1!=-1 and position2!=-1:
                leftright.append(number) 
                #pass#print "lr "+str(state)
#            else:
#                print "non1 "+str(state)
        else:
            restIndexes.append(number)

    no_epochs=len(intervals)
    epoch=-1
    intervals=cumsum(array(intervals))
    emission_probabilities = Matrix(len(break_points), 3)
    lls=0 #P(B_ti=left) that is the probability that both ancestor-lines of the same position in the two genomes are in the left population
    lrs=0 #P(B_ti=both)
    rrs=0 #P(B_ti=right)
    rest=0 #probability that they have coalesced.
    r=ctmc_system.up_to(0)
    i=ctmc_system.initial_ctmc_state
    for k in range(r.shape[1]):
        if k in leftleft:
            lls+=r.item((i,k))
        elif k in rightright:
            rrs+=r.item((i,k))
        elif k in leftright:
            lrs+=r.item((i,k))
        else:
            rest+=r.item((i,k))
    if (lls+rrs+lrs)<1e-7:
#        print "something has gone wrong in tranisition probability"
        lrs=0.1
        rest=0.9

    for j in range(len(break_points[:-1])):
        breaklatest=break_points[j]
        #new epoch is the epoch that corresponds to interval j
        new_epoch=bisect(intervals, j)
        llsafter=0
        lrsafter=0
        rrsafter=0
        restafter=0
        r=ctmc_system.up_to(j+1)
        i=ctmc_system.initial_ctmc_state
        for k in range(r.shape[1]):
            if k in leftleft:
                llsafter+=r.item((i,k))
            elif k in rightright:
                rrsafter+=r.item((i,k))
            elif k in leftright:
                lrsafter+=r.item((i,k))
            else:
                restafter+=r.item((i,k))
        if new_epoch!=epoch:
            epoch=new_epoch
            c1=params[0:no_epochs][epoch]
            c2=params[no_epochs:2*no_epochs][epoch]
            mig12 = params[(2 * no_epochs):(3 * no_epochs)][epoch]
            mig21 = params[(3 * no_epochs):(4 * no_epochs)][epoch]
            Qgamma=array([[-2*mig12-c1, 2*mig12,0],[mig21,-mig21-mig12, mig12], [0,2*mig21,-2*mig21-c2]])
            QgammaA=array([[-2*mig12-c1, 2*mig12,0,c1],[mig21,-mig21-mig12, mig12,0], [0,2*mig21,-2*mig21-c2,c2],[0,0,0,0]])
            w,V=eig(Qgamma)
            Vinv=inv(V)
            emissum=0
            normsum=0
            A=3
            middle=0
            left=0
            right=0
            for i in range(3):
                for k in range(2):
                    normsum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                    left+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j+1]-breaklatest)*w[i]) - 1) / (4.0*w[i]) +
                                                           3.0/4.0*exp(-8.0/3.0*breaklatest)*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                    1 ) / (w[i]-8.0/3.0)    )
                    normsum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                    middle+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j+1]-breaklatest)*w[i]) - 1) / (4.0*w[i]) +
                                                           3.0/4.0*exp(-8.0/3.0*breaklatest)*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                    1 ) / (w[i]-8.0/3.0)    )
                    normsum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                    right+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j+1]-breaklatest)*w[i]) - 1) / (4.0*w[i]) +
                                                           3.0/4.0*exp(-8.0/3.0*breaklatest)*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                    1 ) / (w[i]-8.0/3.0)    )
            emissum=middle+left+right
#            print str((left,middle,right))
#            print str((left/lls,middle/lrs,right/rrs))
            
        else:
            emissum=0
            normsum=0
            A=3
            for i in range(3):
                for k in range(2):
                    normsum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                    emissum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i])) / (4.0*w[i]) +
                                                           3.0/4.0*exp(-8.0/3.0*breaklatest)*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                    exp((break_points[j]-breaklatest)*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
                    normsum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                    emissum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i])) / (4.0*w[i]) +
                                                           3.0/4.0*exp(-8.0/3.0*breaklatest)*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                    exp((break_points[j]-breaklatest)*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
                    normsum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                    emissum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i])) / (4.0*w[i]) +
                                                           3.0/4.0*exp(-8.0/3.0*breaklatest)*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                    exp((break_points[j]-breaklatest)*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
        
        #normsum=(restafter-rest)#this is P(T\in [t_i, t_i+1])
 #       print "e4 "+str(normsum) 
        emissum=emissum/(lls+lrs+rrs)*(1-rest) 
#        if emissum<0 or normsum<0 or emissum>normsum:
#            print "Error in emission matrix 3 " + "calculating row number " + str(j) +"."
#            print "emissum "+str(emissum)
#            print "normsum "+str(normsum)
            #print "guilty parameters " + str(params)
            #print "guilty break points " + str(break_points)
            #print "guilty (transformed) intervals " + str(intervals) 
        emission_probabilities[j,0]=emissum/normsum
        emission_probabilities[j,1]=1-emissum/normsum
        emission_probabilities[j,2]=1.0
        lls=llsafter
        lrs=lrsafter
        rrs=rrsafter
        rest=restafter
        if (lls+rrs+lrs)<1e-7:
#            print "something has gone wrong in tranisition probability"
            lrs=0.1
            rest=0.9
        
#         print "j "+str(j)
#         print "sum("+str([lls,lrs,rrs,rest])+")="+str(sum([lls,lrs,rrs,rest]))
#         print "lls "+str(lls)
#         print "lrs "+str(lrs)
#         print "rrs "+str(rrs)
#        print "rest "+str(rest)
    emissum=0
    normsum=0
    j=len(break_points)-1
    A=3
    for i in range(3):
        for k in range(2):
            normsum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j]*100-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
            emissum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j]*100-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i])) / (4.0*w[i]) +
                                                   3.0/4.0*exp(-8.0/3.0*breaklatest)*(exp((break_points[j]*100-breaklatest)*(w[i]-8.0/3.0)) - exp((break_points[j]-breaklatest)*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
            normsum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j]*100-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
            emissum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j]*100-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i])) / (4.0*w[i]) +
                                                   3.0/4.0*exp(-8.0/3.0*breaklatest)*(exp((break_points[j]*100-breaklatest)*(w[i]-8.0/3.0)) - exp((break_points[j]-breaklatest)*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
            normsum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j]*100-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
            emissum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j]*100-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i])) / (4.0*w[i]) +
                                                   3.0/4.0*exp(-8.0/3.0*breaklatest)*(exp((break_points[j]*100-breaklatest)*(w[i]-8.0/3.0)) - exp((break_points[j]-breaklatest)*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
    emissum=emissum/(lls+lrs+rrs)*(1-rest) 
    emission_probabilities[j,0]=emissum/normsum
    emission_probabilities[j,1]=1-emissum/normsum
    emission_probabilities[j,2]=1.0    
    #print printPyZipHMM(emission_probabilities)
    return emission_probabilities
            


def emission_matrix5(break_points, params,intervals, ctmc_system,offset):
    """
    Like emission_matrix4 except all the states are not necessarily identical.  
    """

 #           print "non2 "+str(state)
            
    no_epochs=len(intervals)
    epoch=-1
    intervals=cumsum(array(intervals))
    emission_probabilities = Matrix(len(break_points), 3)

    for j in range(len(break_points[:-1])):
        breaklatest=break_points[j]
        #new epoch is the epoch that corresponds to interval j
        new_epoch=bisect(intervals, j)
        if new_epoch!=epoch:
            states=ctmc_system.get_state_space(j).states #all states are identical.
            leftleft=[]
            leftright=[]
            rightright=[]
            restIndexes=[]
            for state,number in states.items():
                coalesced=False 
                position1=-1
                position2=-1
                for branch, (b1,_) in state:#we only do this marginally for pop1
                    if 1 in b1:
                        position1=branch
                        if 2 in b1:
                            coalesced=True
                    if 2 in b1:
                        position2=branch
                if not coalesced:
                    if position1==1 and position2==1:
                        leftleft.append(number)
        #                 print "ll "+str(state)
                    elif position1==2 and position2==2:
                        rightright.append(number)
        #                print "rr "+str(state)
                    elif position1!=-1 and position2!=-1:
                        leftright.append(number) 
                        #pass#print "lr "+str(state)
        #            else:
        #                print "non1 "+str(state)
                else:
                    restIndexes.append(number)
        lls=0 #P(B_ti=left) that is the probability that both ancestor-lines of the same position in the two genomes are in the left population
        lrs=0 #P(B_ti=both)
        rrs=0 #P(B_ti=right)
        rest=0 #probability that they have coalesced.
        r=ctmc_system.up_to(j)
        i=ctmc_system.initial_ctmc_state
        for k in range(r.shape[1]):
            if k in leftleft:
                lls+=r.item((i,k))
            elif k in rightright:
                rrs+=r.item((i,k))
            elif k in leftright:
                lrs+=r.item((i,k))
            else:
                rest+=r.item((i,k))
        if new_epoch!=epoch:
            epoch=new_epoch
            c1=params[0:no_epochs][epoch]
            c2=params[no_epochs:2*no_epochs][epoch]
            mig12 = params[(2 * no_epochs):(3 * no_epochs)][epoch]
            mig21 = params[(3 * no_epochs):(4 * no_epochs)][epoch]
            if mig12!=0 or mig21!=0:  #in this case we can use diffusion.
                Qgamma=array([[-2*mig12-c1, 2*mig12,0],[mig21,-mig21-mig12, mig12], [0,2*mig21,-2*mig21-c2]])
                QgammaA=array([[-2*mig12-c1, 2*mig12,0,c1],[mig21,-mig21-mig12, mig12,0], [0,2*mig21,-2*mig21-c2,c2],[0,0,0,0]])
                w,V=eig(Qgamma)
                Vinv=inv(V)
                emissum=0
                normsum=0
                A=3
                middle=0
                left=0
                right=0
                for i in range(3):
                    for k in range(2):
                        normsum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                        left+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j+1]-breaklatest)*w[i]) - 1) / (4.0*w[i]) +
                                                               3.0/4.0*exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                        1 ) / (w[i]-8.0/3.0)    )
                        normsum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                        middle+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j+1]-breaklatest)*w[i]) - 1) / (4.0*w[i]) +
                                                               3.0/4.0*exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                        1 ) / (w[i]-8.0/3.0)    )
                        normsum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                        right+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j+1]-breaklatest)*w[i]) - 1) / (4.0*w[i]) +
                                                               3.0/4.0*exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                        1 ) / (w[i]-8.0/3.0)    )
                emissum=middle+left+right
            else:
                #if we can't use the integrals the two either belongs to the same population or different populations. If they belong to the same it is exponentially distributed.
                emissum=0
                normsum=0
                emissum+=lls*(0.25-3.0/4.0*exp(-c1*(break_points[j+1]-break_points[j]))+3.0/4.0/(3.0/8.0+c1)*exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-break_points[j])*(3.0/8.0+c1)) - 1 ))
                normsum+=lls*(1.0-exp(-c1*(break_points[j+1]-break_points[j]))) 
#                emissum+=lrs*0
 #               normsum+=lrs*0
                emissum+=rrs*(0.25-3.0/4.0*exp(-c2*(break_points[j+1]-break_points[j]))+3.0/4.0/(3.0/8.0+c2)*exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-break_points[j])*(3.0/8.0+c2)) - 1 ))
                normsum+=rrs*(1.0-exp(-c2*(break_points[j+1]-break_points[j]))) 
                
#            print str((left,middle,right))
#            print str((left/lls,middle/lrs,right/rrs))
            
            
        else:
            if mig12!=0 or mig21!=0:
                emissum=0
                normsum=0
                A=3
                for i in range(3):
                    for k in range(2):
                        normsum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                        emissum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i])) / (4.0*w[i]) +
                                                               3.0/4.0*exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                        exp((break_points[j]-breaklatest)*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
                        normsum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                        emissum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i])) / (4.0*w[i]) +
                                                               3.0/4.0*exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                        exp((break_points[j]-breaklatest)*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
                        normsum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                        emissum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j+1]-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i])) / (4.0*w[i]) +
                                                               3.0/4.0*exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                        exp((break_points[j]-breaklatest)*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
        
               
            else:
                emissum=0
                normsum=0
                emissum+=lls*(0.25-1.0/4.0*exp(-c1*(break_points[j+1]-breaklatest))+3.0/4.0/(8.0/3.0+c1)*exp(-8.0/3.0*(breaklatest+offset))*(exp(-(break_points[j+1]-breaklatest)*(8.0/3.0+c1)) - 1 ))
                normsum+=lls*(1.0-exp(-c1*(break_points[j+1]-breaklatest))) 
#               emissum+=lrs*0
#               normsum+=lrs*0
                emissum+=rrs*(0.25-1.0/4.0*exp(-c2*(break_points[j+1]-breaklatest))+3.0/4.0/(8.0/3.0+c2)*exp(-8.0/3.0*(breaklatest+offset))*(exp(-(break_points[j+1]-breaklatest)*(8.0/3.0+c2)) - 1 ))
                normsum+=rrs*(1.0-exp(-c2*(break_points[j+1]-breaklatest))) 
                #emissum=emissum/(lls+rrs)*(1-rest)*(lls+lrs+rrs)/(1-rest)
        #normsum=(restafter-rest)#this is P(T\in [t_i, t_i+1])
 #       print "e4 "+str(normsum) 
         
#        if emissum<0 or normsum<0 or emissum>normsum:
#            print "Error in emission matrix 3 " + "calculating row number " + str(j) +"."
#            print "emissum "+str(emissum)
#            print "normsum "+str(normsum)
            #print "guilty parameters " + str(params)
            #print "guilty break points " + str(break_points)
            #print "guilty (transformed) intervals " + str(intervals) 
        emissum=emissum/(lls+lrs+rrs)*(1-rest)
        emission_probabilities[j,0]=emissum/normsum
        emission_probabilities[j,1]=1-emissum/normsum
        emission_probabilities[j,2]=1.0
        
#         print "j "+str(j)
#         print "sum("+str([lls,lrs,rrs,rest])+")="+str(sum([lls,lrs,rrs,rest]))
#         print "lls "+str(lls)
#         print "lrs "+str(lrs)
#         print "rrs "+str(rrs)
#        print "rest "+str(rest)
    emissum=0
    normsum=0
    j=len(break_points)-1
    if mig12!=0 or mig21!=0:
        A=3
        for i in range(3):
            for k in range(2):
                normsum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j]*100-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                emissum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j]*100-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i])) / (4.0*w[i]) +
                                                       3.0/4.0*exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j]*100-breaklatest)*(w[i]-8.0/3.0)) - exp((break_points[j]-breaklatest)*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
                normsum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j]*100-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                emissum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j]*100-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i])) / (4.0*w[i]) +
                                                       3.0/4.0*exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j]*100-breaklatest)*(w[i]-8.0/3.0)) - exp((break_points[j]-breaklatest)*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
                normsum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j]*100-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i]))/w[i]
                emissum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   (exp((break_points[j]*100-breaklatest)*w[i]) - exp((break_points[j]-breaklatest)*w[i])) / (4.0*w[i]) +
                                                       3.0/4.0*exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j]*100-breaklatest)*(w[i]-8.0/3.0)) - exp((break_points[j]-breaklatest)*(w[i]-8.0/3.0)) ) / (w[i]-8.0/3.0)    )
    else:
        emissum=0
        normsum=0
        emissum+=lls*(0.25-1.0/4.0*exp(-c1*(break_points[j]*100-breaklatest))+3.0/4.0/(3.0/8.0+c1)*exp(-8.0/3.0*(breaklatest+offset))*(exp(-(break_points[j]*100-breaklatest)*(3.0/8.0+c1)) - 1 ))
        normsum+=lls*(1.0-exp(-c1*(break_points[j+1]-breaklatest))) 
#       emissum+=lrs*0
#       normsum+=lrs*0
        emissum+=rrs*(0.25-1.0/4.0*exp(-c2*(break_points[j]*100-breaklatest))+3.0/4.0/(3.0/8.0+c2)*exp(-8.0/3.0*(breaklatest+offset))*(exp(-(break_points[j]*100-breaklatest)*(3.0/8.0+c2)) - 1 ))
        normsum+=rrs*(1.0-exp(-c2*(break_points[j]*100-breaklatest))) 
    
    emissum=emissum/(lls+lrs+rrs)*(1-rest) 
    emission_probabilities[j,0]=emissum/normsum
    emission_probabilities[j,1]=1-emissum/normsum
    emission_probabilities[j,2]=1.0    
    #print printPyZipHMM(emission_probabilities)
    return emission_probabilities









def emission_matrix6(break_points, params,intervals, ctmc_system, offset=0.0, ctmc_postpone=0, outgroup=None,dimOfEmissionMatrix=None):
    """
    Like emission_matrix4 except all the states are not necessarily identical.  
    dimOfEmissionMatrix should always be larger than len(break_points). The extra entries will be filled by default values
    
    """

 #           print "non2 "+str(state)      
    no_epochs=len(intervals)
    epoch=-1
    intervals=cumsum(array(intervals))
    if dimOfEmissionMatrix is None:
        emission_probabilities = Matrix(len(break_points), 3)  #Once, this said ctmc_system.break_points, but I have changed it to 
    else:
        emission_probabilities = Matrix(dimOfEmissionMatrix, 3)
    for j in xrange(ctmc_postpone):
        emission_probabilities[j,0]=1.0
        emission_probabilities[j,1]=0.0
        emission_probabilities[j,2]=1.0
    restbefore=0

    for j in range(len(break_points[:-1])):
        breaklatest=break_points[j]
        #new epoch is the epoch that corresponds to interval j
        new_epoch=bisect(intervals, j)
        if new_epoch!=epoch:
            states=ctmc_system.get_state_space(j+ctmc_postpone).states #all states are identical.
            
            leftleft=[]
            leftright=[]
            rightright=[]
            restIndexes=[]
            for state,number in states.items():
                coalesced=False 
                position1=-1
                position2=-1
                for branch, (b1,_) in state:#we only do this marginally for pop1
                    if 1 in b1:
                        position1=branch
                        if 2 in b1:
                            coalesced=True
                    if 2 in b1:
                        position2=branch
                if not coalesced:
                    if position1==1 and position2==1:
                        leftleft.append(number)
        #                 print "ll "+str(state)
                    elif position1==2 and position2==2:
                        rightright.append(number)
        #                print "rr "+str(state)
                    elif position1==position2:#this is used for initial_migration_model and variable_migration_model_with_ancestral which has 0 as the ancestral population
                        leftleft.append(number)
                    elif position1!=-1 and position2!=-1:
                        leftright.append(number)

                        #pass#print "lr "+str(state)
        #            else:
        #                print "non1 "+str(state)
                else:
                    restIndexes.append(number)
        lls=0 #P(B_ti=left) that is the probability that both ancestor-lines of the same position in the two genomes are in the left population
        lrs=0 #P(B_ti=both)
        rrs=0 #P(B_ti=right)
        rest=0 #probability that they have coalesced.
        r=ctmc_system.up_to(j+ctmc_postpone)
        i=ctmc_system.initial_ctmc_state
        for k in range(r.shape[1]):
            if k in leftleft:
                lls+=r.item((i,k))
            elif k in rightright:
                rrs+=r.item((i,k))
            elif k in leftright:
                lrs+=r.item((i,k))
            else:
                rest+=r.item((i,k))
        if new_epoch!=epoch:
            old_epoch=epoch
            epoch=new_epoch
            c1=params[0:no_epochs][epoch]
            c2=params[no_epochs:2*no_epochs][epoch]
            mig12 = params[(2 * no_epochs):(3 * no_epochs)][epoch]
            mig21 = params[(3 * no_epochs):(4 * no_epochs)][epoch]
            if mig12!=0 or mig21!=0:  #in this case we can use diffusion.
                Qgamma=array([[-2*mig12-c1, 2*mig12,0],[mig21,-mig21-mig12, mig12], [0,2*mig21,-2*mig21-c2]])
                QgammaA=array([[-2*mig12-c1, 2*mig12,0,c1],[mig21,-mig21-mig12, mig12,0], [0,2*mig21,-2*mig21-c2,c2],[0,0,0,0]])
                w,V=eig(Qgamma)
                Vinv=inv(V)
                emissum=0
                normsum=0
                A=3
                middle=0
                left=0
                right=0
                for i in range(3):
                    for k in range(2):
                        normsum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - 1.0)/w[i]
                        left+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*( exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                        1.0 ) / (w[i]-8.0/3.0)    )
                        normsum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - 1.0)/w[i]
                        middle+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*( exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                        1.0 ) / (w[i]-8.0/3.0)    )
                        normsum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - 1.0)/w[i]
                        right+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(   exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                        1.0 ) / (w[i]-8.0/3.0)    )
                emissum=middle+left+right
            else:
                #if we can't use the integrals the two either belongs to the same population or different populations. If they belong to the same it is exponentially distributed.
                emissum=0
                normsum=0
                emissum+=lls*c1/(8.0/3.0+c1)*exp(-8.0/3.0*(breaklatest+offset))*(1.0-exp(-(break_points[j+1]-break_points[j])*(8.0/3.0+c1)) )
                normsum+=lls*(1.0-exp(-c1*(break_points[j+1]-break_points[j]))) 
#                emissum+=lrs*0
 #               normsum+=lrs*0
                emissum+=rrs*c2/(8.0/3.0+c2)*exp(-8.0/3.0*(breaklatest+offset))*(1.0-exp(-(break_points[j+1]-break_points[j])*(8.0/3.0+c2)) )
                normsum+=rrs*(1.0-exp(-c2*(break_points[j+1]-break_points[j]))) 
                
#            print str((left,middle,right))
#            print str((left/lls,middle/lrs,right/rrs))
            
            
        else:
            if mig12!=0 or mig21!=0:
                emissum=0
                normsum=0
                A=3
                for i in range(3):
                    for k in range(2):
                        normsum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - 1.0)/w[i]
                        emissum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                        1.0 ) / (w[i]-8.0/3.0)    )
                        normsum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - 1.0)/w[i]
                        emissum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                        1.0 ) / (w[i]-8.0/3.0)    )
                        normsum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp((break_points[j+1]-breaklatest)*w[i]) - 1.0)/w[i]
                        emissum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp(-8.0/3.0*(breaklatest+offset))*(exp((break_points[j+1]-breaklatest)*(w[i]-8.0/3.0)) - 
                                                                        1.0 ) / (w[i]-8.0/3.0)    )
        
               
            else:
                emissum=0
                normsum=0
                emissum+=lls*c1/(8.0/3.0+c1)*exp(-8.0/3.0*(breaklatest+offset))*(1.0-exp(-(break_points[j+1]-breaklatest)*(8.0/3.0+c1)))
                normsum+=lls*(1.0-exp(-c1*(break_points[j+1]-breaklatest))) 
#               emissum+=lrs*0
#               normsum+=lrs*0
                emissum+=rrs*c2/(8.0/3.0+c2)*exp(-8.0/3.0*(breaklatest+offset))*(1.0-exp(-(break_points[j+1]-breaklatest)*(8.0/3.0+c2)) )
                normsum+=rrs*(1.0-exp(-c2*(break_points[j+1]-breaklatest))) 
                #emissum=emissum/(lls+rrs)*(1-rest)*(lls+lrs+rrs)/(1-rest)
        #normsum=(restafter-rest)#this is P(T\in [t_i, t_i+1])
 #       print "e4 "+str(normsum) 
         
#         if emissum<0 or normsum<=0 or emissum>normsum:
#             print "Error in emission matrix 3 " + "calculating row number " + str(j) +"."
#             print "coming from epoch "+str(old_epoch)+" and going to epoch "+str(epoch)
#             print "from the transition probabilities we derive, lls="+str(lls)+", lrs="+str(lrs)+", rrs="+str(rrs)+"."
#             print "emissum "+str(emissum)
#             print "normsum "+str(normsum)
#             print "guilty parameters " + str(params)
#             print "guilty break points " + str(break_points)
#             print "guilty (transformed) intervals " + str(intervals) 
#        print "rest-restbefore="+str(rest-restbefore)
 #       restbefore=rest
#        print "emissum="+str(emissum)
        emission_probabilities[j+ctmc_postpone,0]=0.25+0.75*emissum/normsum
        emission_probabilities[j+ctmc_postpone,1]=0.75-0.75*emissum/normsum
        emission_probabilities[j+ctmc_postpone,2]=1.0
        old_epoch=epoch
#         print "j "+str(j)
#         print "sum("+str([lls,lrs,rrs,rest])+")="+str(sum([lls,lrs,rrs,rest]))
#         print "lls "+str(lls)
#         print "lrs "+str(lrs)
#         print "rrs "+str(rrs)
#        print "rest "+str(rest)
    emissum=0
    normsum=0
    j=len(break_points)-1
    breaklatest=break_points[j]
    if mig12!=0 or mig21!=0:
        A=3
        for i in range(3):
            for k in range(2):
                normsum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(-1.0)/w[i]
                emissum+=lls*V[0,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp(-8.0/3.0*(breaklatest+offset))*( - 1.0 ) / (w[i]-8.0/3.0))
                
                normsum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(-1.0)/w[i]
                emissum+=lrs*V[1,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp(-8.0/3.0*(breaklatest+offset))*( - 1.0 ) / (w[i]-8.0/3.0))
                
                normsum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(-1.0)/w[i]
                emissum+=rrs*V[2,i]*Vinv[i,k*2]*QgammaA[k*2,A]*(exp(-8.0/3.0*(breaklatest+offset))*( - 1.0 ) / (w[i]-8.0/3.0))
    else:
                emissum=0
                normsum=0
                emissum+=lls*c1/(8.0/3.0+c1)*exp(-8.0/3.0*(breaklatest+offset))*(1.0)
                normsum+=lls*(1.0) 
#               emissum+=lrs*0
#               normsum+=lrs*0
                emissum+=rrs*c2/(8.0/3.0+c2)*exp(-8.0/3.0*(breaklatest+offset))*(1.0)
                normsum+=rrs*(1.0) 
    

    emission_probabilities[j+ctmc_postpone,0]=0.25+0.75*emissum/normsum
    emission_probabilities[j+ctmc_postpone,1]=0.75-0.75*emissum/normsum
    emission_probabilities[j+ctmc_postpone,2]=1.0
    
        
    #print printPyZipHMM(emission_probabilities)
    return emission_probabilities

class ExpMatrix:
    
    MATRIX_EXP=1
    NORMAL_EXP=0
    
    def __init__(self, c1, c2, mig12, mig21):
        if mig12!=0 or mig21!=0:
            self.Qgamma=array([[-2*mig12-c1, 2*mig12,0],[mig21,-mig21-mig12, mig12], [0,2*mig21,-2*mig21-c2]])
            self.QgammaA=array([[-2*mig12-c1, 2*mig12,0,c1],[mig21,-mig21-mig12, mig12,0], [0,2*mig21,-2*mig21-c2,c2],[0,0,0,0]])
            self.w,self.V=eig(self.Qgamma)
            self.Vinv=inv(self.V)
            self.status=self.MATRIX_EXP
        else:
            self.c1,self.c2=c1,c2
            self.status=self.NORMAL_EXP

    def emissAndNorm(self, lls,lrs,rrs, break_new, break_latest, Cs, offset=0.0):
        if self.status==self.MATRIX_EXP:
            emissums=[0]*len(Cs)
            normsum=0
            A=3
            for i in range(3):
                for k in range(2):
                    normsum+=rrs*self.V[2,i]*self.Vinv[i,k*2]*self.QgammaA[k*2,A]*(exp((break_new-break_latest)*self.w[i]) - 1.0)/self.w[i]
                    normsum+=lls*self.V[0,i]*self.Vinv[i,k*2]*self.QgammaA[k*2,A]*(exp((break_new-break_latest)*self.w[i]) - 1.0)/self.w[i]
                    normsum+=lrs*self.V[1,i]*self.Vinv[i,k*2]*self.QgammaA[k*2,A]*(exp((break_new-break_latest)*self.w[i]) - 1.0)/self.w[i]
                    for n,c in enumerate(Cs):
                        emissums[n]+=lls*self.V[0,i]*self.Vinv[i,k*2]*self.QgammaA[k*2,A]*(exp(c*(break_latest+offset))*(exp((break_new-break_latest)*(self.w[i]+c)) - 
                                                                        1.0 ) / (self.w[i]+c)    )
                        
                        emissums[n]+=lrs*self.V[1,i]*self.Vinv[i,k*2]*self.QgammaA[k*2,A]*(exp(c*(break_latest+offset))*(exp((break_new-break_latest)*(self.w[i]+c)) - 
                                                                        1.0 ) / (self.w[i]+c)    )
                        
                        emissums[n]+=rrs*self.V[2,i]*self.Vinv[i,k*2]*self.QgammaA[k*2,A]*(exp(c*(break_latest+offset))*(exp((break_new-break_latest)*(self.w[i]+c)) - 
                                                                        1.0 ) / (self.w[i]+c)    )
                        
            return emissums, normsum
        
               
        else:
            emissums=[0]*len(Cs)
            normsum=0
            
            normsum+=lls*(1.0-exp(-self.c1*(break_new-break_latest))) 
            normsum+=rrs*(1.0-exp(-self.c2*(break_new-break_latest)))
            
            
            
            for n,c in enumerate(Cs):
                emissums[n]+=lls*self.c1/(-c+self.c1)*exp(c*(break_latest+offset))*(1.0-exp((break_new-break_latest)*(c-self.c1)))
                emissums[n]+=rrs*self.c2/(-c+self.c2)*exp(c*(break_latest+offset))*(1.0-exp((break_new-break_latest)*(c-self.c2)))
            return emissums, normsum
        
    def emissAndNormLast(self, lls, lrs, rrs, break_latest, Cs,offset=0.0):
        if self.status==self.MATRIX_EXP:
            emissums=[0]*len(Cs)
            normsum=0
            A=3
            for i in range(3):
                for k in range(2):
                    normsum+=lls*self.V[0,i]*self.Vinv[i,k*2]*self.QgammaA[k*2,A]*(-1.0)/self.w[i]
                    normsum+=lrs*self.V[1,i]*self.Vinv[i,k*2]*self.QgammaA[k*2,A]*(-1.0)/self.w[i]
                    normsum+=rrs*self.V[2,i]*self.Vinv[i,k*2]*self.QgammaA[k*2,A]*(-1.0)/self.w[i]
                    for n,c in enumerate(Cs):
                        emissums[n]+=lls*self.V[0,i]*self.Vinv[i,k*2]*self.QgammaA[k*2,A]*(exp(c*(break_latest+offset))*( - 1.0 ) / (self.w[i]+c))
                        
                        
                        emissums[n]+=lrs*self.V[1,i]*self.Vinv[i,k*2]*self.QgammaA[k*2,A]*(exp(c*(break_latest+offset))*( - 1.0 ) / (self.w[i]+c))
                        
                        
                        emissums[n]+=rrs*self.V[2,i]*self.Vinv[i,k*2]*self.QgammaA[k*2,A]*(exp(c*(break_latest+offset))*( - 1.0 ) / (self.w[i]+c))
        else:
            emissums=[0]*len(Cs)
            normsum=lls+rrs
            for n,c in enumerate(Cs):
                emissums[n]+=lls*self.c1/(-c+self.c1)*exp(c*(break_latest+offset))*(1.0)
    #               emissum+=lrs*0
    #               normsum+=lrs*0
                emissums[n]+=rrs*self.c2/(-c+self.c2)*exp(c*(break_latest+offset))*(1.0)
        return emissums, normsum
        
def extract_starting_positions(states):
    leftleft=[]
    leftright=[]
    rightright=[]
    restIndexes=[]
    for state,number in states.items():
        coalesced=False 
        position1=-1
        position2=-1
        for branch, (b1,_) in state:#we only do this marginally for pop1
            if 1 in b1:
                position1=branch
                if 2 in b1:
                    coalesced=True
            if 2 in b1:
                position2=branch
        if not coalesced:
            if position1==1 and position2==1:
                leftleft.append(number)
#                 print "ll "+str(state)
            elif position1==2 and position2==2:
                rightright.append(number)
#                print "rr "+str(state)
            elif position1==position2:#this is used for initial_migration_model and variable_migration_model_with_ancestral which has 0 as the ancestral population
                leftleft.append(number)
            elif position1!=-1 and position2!=-1:
                leftright.append(number)

                #pass#print "lr "+str(state)
#            else:
#                print "non1 "+str(state)
        else:
            restIndexes.append(number) #this is not empty because some states have not coalesced
    
    return leftleft, leftright, rightright,restIndexes

def getProbabilitiesOfPositions(up_to, initial_state, leftleft,leftright,rightright):
    lls=0 #P(B_ti=left) that is the probability that both ancestor-lines of the same position in the two genomes are in the left population
    lrs=0 #P(B_ti=both)
    rrs=0 #P(B_ti=right)
    rest=0 #probability that they have coalesced.
    for state in range(up_to.shape[1]):
        addOn=up_to.item((initial_state,state))
        if state in leftleft:
            lls+=addOn
        elif state in rightright:
            rrs+=addOn
        elif state in leftright:
            lrs+=addOn
        else:
            rest+=addOn
    return lls,lrs,rrs,rest
        
def emission_matrix7(break_points, params,intervals,  ctmc_system,offset=0.0, ctmc_postpone=0):
    """
    Like emission_matrix6 except the calculations is a little simpler.
    """
 
    no_epochs=len(intervals)
    epoch=-1
    intervals=cumsum(array(intervals))
    emission_probabilities = Matrix(len(break_points)+ctmc_postpone, 3)
    for j in xrange(ctmc_postpone):
        emission_probabilities[j,0]=1.0     #just default values
        emission_probabilities[j,1]=0.0
        emission_probabilities[j,2]=1.0

    for j in range(len(break_points)):
        #breaklatest=break_points[j]
        #new epoch is the epoch that corresponds to interval j
        new_epoch=bisect(intervals, j)
        if new_epoch!=epoch:
            epoch=new_epoch
            states=ctmc_system.get_state_space(j+ctmc_postpone).states
            leftleft,leftright,rightright,_=extract_starting_positions(states)
            c1=params[0:no_epochs][epoch]
            c2=params[no_epochs:2*no_epochs][epoch]
            mig12 = params[(2 * no_epochs):(3 * no_epochs)][epoch]
            mig21 = params[(3 * no_epochs):(4 * no_epochs)][epoch]

            Expms=ExpMatrix(c1, c2, mig12, mig21)
            
        
        lls,lrs,rrs,_ = getProbabilitiesOfPositions(up_to= ctmc_system.up_to(j+ctmc_postpone), 
                                                    initial_state=ctmc_system.initial_ctmc_state,
                                                    leftleft=leftleft, leftright=leftright,
                                                    rightright=rightright)
        #print lls,lrs,rrs
        
        if j==(len(break_points)-1): #if we are at the last interval
            emissum, normsum = Expms.emissAndNormLast(lls=lls,lrs=lrs,rrs=rrs, break_latest=break_points[j], Cs=[-8.0/3.0], offset=offset)
        else:
            emissum, normsum = Expms.emissAndNorm(lls=lls,lrs=lrs,rrs=rrs, break_new=break_points[j+1], break_latest=break_points[j], Cs=[-8.0/3.0], offset=offset)
        
        if normsum<1e-300 or emissum[0]<1e-300:
            if emissum[0]>1e-300:
                print "----------------------------normsum=0 even though emissum>0---------------------------------------"
            emission_probabilities[j+ctmc_postpone,0]=1.0
            emission_probabilities[j+ctmc_postpone,1]=0.0
        else:
            emission_probabilities[j+ctmc_postpone,0]=0.25+0.75*emissum[0]/normsum
            emission_probabilities[j+ctmc_postpone,1]=0.75-0.75*emissum[0]/normsum
        emission_probabilities[j+ctmc_postpone,2]=1.0
    
    return emission_probabilities


def classifyState(observed_states, W):
    """
    Input is the tuple of nucleotides observed and two nucleotides R and W.
    """
    X,Y,Z=observed_states
    return X==W, Y==W, W==Z

LettersToComputation={}
Letters=["A","C","G","T"]
priorOnLetters=[0.25]*4
nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
from collections import defaultdict
computationToLetters = defaultdict(list)
for a in Letters:
    for b in Letters:
        for c in Letters:
            for d in Letters:
                LettersToComputation[a+b+c+d]=classifyState((a,b,c),d)
                computationToLetters[classifyState((a,b,c),d)].append(a+b+c+d)
computationToLetters=dict(computationToLetters)
                    
normalTerms=[[0.25,0.75], [0.75,-0.75]]

def getJCtermsAsFunction(emissums,normsum,outgroup):
    def what(indexes,equals):
        groupSpecicTerm=[[1,1],[1,1],[1,exp(-8.0/3.0*outgroup)]]
        res=1
        for n,(i,e) in enumerate(zip(indexes,equals)):
            res*=normalTerms[e][i]*groupSpecicTerm[n][i]
        numberOf4thirdsTinExponent=-indexes[0]-indexes[1]+indexes[2]
        if numberOf4thirdsTinExponent==-2:
            res*=emissums[0]/normsum
        elif numberOf4thirdsTinExponent==-1:
            res*=emissums[1]/normsum
        elif numberOf4thirdsTinExponent==1:
            res*=emissums[2]/normsum
        return res
    return what

def getVectorOfCombinations(emissums, normsum, break_latest, break_new, outgroup):
    """
    This function takes the integrals E[e^-4/3*T, T in interval], E[e^-8/3*T, T in interval] and E[e^4/3*T, T in interval] in emissums and P(T in interval) as normsum.
    It then calculates the integrals P(X=W, Y=W, Z=R, R=W | T in interval) and saves them in baseIntegrals. It then calculates the  integrals
    P(X,Y,Z | T in interval). At last it puts a 1.0 at the last entry because it is the default value of observing N-observation.
    """
    baseIntegrals={bool_combi:0 for bool_combi in computationToLetters.keys()}    #going to be P(X=W, Y=W, Z=R, R=W | T in interval)
    pbools=getJCtermsAsFunction(emissums, normsum, outgroup)
    for XeqW in range(2):
        for YeqW in range(2):
            for WeqZ in range(2):
                for i in range(2):  #the chosen terms of 1/4,3/4,-3/4 for XeqW 
                    for j in range(2):  #the chosen terms of 1/4,3/4,-3/4 for YeqW 
                        for k in range(2):  #the chosen terms of 1/4,3/4,-3/4 for WeqZ
                            baseIntegrals[(XeqW==0,YeqW==0, WeqZ==0)]+=pbools(indexes=(i,j,k), equals=(XeqW,YeqW,WeqZ))
                                    #print (XeqW==1,YeqW==1, ReqW==1, ReqZ==1), baseIntegrals[(XeqW==1,YeqW==1, ReqW==1, ReqZ==1)]
    resVector=[0.0]*65
    assert_almost_equal(sum(baseIntegrals.itervalues()),1.0)
    for letters,comput_name in LettersToComputation.items():
        ivec=[nuc_map[l] for l in letters]
        resVector[ivec[0]+ivec[1]*4+ivec[2]*16]+= baseIntegrals[comput_name]/float(len(computationToLetters[comput_name]))  #Here we see that 
    resVector[-1]=1.0 
    return resVector
                
        
def emission_matrix8(break_points, params, outgroup, intervals,  ctmc_system, offset=0.0, ctmc_postpone=0):
    """
    Same concept as the others but now with an outgroup meaning that there will be a more complicated evaluation of the integral.
    It is assumed that break_points[-1]<outgroup. It should be controlled elsewhere.
    """
    
    if break_points[-1]<outgroup:
        i=bisect(break_points, outgroup)
 
    no_epochs=len(intervals)
    epoch=-1
    intervals=cumsum(array(intervals))
    emission_probabilities = Matrix(len(break_points)+ctmc_postpone, 65)
    for j in xrange(ctmc_postpone):
        emission_probabilities[j,0]=1.0     #just default values
        for i in xrange(63):
            emission_probabilities[j,i+1]=0.0
        emission_probabilities[j,64]=1.0

    for j in range(len(break_points)):
        #breaklatest=break_points[j]
        #new epoch is the epoch that corresponds to interval j
        new_epoch=bisect(intervals, j)
        if new_epoch!=epoch:
            epoch=new_epoch
            states=ctmc_system.get_state_space(j+ctmc_postpone).states
            leftleft,leftright,rightright,_=extract_starting_positions(states)
            c1=params[0:no_epochs][epoch]
            c2=params[no_epochs:2*no_epochs][epoch]
            mig12 = params[(2 * no_epochs):(3 * no_epochs)][epoch]
            mig21 = params[(3 * no_epochs):(4 * no_epochs)][epoch]

            Expms=ExpMatrix(c1, c2, mig12, mig21)
            
        
        lls,lrs,rrs,_ = getProbabilitiesOfPositions(up_to= ctmc_system.up_to(j+ctmc_postpone), 
                                                    initial_state=ctmc_system.initial_ctmc_state,
                                                    leftleft=leftleft, leftright=leftright,
                                                    rightright=rightright)
        #print lls,lrs,rrs
        
        exponents=[-8.0/3.0, -4.0/3.0, 4.0/3.0]
        
        if j==(len(break_points)-1): #if we are at the last interval
            emissums, normsum = Expms.emissAndNorm(lls=lls,lrs=lrs,rrs=rrs, break_new=outgroup, break_latest=break_points[j], Cs=exponents, offset=offset)
            addrow= getVectorOfCombinations(emissums, normsum, break_latest=break_points[j], break_new=outgroup, outgroup=outgroup)
        else:
            emissums, normsum = Expms.emissAndNorm(lls=lls,lrs=lrs,rrs=rrs, break_new=break_points[j+1], break_latest=break_points[j], Cs=exponents, offset=offset)
            addrow= getVectorOfCombinations(emissums, normsum, break_latest=break_points[j], break_new=break_points[j+1], outgroup=outgroup)
        
#         print "e[-8/3T]=",emissums[0]/normsum
#         print "e[-4/3T]=",emissums[1]/normsum
#         print "e[4/3T]=",emissums[2]/normsum
        
        for n,a in enumerate(addrow):
            emission_probabilities[j+ctmc_postpone,n]=a
        
    
    return emission_probabilities


def main():
    """Test"""

    time_points = [0.0, 8.3381608939051073e-05, 0.0001743533871447778, 0.00027443684570176033, 0.00038566248081198473, 0.0005108256237659907, 0.00065392646740666392, 0.00082098055206983008, 0.0010216512475319816, 0.0012729656758128879, 0.0015213904954082308, 0.001775529617693222, 0.0020422770961952179, 0.0023289318050702446, 0.0026446563966777419, 0.0030025305752202627, 0.0034235104431770335, 0.0039458917535297169, 0.0046542150194674647, 0.0058142576782905598]
    rates=gamma(shape=2, scale=1.0/0.001, size=len(time_points))
    from break_points2 import gamma_break_points
    bre=gamma_break_points(20,beta1=0.001*1,alpha=2,beta2=0.001333333*1, tenthsInTheEnd=5)
    
    print "jetzt"
    from variable_migration_model_with_ancestral import VariableCoalAndMigrationRateAndAncestralModel
    
    substime_first_change=0.0005
    substime_second_change=0.0010
    substime_third_change=0.0030
    def time_modifier():
        return [(5,substime_first_change),(10,substime_second_change)]
    cd=VariableCoalAndMigrationRateAndAncestralModel(VariableCoalAndMigrationRateAndAncestralModel.INITIAL_22, intervals=[5,5,5], breaktimes=1.0,breaktail=3,time_modifier=time_modifier, outgroup=True)
    parameters=[1000,1000,1000,  1000,1000,1000,    0,5000,0,    0,100,0,    0.40]
    cd.outmax=0.02
    ctmc_system= cd.build_ctmc_system(*parameters)
    br=ctmc_system.break_points
    coals1,coals2,migs1,migs2,rho,_=cd.unpack_parameters(parameters)
    assert sum(migs1)+sum(migs2)>0, "migration rates can not all be 0 and any can not be negative"
    indexOfFirstNonZero=min([n for n,(r,s) in enumerate(zip(migs1,migs2)) if r>0 or s>0])
    indexOfFirstNonZeroMeasuredInBreakPoints=cumsum(cd.intervals)[indexOfFirstNonZero-1]
    reducedParameters=concatenate((coals1[indexOfFirstNonZero:],coals2[indexOfFirstNonZero:],migs1[indexOfFirstNonZero:],migs2[indexOfFirstNonZero:],[rho]))
    emission_probs=emission_matrix8(break_points=br[indexOfFirstNonZeroMeasuredInBreakPoints:], params=reducedParameters, intervals=cd.intervals[indexOfFirstNonZero:], 
                                                ctmc_system=ctmc_system, offset=0,ctmc_postpone=indexOfFirstNonZeroMeasuredInBreakPoints, outgroup=0.02)
        
    
    
    otherparams=[2.0/0.000575675566598,2.0/0.00221160347741,2.0/0.000707559309234,2.0/0.00125938374711,2.0/0.00475558231719,2.0/0.000829398438542,2.0/0.000371427015082,2.0/0.000320768239201,1,0,105.490882058,131.840288312,0,0,123.259131284,101.646109897,1.42107787743]
    #print rates
    #print printPyZipHMM(emission_matrix(coalescence_points(time_points,rates)))
    #print printPyZipHMM(emission_matrix2(time_points, rates))
    print printPyZipHMM(emission_probs)
    


if __name__ == '__main__':
    main()
