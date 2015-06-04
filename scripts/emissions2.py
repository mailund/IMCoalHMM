"""Code for computing emission probabilities for pairwise sequence alignments.

The code uses a truncated exponential to get the coalescence time point
and a Jukes-Cantor for emission probabilities, with a pseudo emission
probability of 1 for missing data.
"""

from math import exp
from numpy.random import gamma
from numpy import cumsum,array
from pyZipHMM import Matrix
from bisect import bisect
from numpy.linalg import eig, inv,det


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
            
              
            
            
            


def main():
    """Test"""

    time_points = [0.0, 8.3381608939051073e-05, 0.0001743533871447778, 0.00027443684570176033, 0.00038566248081198473, 0.0005108256237659907, 0.00065392646740666392, 0.00082098055206983008, 0.0010216512475319816, 0.0012729656758128879, 0.0015213904954082308, 0.001775529617693222, 0.0020422770961952179, 0.0023289318050702446, 0.0026446563966777419, 0.0030025305752202627, 0.0034235104431770335, 0.0039458917535297169, 0.0046542150194674647, 0.0058142576782905598]
    rates=gamma(shape=2, scale=1.0/0.001, size=len(time_points))
    from break_points2 import gamma_break_points
    bre=gamma_break_points(20,beta1=0.001*1,alpha=2,beta2=0.001333333*1, tenthsInTheEnd=5)
    otherparams=[2.0/0.000575675566598,2.0/0.00221160347741,2.0/0.000707559309234,2.0/0.00125938374711,2.0/0.00475558231719,2.0/0.000829398438542,2.0/0.000371427015082,2.0/0.000320768239201,127.278907998,124.475750838,105.490882058,131.840288312,137.498454174,114.216001115,123.259131284,101.646109897,1.42107787743]
    #print rates
    #print printPyZipHMM(emission_matrix(coalescence_points(time_points,rates)))
    #print printPyZipHMM(emission_matrix2(time_points, rates))
    print printPyZipHMM(emission_matrix3(bre,otherparams, [5,5,5,5]))
    


if __name__ == '__main__':
    main()
