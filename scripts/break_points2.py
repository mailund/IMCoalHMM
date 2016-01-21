"""Code for computing break points between intervals.

"""

from scipy.stats import expon, uniform,gamma
from math import exp, log
from scipy.interpolate import interp1d


def exp_break_points(no_intervals, coal_rate, offset=0.0):
    """Compute break points for equal probably intervals given the
    coalescence rate. The optional parameter "offset" is added to all
    the break points and can be used for e.g. a speciation time.


    :param no_intervals: Number of intervals desired. The number of
    points will match this.
    :type no_intervals: int

    :param coal_rate: The coalescence rate used for specifying the
    exponential distribution the break points are taken from.
    :type coal_rate: float

    :param offset: This offset is added to all break points.
    :type offset: float

    :returns: a list of no_intervals break points
    :rtype: list
    """
    points = expon.ppf([float(i) / no_intervals for i in xrange(no_intervals)])
    return points / coal_rate + offset


def uniform_break_points(no_intervals, start, end):
    """Uniformly distributed break points between start and end.
    The break points will be placed at equal distance but starting at start and
    ending before end.

    :param no_intervals: Number of intervals desired.
    :type no_intervals: int

    :param start: Start of the interval. This will also be the first break point.
    :type start: float

    :param end: End point of the interval. This is _not_ included as a break point.
    :type end: float

    :returns: a list of no_intervals break points
    :rtype: list
    """
    points = uniform.ppf([float(i) / no_intervals for i in xrange(no_intervals)])
    return points * (end - start) + start


def psmc_break_points(no_intervals=64, t_max=15, mu=1e-9, offset=0.0):
    """Breakpoints taken from Li & Durbin (2011). These break points are placed
    with increasing length, similar to exp_break_points, but based on a max coalescence
    time t_max.

    :param no_intervals: The number of intervals/break points desired.
    :type no_intervals: int

    :param t_max: The maximal coalescence time to consider in the likelihood
    computations.
    :type t_max: float

    :param mu: Mutation rate used to scale the break points. With t_max given in
    coalescence units, mu is needed to scale it to the time unit used in CoalHMMs.
    :type mu: float

    :param offset: An offset that is added all break points. Used to insert
    an isolation model before the PSMC-like period.
    :type offset: float

    :returns: a list of no_intervals intervals.
    :rtype: list

    """
    break_points = [offset] + \
                   [offset + 0.1 * (exp(float(i) / no_intervals * log(1 + 10 * t_max * mu)) - 1.0)
                    for i in xrange(1, no_intervals)]
    return break_points


def gamma_break_points(no_intervals=20, beta1=0.001,alpha=2,beta2=0.005,coveredByExp=0.80,offset=0.0,tenthsInTheEnd=0, fixed_time_points=[]):
    """time is measured in the unit substitutions. 
    fixed_time_points is a list of 2-tuples with fixations of the break_points.
    The first coordinate of a 2-tuple is the index of the interval(end point). There are only no_intervals-1 possible choices ranging from 1 to no_intervals-1.
    The second coordinate of a 2-tuple is the time. """
    no_statesOfExponentialCover=int(no_intervals/2)
    uExp=float(no_statesOfExponentialCover)/float(no_intervals)
    no_statesOfGammaCover=no_intervals-no_statesOfExponentialCover-tenthsInTheEnd
    uGamma=float(no_statesOfGammaCover+no_statesOfExponentialCover)/float(no_intervals)
    uTenths=1.0-uGamma#-uExp
    divisionLineBetweenExpAndGamma=expon.ppf((float(no_statesOfExponentialCover-1) / no_statesOfExponentialCover)*coveredByExp, scale=beta1)
#     if fixed_time_points:
#         for interval,time in fixed_time_points:
#             if interval<no_statesOfExponentialCover and time>divisionLineBetweenExpAndGamma:
#                 divisionLineBetweenExpAndGamma=time
    divisionLineGammaCDF=gamma.cdf(divisionLineBetweenExpAndGamma,alpha, scale=beta2)
    
    divisionLineGammaTenthsInTheEnd=gamma.ppf(divisionLineGammaCDF+(1-divisionLineGammaCDF)*(float(no_statesOfGammaCover+(tenthsInTheEnd==0))/(no_statesOfGammaCover+1)),alpha, scale=beta2)
#     if fixed_time_points:
#         for interval,time in fixed_time_points:
#             if interval<no_statesOfExponentialCover+no_statesOfGammaCover and time> divisionLineGammaTenthsInTheEnd:
#                 divisionLineGammaTenthsInTheEnd=time
    tenthsCDF=gamma.cdf(divisionLineGammaTenthsInTheEnd,alpha, scale=beta2)
    
        
        #extending the 
        
        
    
    #print "divisionLineBetweenExpAndGamma "+str(divisionLineBetweenExpAndGamma)
    #print "divisionLineGammaCDF "+str(divisionLineGammaCDF)
    #print "tenthsCDF "+str(tenthsCDF)
    #print "divisionLineGammaTenthsInTheEnd "+str(divisionLineGammaTenthsInTheEnd)
    if tenthsInTheEnd:
        tenthsBasePoints=[divisionLineGammaTenthsInTheEnd]+[gamma.ppf(tenthsCDF+(1.0-tenthsCDF)*(1.0-(1.0/10.0)**(i+1)), alpha, scale=beta2) for i in range(tenthsInTheEnd)]
        tenthsXpoints=[uGamma]+[float(i+1)/float(no_intervals) for i in range(no_intervals-tenthsInTheEnd, no_intervals)]
        #print "tenthsBasePoints"+str(tenthsBasePoints)
        if fixed_time_points:
            if fixed_time_points[-1][1]>tenthsBasePoints[-1]:
                #this becomes a little hacky, but it should not happen many times
                tenthsBasePoints[-1]=max(zip(*fixed_time_points)[1])
                tenthsXpoints[-1]=tenthsXpoints[-2]+(tenthsXpoints[-1]-tenthsXpoints[-2])/2
                tenthsXpoints.append(1.0)
                tenthsBasePoints.append(tenthsBasePoints[-1]*3.5)
        #print tenthsXpoints
        #print tenthsBasePoints
        cdftenths=interp1d(tenthsBasePoints, tenthsXpoints, bounds_error=False, fill_value=1.0)
        ppftenths=interp1d(tenthsXpoints, tenthsBasePoints)

    def gamma_exp_cdf(x):
        if x < divisionLineBetweenExpAndGamma:
            return expon.cdf(x,scale=beta1)*uExp/(expon.cdf(divisionLineBetweenExpAndGamma, scale=beta1))
        if x<=divisionLineGammaTenthsInTheEnd:
            a=gamma.cdf(x,alpha, scale=beta2)
            return uExp+(uGamma-uExp)*(a-divisionLineGammaCDF)/(tenthsCDF-divisionLineGammaCDF)
        return cdftenths(x)
        

    def gamma_exp_ppf(u):
        if u< uExp:
            return expon.ppf(u/uExp*float(no_statesOfExponentialCover-1)/float(no_statesOfExponentialCover)*coveredByExp,scale=beta1)
        if u<=uGamma:
            lowerCDF=divisionLineGammaCDF
            upperCDF=tenthsCDF
            uOfTruncatedGamma=((u-uExp)/(uGamma-uExp))*(upperCDF-lowerCDF)+lowerCDF
       #     print str(u)+ " => "+ str(uOfTruncatedGamma)
            return gamma.ppf(uOfTruncatedGamma,alpha, scale=beta2)
        else:
            return float(ppftenths(u))
    points=[0.0]
    fromU,toU=0.0,1.0
    lastf=0
    t=None
    if fixed_time_points: #the first target to go for
        f,t=fixed_time_points[0]
        toU=gamma_exp_cdf(t)
    else:
        f=no_intervals
    getcount=1
    while fromU < (1.0-1e-9):
        #print "toU"+str(toU)
        for i in range(f-lastf-1):
            u=fromU+(toU-fromU)*(float(i)+1)/float(f-lastf)
            points.append(gamma_exp_ppf(u))
        fromU=toU
        lastf=f
        if t is not None:
            points.append(t)
        if getcount< len(fixed_time_points): #the first target to go for
            f,t=fixed_time_points[getcount]
            getcount+=1
            toU=gamma_exp_cdf(t)
        else:
            f=no_intervals
            toU=1.0
            t=None
    if len(points)!=no_intervals:
        ##This has only been observed when fixed_time_points are really wrong - so wrong that it is rejected by the prior. Hence, this step will be rejected elsewhere
        print fixed_time_points
        return [float(i)/float(no_intervals)*fixed_time_points[0][1] for i in range(no_intervals)]
    return points

    
    
    


def main():
    """Test"""
    print exp_break_points(5, 1.0)
    print exp_break_points(5, 2.0)
    print exp_break_points(5, 1.0, 3.0)
    print uniform_break_points(5, 1.0, 3.0)
    print "hallo"

    #print len(psmc_break_points(20,t_max=7*4*20000*25))
    #print gamma_break_points(20,beta1=0.001, alpha=2,beta2=float(1)/750)
    #b=gamma_break_points(26,beta1=0.001,alpha=2,beta2=0.001333, fixed_time_points=[(5,0.1),(18,0.5)], tenthsInTheEnd=3)
    b=gamma_break_points(15,beta1=0.001,alpha=2,beta2=0.001333, fixed_time_points=[(5,0.0005856098094),(10,0.0021)], tenthsInTheEnd=3)
    print b

    #print b
    #print len(b)
    #b=gamma_break_points(20,beta1=0.001,alpha=2,beta2=0.001333,tenthsInTheEnd=5)
    #print gamma_break_points(40, beta1=0.001,alpha=2,beta2=0.005,offset=0.0,tenthsInTheEnd=8, fixed_time_points=[(10,0.005),(20,0.01),(39,0.02)])
    #print str(len(b))+" "+str(b)


if __name__ == '__main__':
    main()