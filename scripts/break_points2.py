"""Code for computing break points between intervals.

"""

from scipy.stats import expon, uniform,gamma
from math import exp, log


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

def gamma_break_points(no_intervals=20, beta1=0.001,alpha=2,beta2=0.005,coveredByExp=0.90, no_statesOfExponentialCover=10,offset=0.0):
    no_statesOfGammaCover=no_intervals-no_statesOfExponentialCover
    points1= [expon.ppf((float(i) / no_statesOfExponentialCover)*coveredByExp,scale=beta1) for i in xrange(no_statesOfExponentialCover)]
    toPoint=gamma.cdf(points1[-1],scale=beta2*1.5,a=alpha)
    print toPoint
    points2= [gamma.ppf((float(i)/(no_statesOfGammaCover+1))*(1-toPoint)+toPoint, scale=beta2*1.5,a=alpha) for i in xrange(no_statesOfGammaCover+1)]
    points = points1+points2[1:]
    return points
    
    
    


def main():
    """Test"""
    print exp_break_points(5, 1.0)
    print exp_break_points(5, 2.0)
    print exp_break_points(5, 1.0, 3.0)
    print uniform_break_points(5, 1.0, 3.0)

    print psmc_break_points(20,t_max=7*4*20000*25)
    print gamma_break_points(20,beta1=0.001, beta2=float(1)/750)


if __name__ == '__main__':
    main()