'''Code for computing emission probabilities for pairwise sequence alignments.

The code uses a truncated exponential to get the coalescence time point
and a Jukes-Cantor for emission probabilities, with a pseudo emission
probability of 1 for missing data.
'''

from math import exp

def truncated_exp_midpoint(t1, t2, rate):
    '''Calculates the mean coalescence point between t1 and t2
    from a truncated exponential distribution.'''
    delta_t = t2 - t1
    return t1 + 1.0/rate - (delta_t*exp(-delta_t*rate))/(1-exp(-delta_t*rate))
    
def exp_midpoint(t, rate):
    '''Calculates the mean coalescence point after t
    from an exponential distribution.'''
    return t + 1.0/rate

def coalescence_points(break_points, rate):
    '''Calculates the mean coalescence times (given the rate) between
    each time break point and after the last break point.'''
    result = []
    for i in xrange(1,len(break_points)):
        t = truncated_exp_midpoint(break_points[i-1], break_points[i], rate)
        result.append(t)
    result.append(exp_midpoint(break_points[-1], rate))
    return result

def main():
    '''Test'''

    time_points = [1,2,3,4]
    for i in xrange(1,len(time_points)):
        print time_points[i-1], 
        print truncated_exp_midpoint(time_points[i-1], time_points[i], 1.0),
        print time_points[i]
    print time_points[-1], exp_midpoint(time_points[-1], 1.0)
    print
    print coalescence_points(time_points, 1.0)
    print

if __name__ == '__main__':
    main()