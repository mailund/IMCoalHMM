'''Code for computing break points between intervals.

'''

from scipy.stats import expon
from math import exp, log

def exp_break_points(no_intervals, coal_rate, offset=0.0):
    '''Compute break points for equiprobably intervals given the
    coalescence rate. The optional parameter "offset" is added to all
    the break points and can be used for e.g. a speciation time.'''
    points = expon.ppf([float(i)/no_intervals for i in xrange(no_intervals)])
    return points/coal_rate + offset

def psmc_break_points(n = 64, Tmax = 15, offset = 0.0):
    break_points = [offset] + \
                   [offset + 0.1 * (exp(float(i)/n * log(1+10*Tmax)) - 1.0)
                    for i in xrange(1,n)]
    return break_points
    
    
def main():
    'Test'
    print exp_break_points(5, 1.0)
    print exp_break_points(5, 2.0)
    print exp_break_points(5, 1.0, 3.0)
    
    print psmc_break_points(5)
    
if __name__ == '__main__':
    main()