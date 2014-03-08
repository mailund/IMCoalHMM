"""Code for computing emission probabilities for pairwise sequence alignments.

The code uses a truncated exponential to get the coalescence time point
and a Jukes-Cantor for emission probabilities, with a pseudo emission
probability of 1 for missing data.
"""

from math import exp
from pyZipHMM import Matrix


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
        return 0.25 - 0.25 * exp(-4.0 / 3 * dt)


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


def main():
    """Test"""

    time_points = [1, 2, 3, 4]
    for i in xrange(1, len(time_points)):
        print time_points[i - 1],
        print truncated_exp_midpoint(time_points[i - 1], time_points[i], 1.0),
        print time_points[i]
    print time_points[-1], exp_midpoint(time_points[-1], 1.0)
    print
    print coalescence_points(time_points, 1.0)
    print


if __name__ == '__main__':
    main()
