"""Code for constructing and optimizing the HMM for an isolation model.
"""

from numpy import zeros, matrix
from numpy.testing import assert_almost_equal

from IMCoalHMM.state_spaces import Isolation, make_rates_table_isolation
from IMCoalHMM.state_spaces import Single, make_rates_table_single
from IMCoalHMM.CTMC import make_ctmc
from IMCoalHMM.transitions import CTMCSystem, projection_matrix, compute_upto, compute_between
from IMCoalHMM.emissions import coalescence_points
from IMCoalHMM.break_points import exp_break_points
from IMCoalHMM.model import Model

from multiprocessing import Pool, cpu_count


## Code for computing HMM transition probabilities ####################

# The way multiprocessing works means that we have to define this class for mapping in parallel
# and we have to define the processing pool after we define the class, or it won't be able to see
# it in the sub-processes. It breaks the flow of the code, but it is necessary.

class ComputeThroughInterval(object):
    def __init__(self, single, break_points):
        self.single = single
        self.break_points = break_points
    def __call__(self, i):
        return self.single.probability_matrix(self.break_points[i + 1] - self.break_points[i])


COMPUTATION_POOL = Pool(cpu_count())


def _compute_through(single, break_points):
    """Computes the matrices for moving through an interval"""
    no_states = len(break_points)

    # Construct the transition matrices for going through each interval
    through = COMPUTATION_POOL.map(ComputeThroughInterval(single, break_points),  range(no_states-1))

    # As a hack we set up a pseudo through matrix for the last interval that
    # just puts all probability on ending in one of the end states. This
    # simplifies the HMM transition probability code as it avoids a special case
    # for the last interval.
    # noinspection PyCallingNonCallable
    pseudo_through = matrix(zeros((len(single.state_space.states),
                                   len(single.state_space.states))))
    pseudo_through[:, single.state_space.end_states[0]] = 1.0
    through.append(pseudo_through)

    return through


def _compute_upto0(isolation, single, break_points):
    """Computes the probability matrices for moving to time zero."""

    def state_map(state):
        return frozenset([(0, nucs) for (_, nucs) in state])

    projection = projection_matrix(isolation.state_space, single.state_space, state_map)
    return isolation.probability_matrix(break_points[0]) * projection


class IsolationCTMCSystem(CTMCSystem):
    """Wrapper around CTMC transition matrices for the isolation model."""

    def __init__(self, isolation_ctmc, ancestral_ctmc, break_points):
        """Construct all the matrices and cache them for the
        method calls.

        :param isolation_ctmc: CTMC for the isolation phase.
        :type isolation_ctmc: IMCoalHMM.CTMC.CTMC
        :param ancestral_ctmc: CTMC for the ancestral population.
        :type ancestral_ctmc: IMCoalHMM.CTMC.CTMC
        :param break_points: List of break points between intervals.
        :type break_points: list[int]
        """

        super(IsolationCTMCSystem, self).__init__(no_hmm_states=len(break_points),
                                                  initial_ctmc_state=isolation_ctmc.state_space.i12_index)

        self.ancestral_ctmc = ancestral_ctmc
        self.through_ = _compute_through(ancestral_ctmc, break_points)
        self.upto_ = compute_upto(_compute_upto0(isolation_ctmc, ancestral_ctmc, break_points), self.through_)
        self.between_ = compute_between(self.through_)

    def get_state_space(self, i):
        """Return the state space for interval i. In this case it is always the
        ancestral state space.

        :rtype: Single
        """
        return self.ancestral_ctmc.state_space


## Class that can construct HMMs ######################################
class IsolationModel(Model):
    """Class wrapping the code that generates an isolation model HMM."""

    def __init__(self, no_hmm_states):
        """Construct the model.

        This builds the state spaces for the CTMCs but not the matrices for the
        HMM since those will depend on the rate parameters."""
        super(IsolationModel, self).__init__()
        self.no_hmm_states = no_hmm_states
        self.isolation_state_space = Isolation()
        self.single_state_space = Single()

    def emission_points(self, split_time, coal_rate, _):
        """Points to emit from."""
        break_points = exp_break_points(self.no_hmm_states, coal_rate, split_time)
        return coalescence_points(break_points, coal_rate)

    def build_ctmc_system(self, split_time, coal_rate, recomb_rate):
        """Construct CTMC system."""
        # We assume here that the coalescence rate is the same in the two
        # separate populations as it is in the ancestral. This is not necessarily
        # true but it worked okay in simulations in Mailund et al. (2011).
        isolation_rates = make_rates_table_isolation(coal_rate, coal_rate, recomb_rate)
        isolation_ctmc = make_ctmc(self.isolation_state_space, isolation_rates)
        single_rates = make_rates_table_single(coal_rate, recomb_rate)
        single_ctmc = make_ctmc(self.single_state_space, single_rates)
        break_points = exp_break_points(self.no_hmm_states, coal_rate, split_time)
        return IsolationCTMCSystem(isolation_ctmc, single_ctmc, break_points)


def main():
    """Test"""

    model = IsolationModel(4)
    pi, trans_probs, emis_probs = model.build_hidden_markov_model((1.0, 0.5, 4e-4))

    no_states = pi.getHeight()
    assert no_states == 4

    pi_sum = 0.0
    for row in xrange(no_states):
        pi_sum += pi[row, 0]
    assert_almost_equal(pi_sum, 1.0)

    assert no_states == trans_probs.getWidth()
    assert no_states == trans_probs.getHeight()

    trans_sum = 0.0
    for row in xrange(no_states):
        for col in xrange(no_states):
            trans_sum += trans_probs[row, col]
    assert_almost_equal(trans_sum, no_states)

    print 'Done'


if __name__ == '__main__':
    main()
