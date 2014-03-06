"""Code for constructing and optimizing the HMM for an isolation model.
"""

from numpy import zeros, matrix
from numpy.testing import assert_almost_equal

from IMCoalHMM.state_spaces import Isolation, make_rates_table_isolation
from IMCoalHMM.state_spaces import Single, make_rates_table_single
from IMCoalHMM.CTMC import CTMC
from IMCoalHMM.transitions import CTMCSystem
from IMCoalHMM.emissions import coalescence_points
from IMCoalHMM.break_points import exp_break_points
from IMCoalHMM.model import Model


## Code for computing HMM transition probabilities ####################
def _compute_through(single, break_points):
    """Computes the matrices for moving through an interval"""
    no_states = len(break_points)

    # Construct the transition matrices for going through each interval
    through = [single.probability_matrix(break_points[i + 1] - break_points[i])
               for i in xrange(no_states - 1)]

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


def _compute_upto(isolation, single, break_points, through):
    """Computes the probability matrices for moving from time zero up to,
    but not through, interval i."""

    no_states = len(break_points)

    # Projection matrix needed to go from the isolation to the single
    # state spaces
    # noinspection PyCallingNonCallable
    projection = matrix(zeros((len(isolation.state_space.states),
                               len(single.state_space.states))))
    for state, isolation_index in isolation.state_space.states.items():
        ancestral_state = frozenset([(0, nucs) for (_, nucs) in state])
        ancestral_index = single.state_space.states[ancestral_state]
        projection[isolation_index, ancestral_index] = 1.0

    # We handle the first state as a special case because of the isolation
    # interval
    upto = [None] * no_states
    upto[0] = isolation.probability_matrix(break_points[0]) * projection
    for i in xrange(1, no_states):
        upto[i] = upto[i - 1] * through[i - 1]

    return upto


def _compute_between(single, break_points):
    """Computes the matrices for moving from the end of interval i
    to the beginning of interval j."""

    no_states = len(break_points)
    # Transitions going from the endpoint of interval i to the entry point
    # of interval j
    return dict(
        ((i, j), single.probability_matrix(break_points[j] - break_points[i + 1]))
        for i in xrange(no_states - 1)
        for j in xrange(i + 1, no_states)
    )


class IsolationCTMCSystem(CTMCSystem):
    """Wrapper around CTMC transition matrices for the isolation model."""

    def __init__(self, isolation_ctmc, ancestral_ctmc, break_points):
        """Construct all the matrices and cache them for the
        method calls.

        :param isolation_ctmc: CTMC for the isolation phase.
        :type isolation_ctmc: CTMC
        :param ancestral_ctmc: CTMC for the ancestral population.
        :type ancestral_ctmc: CTMC
        :param break_points: List of break points between intervals.
        :type break_points: list[int]
        """

        super(IsolationCTMCSystem, self).__init__(no_hmm_states=len(break_points),
                                                  initial_ctmc_state=isolation_ctmc.state_space.i12_index)

        self.ancestral_ctmc = ancestral_ctmc
        self.through_ = _compute_through(ancestral_ctmc, break_points)
        self.upto_ = _compute_upto(isolation_ctmc, ancestral_ctmc, break_points, self.through_)
        self.between_ = _compute_between(ancestral_ctmc, break_points)

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
        isolation_ctmc = CTMC(self.isolation_state_space, isolation_rates)
        single_rates = make_rates_table_single(coal_rate, recomb_rate)
        single_ctmc = CTMC(self.single_state_space, single_rates)
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
