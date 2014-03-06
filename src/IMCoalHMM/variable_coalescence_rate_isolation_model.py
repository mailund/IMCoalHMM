"""Code for constructing and optimizing the HMM for an isolation model.
"""

from numpy import zeros, identity, matrix

from IMCoalHMM.isolation_model import Isolation, make_rates_table_isolation
from IMCoalHMM.isolation_model import Single, make_rates_table_single
from IMCoalHMM.CTMC import CTMC
from IMCoalHMM.transitions import CTMCSystem
from IMCoalHMM.break_points import psmc_break_points
from IMCoalHMM.emissions import coalescence_points
from IMCoalHMM.model import Model


## Code for computing HMM transition probabilities ####################
def _compute_through(ctmcs, break_points):
    """Computes the matrices for moving through an interval"""
    no_states = len(break_points)

    # Construct the transition matrices for going through each interval
    through = [ctmcs[i].probability_matrix(break_points[i + 1] - break_points[i])
               for i in xrange(no_states - 1)]

    # As a hack we set up a pseudo through matrix for the last interval that
    # just puts all probability on ending in one of the end states. This
    # simplifies the HMM transition probability code as it avoids a special case
    # for the last interval.
    # noinspection PyCallingNonCallable
    pseudo_through = matrix(zeros((len(ctmcs[-1].state_space.states),
                                   len(ctmcs[-1].state_space.states))))
    pseudo_through[:, ctmcs[-1].state_space.end_states[0]] = 1.0
    through.append(pseudo_through)

    return through


def _compute_upto(isolation, ctmcs, break_points, through):
    """Computes the probability matrices for moving from time zero up to,
    but not through, interval i."""

    no_states = len(break_points)

    # Projection matrix needed to go from the isolation to the single
    # state spaces.
    # noinspection PyCallingNonCallable
    projection = matrix(zeros((len(isolation.state_space.states),
                               len(ctmcs[0].state_space.states))))
    for state, isolation_index in isolation.state_space.states.items():
        ancestral_state = frozenset([(0, nucs) for (_, nucs) in state])
        ancestral_index = ctmcs[0].state_space.states[ancestral_state]
        projection[isolation_index, ancestral_index] = 1.0

    # We handle the first state as a special case because of the isolation
    # interval
    upto = [None] * no_states
    upto[0] = isolation.probability_matrix(break_points[0]) * projection
    for i in xrange(1, no_states):
        upto[i] = upto[i - 1] * through[i - 1]

    return upto


def _compute_between(ctmcs, through):
    """Computes the matrices for moving from the end of interval i
    to the beginning of interval j."""

    no_states = len(ctmcs)
    between = dict()
    # Transitions going from the endpoint of interval i to the entry point
    # of interval j
    for i in xrange(no_states - 1):
        # noinspection PyCallingNonCallable
        between[(i, i + 1)] = matrix(identity(len(ctmcs[0].state_space.states)))
        for j in xrange(i + 2, no_states):
            between[(i, j)] = between[(i, j - 1)] * through[j - 1]
    return between


class VariableCoalRateCTMCSystem(CTMCSystem):
    """Wrapper around CTMC transition matrices for the isolation model."""

    def __init__(self, isolation_ctmc, ancestral_ctmcs, break_points):
        """Construct all the matrices and cache them for the
        method calls.

        :param isolation_ctmc: CTMC for the initial isolation phase.
        :type isolation_ctmc: CTMC
        :param ancestral_ctmcs: CTMCs for the ancestral population.
        :type ancestral_ctmcs: list[CTMC]
        :param break_points: List of break points.
        :type break_points: list[int]
        """

        super(VariableCoalRateCTMCSystem, self).__init__(no_hmm_states=len(ancestral_ctmcs),
                                                         initial_ctmc_state=isolation_ctmc.state_space.i12_index)

        # Even though we have different CTMCs they have the same state space
        self.state_space = ancestral_ctmcs[0].state_space

        self.through_ = _compute_through(ancestral_ctmcs, break_points)
        self.upto_ = _compute_upto(isolation_ctmc, ancestral_ctmcs,
                                   break_points, self.through_)
        self.between_ = _compute_between(ancestral_ctmcs, self.through_)

    def get_state_space(self, _):
        """Return the state space for interval i, but it is always the same."""
        return self.state_space


## Class that can construct HMMs ######################################
class VariableCoalescenceRateIsolationModel(Model):
    """Class wrapping the code that generates an isolation model HMM
    with variable coalescence rates in the different intervals."""

    def __init__(self, intervals, est_split=False):
        """Construct the model.

        This builds the state spaces for the CTMCs but the matrices for the
        HMM since those will depend on the rate parameters."""
        super(VariableCoalescenceRateIsolationModel, self).__init__()
        self.isolation_state_space = Isolation()
        self.single_state_space = Single()
        self.intervals = intervals
        self.est_split = est_split

    def emission_points(self, *parameters):
        """Time points to emit from."""
        if self.est_split:
            # we are trying to estimate a split time as well
            split_time = parameters[0]
            coal_rates = parameters[1:-1]
        else:
            split_time = 0.0
            coal_rates = parameters[0:-1]

        no_states = sum(self.intervals)
        break_points = psmc_break_points(no_states, offset=split_time)

        # FIXME: I don't know how to choose a good rate here
        return coalescence_points(break_points, coal_rates[0])

    def build_ctmc_system(self, *parameters):
        """Construct CTMCs and compute HMM matrices given the split time
        and the rates.

        The split time parameter is for setting a period where it is
        impossible for the two samples to coalesce (an isolation model).
        If it is set to 0.0 the system will work as Li & Durbin (2011)'s PSMC.

        The intervals list specifies how many intervals we should use for
        each coalescence rate. It is the sum over this list that will
        be the number of states.

        The coal_rates list should contain a coalescence rate for each interval
        in the model (except for the time up to split_time). It determins
        both the number of states and the transition probabilities.
        In optimisation it should be constrained somewhat since a free
        rate for each interval will not be possible to estimate, but
        this is left to functionality outside the model.
        """

        if self.est_split:
            # we are trying to estimate a split time as well
            split_time = parameters[0]
            coal_rates = parameters[1:-1]
            recomb_rate = parameters[-1]
        else:
            split_time = 0.0
            coal_rates = parameters[0:-1]
            recomb_rate = parameters[-1]

        # We assume here that the coalescence rate is the same in the two
        # separate populations as in the ancestral just before teh split.
        # This is not necessarily true but it worked okay in simulations
        # in Mailund et al. (2011).

        isolation_rates = make_rates_table_isolation(coal_rates[0], coal_rates[0], recomb_rate)
        isolation_ctmc = CTMC(self.isolation_state_space, isolation_rates)

        ancestral_ctmcs = []
        for epoch, coal_rate in enumerate(coal_rates):
            single_rates = make_rates_table_single(coal_rate, recomb_rate)
            single_ctmc = CTMC(self.single_state_space, single_rates)
            for _ in xrange(self.intervals[epoch]):
                ancestral_ctmcs.append(single_ctmc)

        no_states = len(ancestral_ctmcs)
        break_points = psmc_break_points(no_states, offset=split_time)

        return VariableCoalRateCTMCSystem(isolation_ctmc, ancestral_ctmcs, break_points)
