"""Code for constructing and optimizing the HMM for a model with variable
migration and coalescence.
"""

from numpy import zeros, matrix, identity

from IMCoalHMM.state_spaces import Migration, make_rates_table_migration
from IMCoalHMM.CTMC import CTMC
from IMCoalHMM.transitions import CTMCSystem, compute_upto, compute_between
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


class VariableCoalAndMigrationRateCTMCSystem(CTMCSystem):
    """Wrapper around CTMC transition matrices for the isolation model."""

    def __init__(self, initial_state, ctmcs, break_points):
        """Construct all the matrices and cache them for the
        method calls.

        :param initial_state: The initial state for this CTMC system.
            We include it in the constructor for this model because we want to handle
            both samples from each population and between them.
        :param ctmcs: CTMCs for each interval.
        :type ctmcs: list[CTMC]
        :param break_points: List of break points.
        :type break_points: list[float]
        """

        super(VariableCoalAndMigrationRateCTMCSystem, self).__init__(no_hmm_states=len(ctmcs),
                                                                     initial_ctmc_state=initial_state)

        # Even though we have different CTMCs they have the same state space
        self.state_space = ctmcs[0].state_space

        self.through_ = _compute_through(ctmcs, break_points)

        # noinspection PyCallingNonCallable
        upto0 = matrix(identity(len(ctmcs[0].state_space.states)))
        self.upto_ = compute_upto(upto0, self.through_)

        self.between_ = compute_between(self.through_)

    def get_state_space(self, _):
        """Return the state space for interval i, but it is always the same."""
        return self.state_space


## Class that can construct HMMs ######################################
class VariableCoalAndMigrationRateModel(Model):
    """Class wrapping the code that generates an isolation model HMM
    with variable coalescence rates in the different intervals."""

    # Determines which initial state to start the CTMCs in
    INITIAL_11 = 0
    INITIAL_12 = 1
    INITIAL_22 = 2

    def __init__(self, initial_configuration, intervals):
        """Construct the model.

        This builds the state spaces for the CTMCs but the matrices for the
        HMM since those will depend on the rate parameters."""
        super(VariableCoalAndMigrationRateModel, self).__init__()

        self.migration_state_space = Migration()

        if initial_configuration == self.INITIAL_11:
            self.initial_state = self.migration_state_space.i11_index
        elif initial_configuration == self.INITIAL_12:
            self.initial_state = self.migration_state_space.i12_index
        elif initial_configuration == self.INITIAL_22:
            self.initial_state = self.migration_state_space.i22_index
        else:
            assert False, "We should never reach this point!"

        self.intervals = intervals
        self.no_states = sum(intervals)

    def emission_points(self, *parameters):
        """Time points to emit from."""
        # FIXME: This is just one of the rate parameters and I am not sure it is a particular good choice
        # I need one for the coalescence points, though.
        coal_rate = parameters[0]
        break_points = psmc_break_points(self.no_states)
        return coalescence_points(break_points, coal_rate)

    def unpack_parameters(self, parameters):
        """Unpack the rate parameters for the model from the linear representation
        used in optimizations to the specific rate parameters.
        """
        no_epochs = len(self.intervals)
        coal_rates_1 = parameters[0:no_epochs]
        coal_rates_2 = parameters[no_epochs:(2*no_epochs)]
        mig_rates_12 = parameters[(2*no_epochs):(3*no_epochs)]
        mig_rates_21 = parameters[(3*no_epochs):(4*no_epochs)]
        recomb_rate = parameters[-1]
        return coal_rates_1, coal_rates_2, mig_rates_12, mig_rates_21, recomb_rate

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
        in the model (except for the time up to split_time). It determines
        both the number of states and the transition probabilities.
        In optimisation it should be constrained somewhat since a free
        rate for each interval will not be possible to estimate, but
        this is left to functionality outside the model.
        """

        coal_rates_1, coal_rates_2, mig_rates_12, mig_rates_21, recomb_rate = self.unpack_parameters(parameters)

        ctmcs = []
        for epoch, states_in_interval in enumerate(self.intervals):
            rates = make_rates_table_migration(coal_rates_1[epoch], coal_rates_2[epoch],
                                               mig_rates_12[epoch], mig_rates_21[epoch],
                                               recomb_rate)
            ctmc = CTMC(self.migration_state_space, rates)
            for _ in xrange(states_in_interval):
                ctmcs.append(ctmc)

        break_points = psmc_break_points(self.no_states)

        return VariableCoalAndMigrationRateCTMCSystem(self.initial_state, ctmcs, break_points)
