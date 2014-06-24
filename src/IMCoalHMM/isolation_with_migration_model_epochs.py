"""Code for constructing and optimizing the HMM for an isolation model.
"""

from numpy import zeros, matrix, mean
from numpy.testing import assert_almost_equal

from IMCoalHMM.CTMC import make_ctmc
from IMCoalHMM.transitions import CTMCSystem, projection_matrix, compute_upto, compute_between
from IMCoalHMM.emissions import coalescence_points
from IMCoalHMM.break_points import exp_break_points, uniform_break_points
from IMCoalHMM.model import Model

from IMCoalHMM.state_spaces import Isolation, make_rates_table_isolation
from IMCoalHMM.state_spaces import Single, make_rates_table_single
from IMCoalHMM.state_spaces import Migration, make_rates_table_migration



# # Code for computing HMM transition probabilities ####################

# The way multiprocessing works means that we have to define this class for mapping in parallel
# and we have to define the processing pool after we define the class, or it won't be able to see
# it in the sub-processes. It breaks the flow of the code, but it is necessary.

class ComputeThroughInterval(object):
    def __init__(self, ctmcs, break_points):
        self.ctmcs = ctmcs
        self.break_points = break_points

    def __call__(self, i):
        return self.ctmcs[i].probability_matrix(self.break_points[i + 1] - self.break_points[i])


def _compute_through(migration_ctmcs, migration_break_points,
                     ancestral_ctmcs, ancestral_break_points):
    """Computes the matrices for moving through an interval.

    :param migration_ctmcs: CTMCs for the migration phase.
    :type migration_ctmcs: list[IMCoalHMM.CTMC.CTMC]
    :param ancestral_ctmcs: CTMCs for the ancestral population.
    :type ancestral_ctmcs: list[IMCoalHMM.CTMC.CTMC]
    :param migration_break_points: List of break points in the migration phase.
    :type migration_break_points: list[float]
    :param ancestral_break_points: List of break points in the ancestral population.
    :type ancestral_break_points: list[float]
    """

    def state_map(state):
        return frozenset([(0, nucs) for (_, nucs) in state])

    projection = projection_matrix(migration_ctmcs[0].state_space, ancestral_ctmcs[0].state_space, state_map)

    no_migration_states = len(migration_break_points)
    no_ancestral_states = len(ancestral_break_points)

    # Construct the transition matrices for going through each interval in
    # the migration phase
    migration_through = map(ComputeThroughInterval(migration_ctmcs, migration_break_points),
                            range(no_migration_states - 1))
    last_migration = migration_ctmcs[-1].probability_matrix(
        ancestral_break_points[0] - migration_break_points[-1]) * projection
    migration_through.append(last_migration)

    ancestral_through = map(ComputeThroughInterval(ancestral_ctmcs, ancestral_break_points),
                            range(no_ancestral_states - 1))

    # As a hack we set up a pseudo through matrix for the last interval that
    # just puts all probability on ending in one of the end states. This
    # simplifies the HMM transition probability code as it avoids a special case
    # for the last interval.
    # noinspection PyCallingNonCallable
    pseudo_through = matrix(zeros((len(ancestral_ctmcs[0].state_space.states),
                                   len(ancestral_ctmcs[0].state_space.states))))
    pseudo_through[:, ancestral_ctmcs[0].state_space.end_states[0]] = 1.0
    ancestral_through.append(pseudo_through)

    return migration_through + ancestral_through


def _compute_upto0(isolation, migration, break_points):
    """Computes the probability matrices for moving to time zero."""
    # the states in the isolation state space are the same in the migration
    state_map = lambda x: x
    projection = projection_matrix(isolation.state_space, migration.state_space, state_map)
    return isolation.probability_matrix(break_points[0]) * projection


class IsolationMigrationEpochsCTMCSystem(CTMCSystem):
    """Wrapper around CTMC transition matrices for the isolation model."""

    def __init__(self, isolation_ctmc, migration_ctmcs, ancestral_ctmcs,
                 migration_break_points, ancestral_break_points):
        """Construct all the matrices and cache them for the
        method calls.

        :param isolation_ctmc: CTMC for the isolation phase.
        :type isolation_ctmc: IMCoalHMM.CTMC.CTMC
        :param migration_ctmcs: CTMCs for the migration phase.
        :type migration_ctmcs: list[IMCoalHMM.CTMC.CTMC]
        :param ancestral_ctmcs: CTMCs for the ancestral population.
        :type ancestral_ctmcs: list[IMCoalHMM.CTMC.CTMC]
        :param migration_break_points: List of break points in the migration phase.
        :type migration_break_points: list[float]
        :param ancestral_break_points: List of break points in the ancestral population.
        :type ancestral_break_points: list[float]
        """

        self.no_migration_states = len(migration_break_points)
        self.no_ancestral_states = len(ancestral_break_points)
        no_states = self.no_migration_states + self.no_ancestral_states
        super(IsolationMigrationEpochsCTMCSystem, self).__init__(no_states, isolation_ctmc.state_space.i12_index)

        # Don't include isolation_ctmc here... the ctmcs here are only those we can coalesce in!
        self.ctmcs = migration_ctmcs + ancestral_ctmcs
        # This is a hack to match the "pseudo_through" probability matrix for the last interval,
        # where we need the state space of the interval _past_ the last
        self.ctmcs.append(ancestral_ctmcs[-1])

        break_points = list(migration_break_points) + list(ancestral_break_points)
        upto0 = _compute_upto0(isolation_ctmc, migration_ctmcs[0], break_points)
        self.through_ = _compute_through(migration_ctmcs, migration_break_points,
                                         ancestral_ctmcs, ancestral_break_points)

        self.upto_ = compute_upto(upto0, self.through_)
        self.between_ = compute_between(self.through_)

    def get_state_space(self, i):
        """Return the right state space for the interval."""
        return self.ctmcs[i].state_space


## Class that can construct HMMs ######################################
class IsolationMigrationEpochsModel(Model):
    """Class wrapping the code that generates an isolation model HMM."""

    def __init__(self, no_epochs, no_mig_states, no_ancestral_states):
        """Construct the model.

        This builds the state spaces for the CTMCs but not the matrices for the
        HMM since those will depend on the rate parameters."""
        super(IsolationMigrationEpochsModel, self).__init__()
        self.isolation_state_space = Isolation()
        self.migration_state_space = Migration()
        self.single_state_space = Single()

        self.no_epochs = no_epochs
        self.no_mig_states = no_mig_states
        self.no_ancestral_states = no_ancestral_states

    def emission_points(self, *parameters):
        """Compute model specific coalescence points."""

        isolation_time, migration_time, recomb_rate = parameters[:3]
        coal_rates = parameters[3:2 * self.no_epochs + 1 + 3]

        tau1 = isolation_time
        tau2 = isolation_time + migration_time

        migration_break_points = uniform_break_points(self.no_epochs * self.no_mig_states, tau1, tau2)

        # FIXME: This should really take into account that the coal rate varies between epochs...
        coal_rate = mean(coal_rates)
        ancestral_break_points = exp_break_points(self.no_epochs * self.no_ancestral_states, coal_rate, tau2)

        break_points = list(migration_break_points) + list(ancestral_break_points)
        return coalescence_points(break_points, coal_rate)

    def build_ctmc_system(self, *parameters):
        """Construct CTMCs and compute HMM matrices given the split times
        and the rates."""

        isolation_time, migration_time, recomb_rate = parameters[:3]
        coal_rates = parameters[3:2 * self.no_epochs + 1 + 3]
        mig_rates = parameters[2 * self.no_epochs + 1 + 3:]

        assert len(coal_rates) == self.no_epochs * 2 + 1, "Isolation + #Epochs migration + #Epochs ancestral"
        assert len(mig_rates) == self.no_epochs, "#Epochs migration epochs"

        isolation_rates = make_rates_table_isolation(coal_rates[0], coal_rates[0], recomb_rate)
        isolation_ctmc = make_ctmc(self.isolation_state_space, isolation_rates)

        migration_ctmcs = []
        ancestral_ctmcs = []

        for epoch in xrange(self.no_epochs):
            migration_rates = make_rates_table_migration(coal_rates[epoch + 1], coal_rates[epoch + 1], recomb_rate,
                                                         mig_rates[epoch], mig_rates[epoch])
            migration_ctmc = make_ctmc(self.migration_state_space, migration_rates)

            # Repeat of the same CTMC throughout the epoch. Change here if different number of states per epoch
            for _ in xrange(self.no_mig_states):
                migration_ctmcs.append(migration_ctmc)

        for epoch in xrange(self.no_epochs):
            ancestral_rates = make_rates_table_single(coal_rates[epoch + self.no_epochs + 1], recomb_rate)
            ancestral_ctmc = make_ctmc(self.single_state_space, ancestral_rates)

            # Repeat of the same CTMC throughout the epoch. Change here if different number of states per epoch
            for _ in xrange(self.no_ancestral_states):
                ancestral_ctmcs.append(ancestral_ctmc)

        tau1 = isolation_time
        tau2 = isolation_time + migration_time
        migration_break_points = uniform_break_points(self.no_epochs * self.no_mig_states, tau1, tau2)

        # FIXME: This should take into account that the coal rate varies between epochs...
        coal_rate = mean(coal_rates[self.no_epochs + 1:])
        ancestral_break_points = exp_break_points(self.no_epochs * self.no_ancestral_states, coal_rate, tau2)

        return IsolationMigrationEpochsCTMCSystem(isolation_ctmc, migration_ctmcs, ancestral_ctmcs,
                                                  migration_break_points, ancestral_break_points)


def main():
    """Test"""

    no_epochs = 2
    no_mig_states = 4
    no_ancestral_states = 4
    isolation_time = 0.5
    migration_time = 1.0

    coal_rate = 1
    recomb_rate = 0.4
    mig_rate = 0.1

    model = IsolationMigrationEpochsModel(no_epochs, no_mig_states, no_ancestral_states)

    parameters = [isolation_time, migration_time, recomb_rate]
    parameters.extend([coal_rate] * (2 * no_epochs + 1))
    parameters.extend([mig_rate] * no_epochs)

    pi, transition_probs, emission_probs = model.build_hidden_markov_model(parameters)

    no_states = pi.getHeight()
    assert no_states == no_epochs * no_mig_states + no_epochs * no_ancestral_states

    pi_sum = 0.0
    for row in xrange(no_states):
        pi_sum += pi[row, 0]
    assert_almost_equal(pi_sum, 1.0)

    assert no_states == transition_probs.getWidth()
    assert no_states == transition_probs.getHeight()

    transitions_sum = 0.0
    for row in xrange(no_states):
        for col in xrange(no_states):
            transitions_sum += transition_probs[row, col]
    assert_almost_equal(transitions_sum, no_states)

    print 'Done'


if __name__ == '__main__':
    main()
