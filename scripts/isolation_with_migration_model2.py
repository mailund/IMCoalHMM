"""Code for constructing and optimizing the HMM for an isolation model.
"""

from numpy import zeros, matrix, ndarray
from numpy.testing import assert_almost_equal

from IMCoalHMM.CTMC import make_ctmc
from IMCoalHMM.transitions import CTMCSystem, projection_matrix, compute_upto, compute_between, compute_transition_probabilities
from emissions2 import coalescence_points,emission_matrix4,emission_matrix7, emission_matrix, emission_matrix8
from IMCoalHMM.break_points import exp_break_points, uniform_break_points
from IMCoalHMM.model import Model

from IMCoalHMM.state_spaces import Isolation, make_rates_table_isolation
from IMCoalHMM.state_spaces import Single, make_rates_table_single
from IMCoalHMM.state_spaces import Migration, make_rates_table_migration



def printPyZipHMM(Matrix):
    finalString=""
    for i in range(Matrix.getHeight()):
        for j in range(Matrix.getWidth()):
            finalString=finalString+" "+str(Matrix[i,j])
        finalString=finalString+"\n"
    return finalString


## Code for computing HMM transition probabilities ####################

# The way multiprocessing works means that we have to define this class for mapping in parallel
# and we have to define the processing pool after we define the class, or it won't be able to see
# it in the sub-processes. It breaks the flow of the code, but it is necessary.

class ComputeThroughInterval(object):
    def __init__(self, ctmc, break_points):
        self.ctmc = ctmc
        self.break_points = break_points

    def __call__(self, i):
        return self.ctmc.probability_matrix(self.break_points[i + 1] - self.break_points[i])


def _compute_through(migration, migration_break_points,
                     ancestral, ancestral_break_points):
    """Computes the matrices for moving through an interval"""

    def state_map(state):
        return frozenset([(0, nucs) for (_, nucs) in state])
    projection = projection_matrix(migration.state_space, ancestral.state_space, state_map)

    no_migration_states = len(migration_break_points)
    no_ancestral_states = len(ancestral_break_points)

    # Construct the transition matrices for going through each interval in
    # the migration phase
    migration_through = map(ComputeThroughInterval(migration, migration_break_points),
                            range(no_migration_states - 1))
    last_migration = migration.probability_matrix(ancestral_break_points[0] - migration_break_points[-1]) * projection
    migration_through.append(last_migration)

    ancestral_through = map(ComputeThroughInterval(ancestral, ancestral_break_points),
                            range(no_ancestral_states - 1))

    # As a hack we set up a pseudo through matrix for the last interval that
    # just puts all probability on ending in one of the end states. This
    # simplifies the HMM transition probability code as it avoids a special case
    # for the last interval.
    # noinspection PyCallingNonCallable
    pseudo_through = matrix(zeros((len(ancestral.state_space.states),
                                   len(ancestral.state_space.states))))
    pseudo_through[:, ancestral.state_space.end_states[0]] = 1.0
    ancestral_through.append(pseudo_through)

    return migration_through + ancestral_through


def _compute_upto0(isolation, migration, break_points):
    """Computes the probability matrices for moving to time zero."""
    # the states in the isolation state space are the same in the migration
    state_map = lambda x: x
    projection = projection_matrix(isolation.state_space, migration.state_space, state_map)
    return isolation.probability_matrix(break_points[0]) * projection


class IsolationMigrationCTMCSystem(CTMCSystem):
    """Wrapper around CTMC transition matrices for the isolation model."""

    def __init__(self, isolation_ctmc, migration_ctmc, ancestral_ctmc,
                 migration_break_points, ancestral_break_points):
        """Construct all the matrices and cache them for the
        method calls.

        :param isolation_ctmc: CTMC for the isolation phase.
        :type isolation_ctmc: IMCoalHMM.CTMC.CTMC
        :param migration_ctmc: CTMC for the migration phase.
        :type migration_ctmc: IMCoalHMM.CTMC.CTMC
        :param ancestral_ctmc: CTMC for the ancestral population.
        :type ancestral_ctmc: IMCoalHMM.CTMC.CTMC
        :param migration_break_points: List of break points in the migration phase.
        :type migration_break_points: list[int]
        :param ancestral_break_points: List of break points in the ancestral population.
        :type ancestral_break_points: list[int]
        """

        self.no_migration_states = len(migration_break_points)
        self.no_ancestral_states = len(ancestral_break_points)
        no_states = self.no_migration_states + self.no_ancestral_states
        super(IsolationMigrationCTMCSystem, self).__init__(no_states, isolation_ctmc.state_space.i12_index)

        self.state_spaces = [migration_ctmc.state_space, ancestral_ctmc.state_space]

        break_points = list(migration_break_points) + list(ancestral_break_points)

        self.through_ = _compute_through(migration_ctmc, migration_break_points,
                                         ancestral_ctmc, ancestral_break_points)
        self.upto_ = compute_upto(_compute_upto0(isolation_ctmc, migration_ctmc, break_points), self.through_)
        self.between_ = compute_between(self.through_)

    def get_state_space(self, i):
        """Return the right state space for the interval."""
        return self.state_spaces[self.no_migration_states <= i]


## Class that can construct HMMs ######################################
class IsolationMigrationModel(Model):
    """Class wrapping the code that generates an isolation model HMM."""

    INITIAL_11=0
    INITIAL_12=1
    INITIAL_22=2
    
    def __init__(self, no_mig_states, no_ancestral_states, config=INITIAL_12, outgroup=False):
        """Construct the model.

        This builds the state spaces for the CTMCs but not the matrices for the
        HMM since those will depend on the rate parameters."""
        super(IsolationMigrationModel, self).__init__()
        self.config=config
        if config==IsolationMigrationModel.INITIAL_12:
            self.isolation_state_space = Isolation()
        elif config==IsolationMigrationModel.INITIAL_11:
            self.isolation_state_space = IsolationSingle(1,2)
        else:
            self.isolation_state_space = IsolationSingle(2,1)
        self.migration_state_space = Migration()
        self.single_state_space = Single()
        self.no_mig_states = no_mig_states
        self.no_ancestral_states = no_ancestral_states
        self.outgroup=outgroup
        
    # noinspection PyMethodMayBeStatic
    def valid_parameters(self, parameters):
        """Predicate testing if a given parameter point is valid for the model.
        :param parameters: Model specific parameters
        :type parameters: numpy.ndarray
        :returns: True if all parameters are valid, otherwise False
        :rtype: bool
        """
        # This works but pycharm gives a type warning... I guess it doesn't see > overloading
        assert isinstance(parameters, ndarray), "the argument parameters="+str(parameters)+ " is not an numpy.ndarray but an "+str(type(parameters))
        # noinspection PyTypeChecker
        
        
        if parameters[2]<1e-8: #checking specifically for the coalescense rate
            return False
        
        #checking the outgroup is larger than the split time
        if self.outgroup:
            if parameters[5]<parameters[0] or parameters[5]<parameters[1]:
                return False
        
        return all(parameters >= 0)

    def emission_points(self, isolation_time, migration_time, coal_rate, recomb_rate, mig_rate):
        """Compute model specific coalescence points."""
        tau1 = isolation_time
        tau2 = isolation_time + migration_time
        migration_break_points = uniform_break_points(self.no_mig_states, tau1, tau2)
        ancestral_break_points = exp_break_points(self.no_ancestral_states, coal_rate, tau2)
        if self.outgroup:
            if ancestral_break_points[-1]>self.outmax:
                ancestral_break_points=uniform_break_points(self.no_mig_states, tau2, self.outmax-(self.outmax-tau2)/20.0)
        break_points = list(migration_break_points) + list(ancestral_break_points)
        return coalescence_points(break_points, coal_rate)

    def build_ctmc_system(self, isolation_time, migration_time, coal_rate, recomb_rate, mig_rate):
        """Construct CTMCs and compute HMM matrices given the split times
        and the rates."""

        # We assume here that the coalescence rate is the same in the two
        # separate populations as it is in the ancestral. This is not necessarily
        # true but it worked okay in simulations in Mailund et al. (2012).

        isolation_rates = make_rates_table_isolation(coal_rate, coal_rate, recomb_rate)
        isolation_ctmc = make_ctmc(self.isolation_state_space, isolation_rates)

        migration_rates = make_rates_table_migration(coal_rate, coal_rate, recomb_rate,
                                                     mig_rate, mig_rate)
        migration_ctmc = make_ctmc(self.migration_state_space, migration_rates)

        single_rates = make_rates_table_single(coal_rate, recomb_rate)
        single_ctmc = make_ctmc(self.single_state_space, single_rates)

        tau1 = isolation_time
        tau2 = isolation_time + migration_time
        migration_break_points = uniform_break_points(self.no_mig_states, tau1, tau2)
        ancestral_break_points = exp_break_points(self.no_ancestral_states, coal_rate, tau2)
        if self.outgroup:
            if ancestral_break_points[-1]>self.outmax:
                ancestral_break_points=uniform_break_points(self.no_mig_states, tau2, self.outmax-(self.outmax-tau2)/20.0)

        return IsolationMigrationCTMCSystem(isolation_ctmc, migration_ctmc, single_ctmc,
                                            migration_break_points, ancestral_break_points)
        
        
    def build_hidden_markov_model(self, parameters):
        """Build the hidden Markov model matrices from the model-specific parameters."""
        
        if self.outgroup:
            outgroup=parameters[-1]
            self.outmax=outgroup
            parameters=parameters[:-1]
        
        isolation_time, migration_time, coal_rate, recomb_rate, mig_rate = parameters
        ctmc_system = self.build_ctmc_system(isolation_time, migration_time, coal_rate, recomb_rate, mig_rate)
        #changing the break_points
        tau1 = isolation_time
        tau2 = isolation_time + migration_time
        migration_break_points = uniform_break_points(self.no_mig_states, 0, tau2-tau1)
        ancestral_break_points = exp_break_points(self.no_ancestral_states, coal_rate, tau2-tau1)
        if self.outgroup:
            if ancestral_break_points[-1]>self.outmax:
                ancestral_break_points=uniform_break_points(self.no_mig_states, tau2, outgroup-(outgroup-tau2)/20.0)
        break_points=list(migration_break_points)+list(ancestral_break_points)
        initial_probs, transition_probs = compute_transition_probabilities(ctmc_system)
        parameters2=[coal_rate]*4+[mig_rate,0,mig_rate,0]+[recomb_rate]
        intervals=[self.no_mig_states,self.no_ancestral_states]
#         emission_probs = emission_matrix(self.emission_points(*parameters))
#         print " ------------- Emis 0 --------------"
#         print printPyZipHMM(emission_probs)
        if self.outgroup:
            emission_probs = emission_matrix8(break_points, parameters2, outgroup, intervals, ctmc_system, 0)
        else:
            emission_probs = emission_matrix7(break_points, parameters2, intervals, ctmc_system, 0)
        
#         emission_probs = emission_matrix3(br, parameters, self.intervals)
#           
#         def printPyZipHMM(Matrix):
#             finalString=""
#             for i in range(Matrix.getHeight()):
#                 for j in range(Matrix.getWidth()):
#                     finalString=finalString+" "+str(Matrix[i,j])
#                 finalString=finalString+"\n"
#             return finalString
#         strToWirte=str(parameters)+"\n"+str("3:")+printPyZipHMM(emission_probs)+"\n"
        
#         strToWirte=strToWirte+str("4:")+printPyZipHMM(emission_probs)+"\n"+"initial_probs: "+printPyZipHMM(initial_probs)
#         emission_probs = emission_matrix3b(br, parameters, self.intervals,ctmc_system)
#         print strToWirte+str("3b:")+printPyZipHMM(emission_probs)
#         print " ------------- Emis 6 --------------"
#         print printPyZipHMM(emission_probs)
#         
        return initial_probs, transition_probs, emission_probs



def main():
    """Test"""

    no_mig_states = 4
    no_ancestral_states = 4
    isolation_time = 0.0005
    migration_time = 0.0010
    coal_rate = 1000
    recomb_rate = 0.4
    mig_rate = 300

    model = IsolationMigrationModel(no_mig_states, no_ancestral_states)
    parameters = isolation_time, migration_time, coal_rate, recomb_rate, mig_rate
    pi, transition_probs, emission_probs = model.build_hidden_markov_model(parameters)
    print "--------- EMISS ---------"
    print printPyZipHMM(emission_probs)
    print "------- TRANS -----------"
    print printPyZipHMM(transition_probs)

    no_states = pi.getHeight()
    assert no_states == no_mig_states + no_ancestral_states

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
