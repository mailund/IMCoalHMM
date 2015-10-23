"""Code for constructing and optimizing the HMM for a model with variable
migration and coalescence.
"""

from numpy import zeros, matrix, identity,cumsum,array, concatenate

from IMCoalHMM.state_spaces import Migration, make_rates_table_migration, make_rates_table_single, Single
from CTMC2 import make_ctmc
from IMCoalHMM.transitions import CTMCSystem, compute_upto, compute_between, compute_transition_probabilities, projection_matrix
from break_points2 import psmc_break_points, uniform_break_points, gamma_break_points
from IMCoalHMM.emissions import coalescence_points
from IMCoalHMM.model import Model
from emissions2 import emission_matrix3,emission_matrix4, emission_matrix3b, emission_matrix5, emission_matrix6
import pyZipHMM
from IMCoalHMM import break_points



## Code for computing HMM transition probabilities ####################

# The way multiprocessing works means that we have to define this class for mapping in parallel
# and we have to define the processing pool after we define the class, or it won't be able to see
# it in the sub-processes. It breaks the flow of the code, but it is necessary.

class ComputeThroughInterval(object):
    def __init__(self, ctmcs, break_points):
        self.ctmcs = ctmcs
        self.break_points = break_points

    def __call__(self, i):
        return self.ctmcs[i].probability_matrix(self.break_points[i + 1] - self.break_points[i])


def _compute_through(ctmcs, break_points, migration_state_space, ancestral_state_space, no_migration_states):
    """Computes the matrices for moving through an interval"""
    def state_map(state):
        return frozenset([(0, nucs) for (_, nucs) in state])
    projection = projection_matrix(migration_state_space, ancestral_state_space, state_map)
    no_states = len(break_points)

    # Construct the transition matrices for going through each interval
    through = map(ComputeThroughInterval(ctmcs[:no_migration_states], break_points), range(no_migration_states))
    last_migration = through[-1] * projection
    through[-1]=last_migration
    
    ancestral_through = map(ComputeThroughInterval(ctmcs, break_points),
                            range(no_migration_states, no_states-1))
    through+=ancestral_through

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


class VariableCoalAndMigrationRateAndAncestralCTMCSystem(CTMCSystem):
    """Wrapper around CTMC transition matrices for the isolation model."""

    def __init__(self, initial_state,no_migration_states, ctmcs, break_points):
        """Construct all the matrices and cache them for the
        method calls.

        :param initial_state: The initial state for this CTMC system.
            We include it in the constructor for this model because we want to handle
            both samples from each population and between them.
        :param ctmcs: CTMCs for each interval.
        :type ctmcs: list[IMCoalHMM.CTMC.CTMC]
        :param break_points: List of break points.
        :type break_points: list[float]
        """

        super(VariableCoalAndMigrationRateAndAncestralCTMCSystem, self).__init__(no_hmm_states=len(ctmcs),
                                                                     initial_ctmc_state=initial_state)

        # Even though we have different CTMCs they have the same state space
        self.state_spaces = [ctmcs[0].state_space, ctmcs[-1].state_space]
        self.no_migration_states=no_migration_states
        self.break_points=break_points

        self.through_ = _compute_through(ctmcs, break_points,self.state_spaces[0], self.state_spaces[1], self.no_migration_states)

        # noinspection PyCallingNonCallable
        upto0 = matrix(identity(len(ctmcs[0].state_space.states)))
        self.upto_ = compute_upto(upto0, self.through_)

        self.between_ = compute_between(self.through_)

    def get_state_space(self, i):
        """Return the state space for interval i, but it is always the same."""
        return self.state_spaces[self.no_migration_states<=i]


## Class that can construct HMMs ######################################
class VariableCoalAndMigrationRateAndAncestralModel(Model):
    """Class wrapping the code that generates an isolation model HMM
    with variable coalescence rates in the different intervals."""

    # Determines which initial state to start the CTMCs in
    INITIAL_11 = 0
    INITIAL_12 = 1
    INITIAL_22 = 2

    def __init__(self, initial_configuration, intervals, breaktimes, breaktail=0, time_modifier=None):
        self.breaktimes=breaktimes
        self.breaktail=breaktail
        self.time_modifier=time_modifier
        """Construct the model.

        This builds the state spaces for the CTMCs but the matrices for the
        HMM since those will depend on the rate parameters."""
        super(VariableCoalAndMigrationRateAndAncestralModel, self).__init__()

        self.migration_state_space = Migration()
        self.single_state_space = Single()
        
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

    def unpack_parameters(self, parameters):
        """Unpack the rate parameters for the model from the linear representation
        used in optimizations to the specific rate parameters.
        """
        no_epochs = len(self.intervals)
        coal_rates_1 = parameters[0:no_epochs]
        coal_rates_2 = parameters[no_epochs:(2*no_epochs)]
        mig_rates_12 = parameters[(2*no_epochs):(3*no_epochs)]
        mig_rates_21 = parameters[(3*no_epochs):(4*no_epochs)]
        recomb_rate = parameters[4*no_epochs]
        if no_epochs*4+1<len(parameters):
            fixed_time_points=self.time_modifier(parameters[(len(coal_rates_1)*4+1):])
        elif self.time_modifier is not None:
            fixed_time_points=self.time_modifier()
        else:
            fixed_time_points=[]
        return coal_rates_1, coal_rates_2, mig_rates_12, mig_rates_21, recomb_rate, fixed_time_points

    def _map_rates_to_intervals(self, coal_rates):
        """Takes the coalescence rates as specified when building the CTMC
        and maps them to each interval based on the intervals specification."""
        assert len(coal_rates) == len(self.intervals)
        interval_rates = []
        for epoch, coal_rate in enumerate(coal_rates):
            for _ in xrange(self.intervals[epoch]):
                interval_rates.append(coal_rate)
        return interval_rates

    def emission_points(self, *parameters):
        """Time points to emit from."""
        # Emitting from points given by the mean of the coalescence rates in both populations.
        # When there is migration it is hard to know where the mean coalescence rate actually is
        # and it will depend on the starting point. This is a compromise at least.
        coal_rates_1, coal_rates_2, _, _, _,fixed_time_points = self.unpack_parameters(parameters)
        mean_coal_rates = [(c1+c2)/2.0 for c1, c2 in zip(coal_rates_1, coal_rates_2)]
        #break_points = psmc_break_points(self.no_states,t_max=self.tmax)
        break_points=gamma_break_points(self.no_states,beta1=0.001*self.breaktimes,alpha=2,beta2=0.001333333*self.breaktimes, tenthsInTheEnd=self.breaktail,fixed_time_points=fixed_time_points)
        return coalescence_points(break_points, self._map_rates_to_intervals(mean_coal_rates))

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

        coal_rates_1, coal_rates_2, mig_rates_12, mig_rates_21, recomb_rate,fixed_time_points = self.unpack_parameters(parameters)

        ctmcs = []
        for epoch, states_in_interval in enumerate(self.intervals[:-1]):
            rates = make_rates_table_migration(coal_rates_1[epoch], coal_rates_2[epoch], recomb_rate,
                                               mig_rates_12[epoch], mig_rates_21[epoch])
            ctmc = make_ctmc(self.migration_state_space, rates)
            for _ in xrange(states_in_interval):
                ctmcs.append(ctmc)
        rates= make_rates_table_single(float(coal_rates_1[-1]+coal_rates_2[-1])/2.0, recomb_rate)
        ctmc = make_ctmc(self.single_state_space, rates)
        for _ in xrange(self.intervals[-1]):
            ctmcs.append(ctmc)
        


        #break_points = psmc_break_points(self.no_states, t_max=self.tmax)
        break_points=gamma_break_points(self.no_states,beta1=0.001*self.breaktimes,alpha=2,beta2=0.001333333*self.breaktimes, tenthsInTheEnd=self.breaktail, fixed_time_points=fixed_time_points)
        #break_points = uniform_break_points(self.no_states,0,self.tmax*1e-9)

        return VariableCoalAndMigrationRateAndAncestralCTMCSystem(self.initial_state, sum(self.intervals[:-1]), ctmcs, break_points)
    
    #override for trying out special things
    def build_hidden_markov_model(self, parameters):
        """Build the hidden Markov model matrices from the model-specific parameters."""
        ctmc_system = self.build_ctmc_system(*parameters)
        initial_probs, transition_probs = compute_transition_probabilities(ctmc_system) #this code might throw a runtimeerror because NaNs are produced. If they are produced, they should be fixed later.
        br=ctmc_system.break_points
#         emission_probs = emission_matrix3(br, parameters, self.intervals)
#         
        if self.initial_state==self.migration_state_space.i12_index: #here we are checking for 0s in the first migration parameters.
            coals1,coals2,migs1,migs2,rho,_=self.unpack_parameters(parameters)
#             print "coals1 ",coals1
#             print "coals2 ",coals2
#             print "migs1 ",migs1
#             print "migs2 ",migs2
#             migs1=migs[:len(migs)/2]
#             migs2=migs[len(migs)/2:]
            assert sum(migs1)+sum(migs2)>0, "migration rates can not all be 0 and any can not be negative"
            indexOfFirstNonZero=min([n for n,(r,s) in enumerate(zip(migs1,migs2)) if r>0 or s>0])
            if indexOfFirstNonZero>0:
                indexOfFirstNonZeroMeasuredInBreakPoints=cumsum(self.intervals)[indexOfFirstNonZero-1]
#                 coals1=parameters[0:(len(parameters)-1)/4]
#                 coals2=parameters[(len(parameters)-1)/4:(len(parameters)-1)/2]
#                 print "indexOfFirstNonZero ",indexOfFirstNonZero
#                 print "indexOfFirstNonZeroMeasuredInBreakPoints", indexOfFirstNonZeroMeasuredInBreakPoints
#                 print "remaining intervals ", self.intervals[indexOfFirstNonZero:]
#                 print "coals1[indexOfFirstNonZero:] ",coals1[indexOfFirstNonZero:]
#                 print "coals2[indexOfFirstNonZero:] ",coals2[indexOfFirstNonZero:]
#                 print "migs1[indexOfFirstNonZero:] ",migs1[indexOfFirstNonZero:]
#                 print "migs2[indexOfFirstNonZero:] ",migs2[indexOfFirstNonZero:]
#                 print "[rho] ",[rho]
                reducedParameters=concatenate((coals1[indexOfFirstNonZero:],coals2[indexOfFirstNonZero:],migs1[indexOfFirstNonZero:],migs2[indexOfFirstNonZero:],[rho]))
#                 print "reducedParameters ",reducedParameters
#                 print "original breakpoints", br
#                 print "reduced breakpoints", br[indexOfFirstNonZeroMeasuredInBreakPoints:]
#                 print "reduced parameters", reducedParameters
#                 print "intervals", self.intervals[indexOfFirstNonZero:]
#                 print "offset", br[indexOfFirstNonZeroMeasuredInBreakPoints]
#                 print "postponing", indexOfFirstNonZeroMeasuredInBreakPoints
                emission_probs=emission_matrix6(br[indexOfFirstNonZeroMeasuredInBreakPoints:], reducedParameters, self.intervals[indexOfFirstNonZero:], ctmc_system, offset=0,ctmc_postpone=indexOfFirstNonZeroMeasuredInBreakPoints)
                
                ##More like a hack but here we clean up the transition matrix who have produced nans but the inital_probabilities are already okay##
                
                for i in xrange(indexOfFirstNonZeroMeasuredInBreakPoints):
                    for j in xrange(len(br)):
                        if i>=j:
                            transition_probs[i,j]=0.0
                        else:
                            transition_probs[i,j]=float(1.0)/(len(br)-i-1)
            
            else:
                emission_probs=emission_matrix6(br, parameters, self.intervals, ctmc_system,0.0)
        else:
            emission_probs=emission_matrix6(br, parameters, self.intervals, ctmc_system,0.0)
#         emission_probs = emission_matrix4(br, parameters, self.intervals, ctmc_system)
#         print "emission 4"
#         print printPyZipHMM(emission_probs)
        
#         print "emission 6"
#         print printPyZipHMM(emission_probs)
#         strToWirte=strToWirte+str("4:")+printPyZipHMM(emission_probs)+"\n"+"initial_probs: "+printPyZipHMM(initial_probs)
#         emission_probs = emission_matrix3b(br, parameters, self.intervals,ctmc_system)
#         print strToWirte+str("3b:")+printPyZipHMM(emission_probs)
        return initial_probs, transition_probs, emission_probs, ctmc_system.break_points


      #strToWirte=str(parameters)+"\n"+str("3:")+printPyZipHMM(emission_probs)+"\n"

if __name__ == '__main__':
    def printPyZipHMM(Matrix):
        finalString=""
        for i in range(Matrix.getHeight()):
            for j in range(Matrix.getWidth()):
                finalString=finalString+" "+str(Matrix[i,j])
            finalString=finalString+"\n"
        return finalString
    substime_first_change=0.0005
    substime_second_change=0.0010
    substime_third_change=0.0030
    def time_modifier():
        return [(5,substime_first_change),(10,substime_second_change)]
    cd=VariableCoalAndMigrationRateAndAncestralModel(VariableCoalAndMigrationRateAndAncestralModel.INITIAL_12, intervals=[5,5,5], breaktimes=1.0,breaktail=3,time_modifier=time_modifier)
    ad= cd.build_hidden_markov_model(array([1000,1000,1000,  1000,1000,1000,    0,500,0,    0,100,0,    0.40]))
    print printPyZipHMM(ad[0])
    with open("/home/svendvn/Dropbox/Bioinformatik/transition_matrix.txt", 'w') as f:
        f.write(printPyZipHMM(ad[1]))
    with open("/home/svendvn/Dropbox/Bioinformatik/emission_matrix.txt", 'w') as f:
        f.write(printPyZipHMM(ad[2]))
        
    
