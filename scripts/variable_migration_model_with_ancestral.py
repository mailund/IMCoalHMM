"""Code for constructing and optimizing the HMM for a model with variable
migration and coalescence.
"""

from numpy import zeros, matrix, identity,cumsum,array, concatenate, ndarray
from math import isnan, exp, log
from IMCoalHMM.state_spaces import Migration, make_rates_table_migration, make_rates_table_single, Single
from CTMC2 import make_ctmc
from IMCoalHMM.transitions import CTMCSystem, compute_upto, compute_between, compute_transition_probabilities, projection_matrix
from break_points2 import psmc_break_points, uniform_break_points, gamma_break_points
from IMCoalHMM.emissions import coalescence_points
from IMCoalHMM.model import Model
from emissions2 import emission_matrix3,emission_matrix4, emission_matrix3b, emission_matrix5, emission_matrix6, emission_matrix7,emission_matrix8
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

    def __init__(self, initial_configuration, intervals, breaktimes, breaktail=0, time_modifier=None, outgroup=False, time_point_threshold=0.02):
        self.breaktimes=breaktimes
        self.breaktail=breaktail
        self.time_modifier=time_modifier
        self.outgroup=outgroup
        self.time_point_threshold=time_point_threshold
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
        if no_epochs*4+1<len(parameters) and self.time_modifier is not None:
#             if self.outgroup:
#                 fixed_time_points=self.time_modifier(parameters[(len(coal_rates_1)*4+1):-1])    #last variable is reserved for the outgroup parameter 
#             else:
            fixed_time_points=self.time_modifier(parameters[(len(coal_rates_1)*4+1):])
        elif self.time_modifier is not None:
            fixed_time_points=self.time_modifier()
        else:
            fixed_time_points=[]
        ans=coal_rates_1, coal_rates_2, mig_rates_12, mig_rates_21, recomb_rate, fixed_time_points
#         if self.outgroup:
#             ans.append(parameters[-1])
        return ans

    def _map_rates_to_intervals(self, coal_rates):
        """Takes the coalescence rates as specified when building the CTMC
        and maps them to each interval based on the intervals specification."""
        assert len(coal_rates) == len(self.intervals)
        interval_rates = []
        for epoch, coal_rate in enumerate(coal_rates):
            for _ in xrange(self.intervals[epoch]):
                interval_rates.append(coal_rate)
        return interval_rates
    
    def valid_parameters(self, parameters):
        """Predicate testing if a given parameter point is valid for the model.
        :param parameters: Model specific parameters
        :type parameters: numpy.ndarray
        :returns: True if all parameters are valid, otherwise False
        :rtype: bool
        """
      
        assert isinstance(parameters, ndarray)
        
        if not all(parameters >= 0):  # This is the default test, useful for most models.
            return False
        if not all(parameters <=1e7): #to avoid the worst outliers
            return False
        
        if self.outgroup:
            parameters=parameters[:-1]
            
        coal_rates_1, coal_rates_2, mig_rates_12, mig_rates_21, recomb_rate,fixed_time_points = self.unpack_parameters(parameters)
        
        #Here we check if all fixed_time_points leave a positive gap between them
        for i,ti in fixed_time_points:
            if ti>self.time_point_threshold:
                return False
            for j,tj in fixed_time_points:
                if i>j and ti<=tj:
                    return False
                if j>i and tj<=ti:
                    return False
        return True
        
        

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
        if self.outgroup: # if break_points is not suitable with the outgroup size, new breakpoints will be made.
            if break_points[-1]>self.outmax:
                #print "Breakpoints redone to match outgroup"
                fixed_time_points.append((len(break_points)-1, self.outmax*9.0/9.5))
                break_points=gamma_break_points(self.no_states,beta1=0.001*self.breaktimes,alpha=2,beta2=0.001333333*self.breaktimes, tenthsInTheEnd=self.breaktail, fixed_time_points=fixed_time_points)
        #break_points = uniform_break_points(self.no_states,0,self.tmax*1e-9)

        return VariableCoalAndMigrationRateAndAncestralCTMCSystem(self.initial_state, sum(self.intervals[:-1]), ctmcs, break_points)
    
    #override for trying out special things
    def build_hidden_markov_model(self, parameters):
        """Build the hidden Markov model matrices from the model-specific parameters."""

        #checking for an outgroup:

        if self.outgroup:
            outgroup=parameters[-1]
            parameters=parameters[:-1]
            self.outmax=outgroup

        ctmc_system = self.build_ctmc_system(*parameters)

        try:
            initial_probs, transition_probs = compute_transition_probabilities(ctmc_system) #this code might throw a runtimeerror because NaNs are produced. If they are produced, they should be fixed later.
        except AssertionError:
            print "ASSERTION ERROR",parameters
            print "break_points", ctmc_system.break_points
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
                if self.outgroup:
                    emission_probs=emission_matrix8(br[indexOfFirstNonZeroMeasuredInBreakPoints:], reducedParameters, outgroup, self.intervals[indexOfFirstNonZero:], 
                                                    ctmc_system, offset=0,ctmc_postpone=indexOfFirstNonZeroMeasuredInBreakPoints)
                else:
                    emission_probs=emission_matrix7(br[indexOfFirstNonZeroMeasuredInBreakPoints:], reducedParameters, self.intervals[indexOfFirstNonZero:], 
                                                    ctmc_system, offset=0,ctmc_postpone=indexOfFirstNonZeroMeasuredInBreakPoints)
#                 emission_probs=emission_matrix7(br[indexOfFirstNonZeroMeasuredInBreakPoints:], reducedParameters, self.intervals[indexOfFirstNonZero:], 
#                                                 ctmc_system, offset=0,ctmc_postpone=indexOfFirstNonZeroMeasuredInBreakPoints)
                
                ##More like a hack but here we clean up the transition matrix who have produced nans but the inital_probabilities are already okay##
                
                for i in xrange(indexOfFirstNonZeroMeasuredInBreakPoints):
                    for j in xrange(len(br)):
                        if i>=j:
                            transition_probs[i,j]=0.0
                        else:
                            transition_probs[i,j]=float(1.0)/(len(br)-i-1)
            
            else:
                if self.outgroup:
                    emission_probs=emission_matrix8(br, parameters, outgroup, self.intervals, ctmc_system,0.0)
                else:
                    emission_probs=emission_matrix7(br, parameters, self.intervals, ctmc_system,0.0)
        else:
            if self.outgroup:
                emission_probs=emission_matrix8(br, parameters, outgroup, self.intervals, ctmc_system,0.0)
            else:
                emission_probs=emission_matrix7(br, parameters, self.intervals, ctmc_system,0.0)
#         emission_probs = emission_matrix4(br, parameters, self.intervals, ctmc_system)
#         print "emission 4"
#         print printPyZipHMM(emission_probs)
        
#         print "emission 6"
#         print printPyZipHMM(emission_probs)
#         strToWirte=strToWirte+str("4:")+printPyZipHMM(emission_probs)+"\n"+"initial_probs: "+printPyZipHMM(initial_probs)
#         emission_probs = emission_matrix3b(br, parameters, self.intervals,ctmc_system)
#         print strToWirte+str("3b:")+printPyZipHMM(emission_probs)

#         def printPyZipHMM(Matrix):
#             finalString=""
#             for i in range(Matrix.getHeight()):
#                 for j in range(Matrix.getWidth()):
#                     finalString=finalString+" "+str(Matrix[i,j])
#                 finalString=finalString+"\n"
#             print finalString
#         
#         printPyZipHMM(initial_probs)
#         printPyZipHMM(transition_probs)
#         printPyZipHMM(emission_probs)

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
    substime_first_change=0.0005*0.21
    substime_second_change=0.0010*0.21
    substime_third_change=0.0030
    def time_modifier(a):
        return [(5,substime_first_change*a[0]),(10,substime_second_change*a[1])]
    cd=VariableCoalAndMigrationRateAndAncestralModel(VariableCoalAndMigrationRateAndAncestralModel.INITIAL_12, intervals=[5,5,5], breaktimes=1.0,breaktail=3,time_modifier=time_modifier)
    cd11=VariableCoalAndMigrationRateAndAncestralModel(VariableCoalAndMigrationRateAndAncestralModel.INITIAL_11, intervals=[5,5,5], breaktimes=1.0,breaktail=3,time_modifier=time_modifier)
    cd22=VariableCoalAndMigrationRateAndAncestralModel(VariableCoalAndMigrationRateAndAncestralModel.INITIAL_22, intervals=[5,5,5], breaktimes=1.0,breaktail=3,time_modifier=time_modifier)

    param2=[3.14536287e+02 ,  3.14536287e+02 ,  3.14536287e+02 ,  3.14536287e+02, 3.14536287e+02,   3.14536287e+02,
       0.00000000e+00,   6.37982897e+03, 0.00000000e+00,   0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00,
   2.31825062e-01, 6.23481741e+00,   5.51867733e+00]
    param3=[2.62858654e+01,   2.62858654e+01,   2.62858654e+01,   2.62858654e+01,
   2.62858654e+01,   2.62858654e+01,   0.00000000e+00,   5.75033247e+03,
   0.00000000e+00,   0.00000000e+00  , 0.00000000e+00,   0.00000000e+00,
   5.55331709e-01,   2.16680768e+00,   8.03514202e+00]
    param4=[2.05335908e+03,   2.05335908e+03 ,  2.05335908e+03 ,  2.05335908e+03,
   2.05335908e+03,   2.05335908e+03 ,  0.00000000e+00  , 1.67634371e+02,
   0.00000000e+00  , 0.00000000e+00 ,  0.00000000e+00  , 0.00000000e+00,
   4.82299596e-01 , 6.69539765e-01 ,  3.15441969e+00]
    param5=[3.88861162e+05 ,  3.88861162e+05  , 3.88861162e+05,   3.88861162e+05,
   3.88861162e+05 ,  3.88861162e+05  , 0.00000000e+00,  7.87207757e+07,
   0.00000000e+00 ,  0.00000000e+00  , 0.00000000e+00,   0.00000000e+00,
   1.00000000e+00 ,  5.57723628e+00 ,  1.00000000e+01]
    param6=[5.64218998e+13 ,  5.64218998e+13,   5.64218998e+13,   5.64218998e+13,
   5.64218998e+13,   5.64218998e+13 ,  0.00000000e+00,   5.67125841e-06,
   0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00  , 0.00000000e+00,
   6.09890011e-01,   3.16923726e+00,   2.83031370e+01]
    param7=[8.35908133e+12 ,  8.35908133e+12,   8.35908133e+12 ,  8.35908133e+12,
   8.35908133e+12,   8.35908133e+12 ,  0.00000000e+00 ,  4.30441209e+36,
   0.00000000e+00,   0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00,
   4.72465675e+00 ,  1.73293016e+01 ,  3.32063571e+01]
    param8=[5.52249123e+02 ,  5.52249123e+02,   5.52249123e+02,   5.52249123e+02,
   5.52249123e+02,   5.52249123e+02,   0.00000000e+00,   1.55351268e+04,
   0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00,
   1.39737743e-01,   4.06228914e+00  , 1.02141848e+01]
    param9=[3.35926895e+02,   3.35926895e+02  , 3.35926895e+02,   3.35926895e+02,
   3.35926895e+02,  3.35926895e+02  , 0.00000000e+00,   3.77618998e+00,
   0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00,   0.00000000e+00,
   4.28667554e-02,  1.11438715e+01 ,  6.95000151e+00]
    param10=[39.79413056 , 39.79413056  ,39.79413056,  39.79413056 , 39.79413056,
  39.79413056,   0.    ,       0.42004685  , 0.   ,        0. ,          0.       ,    0.,
   0.30270784 ,  6.46620357 ,  6.03938501]
    param11=[ 9.09855530e+03 ,  9.09855530e+03 ,  9.09855530e+03 ,  9.09855530e+03,
   9.09855530e+03   ,9.09855530e+03 ,  0.00000000e+00 ,  5.48347909e-21,
   0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 , 0.00000000e+00,
   7.44164128e-01 ,  3.64138055e+01 ,  5.77346971e+01]
    param12=[5.99735288e+009,   5.99735288e+009,   5.99735288e+009 ,  5.99735288e+009,
   5.99735288e+009  , 5.99735288e+009 ,  0.00000000e+000 ,  5.86978280e-147,
   0.00000000e+000 ,  0.00000000e+000 ,  0.00000000e+000 ,  0.00000000e+000,
   4.34518694e+000 ,  2.10788844e+002 ,  3.75838496e+002]
    param13=[399.15988633 , 399.15988633 , 399.15988633,  399.15988633 , 399.15988633,
  399.15988633 ,   0.      ,    102.79775413  ,  0.     ,       0.  ,          0.,
    0.      ,      0.43878485 ,   4.11523575  ,  4.40175474]
    param14=[1.58608288e+02  , 1.58608288e+02  , 1.58608288e+02  , 1.58608288e+02,
   1.58608288e+02 ,  1.58608288e+02,   0.00000000e+00,   1.48044094e+03,
   0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00,
   4.49882939e-01 ,  7.46258111e+00 ,  5.26952249e+00]
    param15=[7.97457501e-04  , 7.97457501e-04  , 7.97457501e-04  , 7.97457501e-04,
   7.97457501e-04  , 7.97457501e-04   ,0.00000000e+00 , 9.74844240e+06,
   0.00000000e+00  , 0.00000000e+00,  0.00000000e+00 ,  0.00000000e+00,
   1.97190635e+00 ,  2.35674926e+01  , 2.26254669e+01]
    param16=[6.40656479e+02  , 6.40656479e+02 ,  6.40656479e+02 ,  6.40656479e+02,
   6.40656479e+02  , 6.40656479e+02,   0.00000000e+00,   1.31333975e+01,
   0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00,
   7.84365671e-02 ,  1.41428046e+00 ,  4.26427519e+00]
    param17=[8.38656218e+02 ,  8.38656218e+02 ,  8.38656218e+02 ,  8.38656218e+02,
   8.38656218e+02 ,  8.38656218e+02  , 0.00000000e+00   ,3.39907010e+03,
   0.00000000e+00 ,  0.00000000e+00  , 0.00000000e+00  , 0.00000000e+00,
   3.82257625e-01 ,  7.37601871e+00  , 6.52240101e+00]
    
    def log_transformfunc(fro,to):
        def transform(num):
            print num*log(to/fro)+log(fro)
            return exp(num*log(to/fro)+log(fro))
        return transform
    def linear_transformfunc(scale,offset=0):    
        def transform(num):
            return num*scale+offset
        return transform
    coal=log_transformfunc(1.0, 10000.0)
    mig=log_transformfunc(1.0, 10000.0)
    rho=linear_transformfunc(1.0)
    time=linear_transformfunc(10.0)
    
    param18mark=[60.96492709925478, 87.5066394820901, 65.29959132760096, 41.80847054348992, -0.4570658194957846]
    param18=[coal(param18mark[0])]*6+[0.0]*3+[0.0]+[mig(param18mark[1])]+[0.0]+[rho(param18mark)]+[time(i) for i in param18mark[-2:]]
    
    param=array(param18)
    
    print "valid parameters", cd.valid_parameters(array(param))

    print substime_first_change*param[-2]
    print substime_second_change*param[-1]
    ad= cd.build_hidden_markov_model(param)
# 
    print printPyZipHMM(ad[0])
    print printPyZipHMM(ad[2])
    print ad[3]

    if True:    
        from likelihood2 import Likelihood
        from pyZipHMM import Forwarder
        
        pathToSim="/home/svendvn/IMCoalHMM-simulations.10977"
        
        a11s=[pathToSim + "/alignment."+ s+".ziphmm" for s in ["1.11"]]
        a12s=[pathToSim + "/alignment."+ s+".ziphmm" for s in ["1.12"]]
        a22s=[pathToSim + "/alignment."+ s+".ziphmm" for s in ["1.22"]]
        
        forwarders11=[Forwarder.fromDirectory(a11) for a11 in a11s]
        forwarders12=[Forwarder.fromDirectory(a12) for a12 in a12s]
        forwarders22=[Forwarder.fromDirectory(a22) for a22 in a22s]
        likeli11=Likelihood(cd11, forwarders11)
        likeli12=Likelihood(cd, forwarders12)
        likeli22=Likelihood(cd22, forwarders22)
        
        print likeli11(param)
        print likeli12(param)
        print likeli22(param)

    with open("/home/svendvn/Dropbox/Bioinformatik/transition_matrix.txt", 'w') as f:
        f.write(printPyZipHMM(ad[1]))
    with open("/home/svendvn/Dropbox/Bioinformatik/emission_matrix.txt", 'w') as f:
        f.write(printPyZipHMM(ad[2]))
        
    
