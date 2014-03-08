"""Code for constructing CTMCs and computing transition probabilities
in them."""

from numpy import zeros
from scipy import matrix
from scipy.linalg import expm


class CTMC(object):
    """Class representing the CTMC for the back-in-time coalescent."""

    def __init__(self, state_space, rates_table):
        """Create the CTMC based on a state space and a mapping
        from transition labels to rates.

        :param state_space: The state space the CTMC is over.
        :type state_space: IMCoalHMM.CoalSystem
        :param rates_table: A table where transition rates can
         be looked up.
        :type rates_table: dict
        """

        # Remember this, just to decouple state space from CTMC
        # in other parts of the code...
        self.state_space = state_space

        # noinspection PyCallingNonCallable
        self.rate_matrix = matrix(zeros((len(state_space.states),
                                         len(state_space.states))))

        for src, trans, dst in state_space.transitions:
            self.rate_matrix[src, dst] = rates_table[trans]

        for i in xrange(len(state_space.states)):
            self.rate_matrix[i, i] = - self.rate_matrix[i, :].sum()

        self.prob_matrix_cache = dict()

    def probability_matrix(self, delta_t):
        """Computes the transition probability matrix for a
        time period of delta_t.

        :param delta_t: The time period the CTMC should run for.
        :type delta_t: float

        :returns: The probability transition matrix
        :rtype: matrix
        """
        if not delta_t in self.prob_matrix_cache:
            self.prob_matrix_cache[delta_t] = expm(self.rate_matrix * delta_t)
        return self.prob_matrix_cache[delta_t]


# We cache the CTMCs because in the optimisations, especially the models with a large number
# of parameters, we are creating the same CTMCs again and again and computing the probability
# transition matrices is where we spend most of the time.
from cache import Cache
CTMC_CACHE = Cache()


def make_ctmc(state_space, rates_table):
    """Create the CTMC based on a state space and a mapping
    from transition labels to rates.

    :param state_space: The state space the CTMC is over.
    :type state_space: IMCoalHMM.CoalSystem
    :param rates_table: A table where transition rates can be looked up.
    :type rates_table: dict
    """
    cache_key = (state_space, tuple(rates_table.items()))
    if not cache_key in CTMC_CACHE:
        CTMC_CACHE[cache_key] = CTMC(state_space, rates_table)
    return CTMC_CACHE[cache_key]
