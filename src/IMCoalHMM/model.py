"""
Abstract class for demographic models.
"""

from abc import ABCMeta, abstractmethod
from IMCoalHMM.transitions import compute_transition_probabilities
from IMCoalHMM.emissions import emission_matrix


class Model(object):
    """Abstract class for demographic models. Responsible for building the state space of the model,
    the continuous time Markov chains and the hidden Markov model matrices.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_ctmc_system(self, *parameters):
        """Build the CTMC system from the model-specific parameters."""
        pass

    @abstractmethod
    def emission_points(self, *parameters):
        """Build the time points to emit from using the model-specific parameters."""
        pass

    def build_hidden_markov_model(self, *parameters):
        """Build the hidden Markov model matrices from the model-specific parameters."""
        ctmc_system = self.build_ctmc_system(*parameters)
        initial_probs, transition_probs = compute_transition_probabilities(ctmc_system)
        emission_probs = emission_matrix(self.emission_points(*parameters))

        return initial_probs, transition_probs, emission_probs
