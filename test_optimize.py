
from IMCoalHMM.isolation_model import IsolationModel, MinimizeWrapper
from IMCoalHMM.likelihood import Likelihood, maximum_likelihood_estimate
from pyZipHMM import Forwarder

no_states = 5
Ne = 20000
u = 1e-9
g = 25

init_split_time = 5e6 * u            # 5 mya in substitutions
init_coal_rate = 1.0/(2*Ne*u*g)      # 1/(theta/2) in substitutions
init_recomb_rate = 0.01/1e6 / (g*u)  # 1 cM/Mb in substitutions

forwarder = Forwarder.fromDirectory('examples/example_data.ziphmm')
logL = Likelihood(IsolationModel(), forwarder)

mle_split_time, mle_coal_rate, mle_recomb_rate = \
    maximum_likelihood_estimate(MinimizeWrapper(logL, no_states),
                                (init_split_time, init_coal_rate, init_recomb_rate))
maxL = logL(no_states, mle_split_time, mle_coal_rate, mle_recomb_rate)

print mle_split_time, mle_coal_rate, mle_recomb_rate, maxL

