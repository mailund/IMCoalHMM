
from IMCoalHMM.isolation_model import IsolationModel, MinimizeWrapper
from IMCoalHMM.likelihood import Likelihood, maximum_likelihood_estimate
from pyZipHMM import Forwarder

no_states = 5
Ne = 20000
u = 1e-9
g = 25

split_time = 5e6 * u            # 5 mya in substitutions
coal_rate = 1.0/(2*Ne*u*g)      # 1/(theta/2) in substitutions
recomb_rate = 0.01/1e6 / (g*u)  # 1 cM/Mb in substitutions

forwarder = Forwarder.fromDirectory('examples/example_data.ziphmm')
logL = Likelihood(IsolationModel(), forwarder)

split_time, coal_rate, recomb_rate = \
    maximum_likelihood_estimate(MinimizeWrapper(logL, no_states),
                                (split_time, coal_rate, recomb_rate))
maxL = logL(no_states, split_time, coal_rate, recomb_rate)

print split_time, coal_rate, recomb_rate, maxL

