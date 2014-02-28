
from IMCoalHMM.isolation_model import IsolationModel
from IMCoalHMM.likelihood import Likelihood
from pyZipHMM import Forwarder

model = IsolationModel()
forwarder = Forwarder.fromDirectory('examples/example_data.ziphmm')
logL = Likelihood(model, forwarder)

no_states = 10
Ne = 20000
u = 1e-9
g = 25

split_time = 5e6 * u            # 5 mya in substitutions
coal_rate = 1.0/(2*Ne*u*g)      # 1/(theta/2)
recomb_rate = 0.01/1e6 / (g*u)  # 1 cM/Mb in substitutions

from scipy.optimize import fmin

class Minimizer(object):
    def __init__(self, logL, no_states):
        self.logL = logL
        self.no_states = no_states
        
    def __call__(self, parameters):
        if min(parameters) <= 0:
            return 1e18 # fixme: return infinity
        return -self.logL(self.no_states, *parameters)


foobar = Minimizer(logL, 5)
x = fmin(foobar, (split_time, coal_rate, recomb_rate))
print x